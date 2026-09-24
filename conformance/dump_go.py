"""Dump Go (19x19) positions from pgx for the Rust conformance test.

Run in the pgx environment (it imports pgx; it lives here so the pgx repo is untouched):

    JAX_PLATFORMS=cpu python conformance/dump_go.py

Go is pgx's `go_19x19` env. Writes two files next to this script:

  go.jsonl.gz    for each of a set of games, a `game` line with the pgx actions played, then one
                 line per position reached: a digest of the position, its outcome, and unless it
                 is over, its legal actions (as a bitmask), a combined digest of every successor
                 and each successor's outcome.
  go_perft.json  move-tree node counts from the start (depths 1 to 3) and depth 2 from a few
                 later positions.

A move is one pgx action: a point `0..=360`, or `361` to pass. Positions are 361 points a move
from the start, so the file stores each game once, as its actions, and the Rust test replays them.
A successor is only checked through the combined digest and its outcome; on a mismatch, rerun this
for the one position to see which.

What the games have to reach, and how:

- Captures, suicide and ko: random play on a crowded board does all three, often. `capture`
  prefers capturing moves, which makes more of them.
- The 722-move cap: `no_pass` only passes when nothing else is legal.
- Two passes, scored: `passy` plays 100 random stones and then passes, so the board is still
  mostly open areas, which touch both colours and count for neither.
- Positional superko, which ends the game and loses it for the mover. Random play gets there
  only now and then, and not reliably for both sides, so two scripted games build two kos and
  cycle them until the position repeats, once with black making the repeat and once with white.
- Area scoring's corner cases: `pass, pass` on the empty board (361 each; komi decides), and one
  black stone then two passes (all 361 are black's).

Regenerate whenever pgx's Go rules change, and commit the result.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pgx  # noqa: E402
from pgx._src.games.go import Game, GameState  # noqa: E402

SIZE = 19
POINTS = SIZE * SIZE
PASS = POINTS
GAME = Game(size=SIZE, komi=7.5)
HERE = Path(__file__).resolve().parent

_mask = jax.jit(jax.vmap(GAME.legal_action_mask))
_step = jax.jit(jax.vmap(GAME.step))
_terminal = jax.jit(jax.vmap(GAME.is_terminal))
_rewards = jax.jit(jax.vmap(GAME.rewards))


def _stack(state: GameState) -> GameState:
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], state)


def _row(batch: GameState, i: int) -> GameState:
    return jax.tree_util.tree_map(lambda x: x[i], batch)


def all_moves(state: GameState) -> tuple[np.ndarray, GameState]:
    """Every legal action from a state, ascending, and the batch of successors in the same order,
    padded to 512 (a single jit shape) by repeating the first."""
    one = _stack(state)
    actions = np.flatnonzero(np.asarray(_mask(one)[0]))
    padded = np.concatenate([actions, np.full(512 - len(actions), actions[0])])
    batch = jax.tree_util.tree_map(lambda x: jnp.repeat(x, len(padded), axis=0), one)
    return actions, _step(batch, jnp.asarray(padded, dtype=jnp.int32))


def fnv(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def digests(boards: np.ndarray, colors: np.ndarray, kos: np.ndarray) -> list[str]:
    """FNV-1a (32-bit) of each state's wire fields as bytes: the 361 points as colour + 1 (0 white,
    1 empty, 2 black), `to_move`, then the ko point + 1 as two bytes, low first (0 for none).
    Vectorised over states: FNV is sequential in the bytes, not across states. Mirrored in the
    Rust test."""
    data = np.concatenate(
        [
            np.sign(boards).astype(np.int64) + 1,
            colors.astype(np.int64)[:, None],
            ((kos.astype(np.int64) + 1) & 0xFF)[:, None],
            ((kos.astype(np.int64) + 1) >> 8)[:, None],
        ],
        axis=1,
    )
    h = np.full(len(data), 0x811C9DC5, dtype=np.uint64)
    for j in range(data.shape[1]):
        h = ((h ^ data[:, j].astype(np.uint64)) * np.uint64(0x01000193)) & np.uint64(0xFFFFFFFF)
    return [f"{int(v):08x}" for v in h]


def digest_of(state: GameState) -> str:
    return digests(
        np.asarray(state.board)[None], np.asarray(state.color)[None], np.asarray(state.ko)[None]
    )[0]


def outcome_code(terminal: bool, rewards: np.ndarray) -> str:
    """'-' ongoing, '0' black won, '1' white won, 'd' drawn (never: komi is 7.5). pgx's game-level
    rewards are indexed by colour, and colour 0 (black) moves first."""
    if not terminal:
        return "-"
    if float(rewards[0]) == float(rewards[1]):
        return "d"
    return str(int(np.argmax(rewards)))


def outcome_of(state: GameState) -> str:
    one = _stack(state)
    return outcome_code(bool(_terminal(one)[0]), np.asarray(_rewards(one)[0]))


def mask_hex(actions: np.ndarray) -> str:
    """Legal actions as a bitmask: bit `a % 8` of byte `a // 8` is action `a`, bytes in hex."""
    bits = np.zeros(8 * ((POINTS + 1 + 7) // 8), dtype=np.uint8)
    bits[actions] = 1
    return np.packbits(bits, bitorder="little").tobytes().hex()


def point(name: str) -> int:
    """`D4` to a point: columns A-T without I, rows 1-19; row 1 is points 0..=18."""
    if name == "pass":
        return PASS
    col = "ABCDEFGHJKLMNOPQRST".index(name[0])
    return (int(name[1:]) - 1) * SIZE + col


def ko_frame(row: int, col: int) -> tuple[list[str], list[str], str, str]:
    """A ko shape with its two points at (row, col) and (row, col + 1): black's three stones
    around the left one, white's three around the right. Returns (black, white, left, right)."""

    def at(r: int, c: int) -> str:
        return f"{'ABCDEFGHJKLMNOPQRST'[c]}{r + 1}"

    black = [at(row + 1, col), at(row - 1, col), at(row, col - 1)]
    white = [at(row + 1, col + 1), at(row - 1, col + 1), at(row, col + 2)]
    return black, white, at(row, col), at(row, col + 1)


def double_ko(white_repeats: bool) -> list[int]:
    """Two kos, the first with a white stone to capture and the second with a black one, then the
    cycle that brings the position back: a capture each, a pass (which lifts the ban), a capture
    each. The last capture repeats a position and loses. With `white_repeats`, black passes first
    so that the roles are swapped."""
    b1, w1, left1, right1 = ko_frame(4, 4)
    b2, w2, left2, right2 = ko_frame(12, 12)
    black, white = b1 + b2 + [right2], w1 + w2 + [left1]
    setup = [m for pair in zip(black, white, strict=True) for m in pair]
    if white_repeats:
        cycle = ["pass", left2, right1, "pass", right2, left1]
    else:
        cycle = [right1, left2, "pass", left1, right2]
    return [point(m) for m in setup + cycle]


SCRIPTED = {
    "black_superko": double_ko(white_repeats=False),
    "white_superko": double_ko(white_repeats=True),
    "empty_board": [PASS, PASS],
    "one_stone": [point("K10"), PASS, PASS],
}
POLICIES = ("uniform", "capture", "no_pass", "passy")


def pick(
    policy: str, ply: int, rng: random.Random, actions: np.ndarray, captures: np.ndarray
) -> int:
    """Index into `actions` of the move to play at move `ply`."""
    n = len(actions)
    others = [i for i in range(n) if actions[i] != PASS]
    if policy == "capture":
        taking = [i for i in others if captures[i]]
        if taking and rng.random() < 0.8:
            return rng.choice(taking)
    if policy == "no_pass" and others:
        return rng.choice(others)
    if policy == "passy":
        return n - 1 if ply >= 100 else rng.choice(others)  # pass is always legal, and last
    return rng.randrange(n)


def play_game(g: int, policy: str, script: list[int], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed * 1000 + g)
    state = GAME.init()
    played: list[int] = []
    records: list[dict[str, Any]] = []
    while True:
        record: dict[str, Any] = {
            "ply": len(played),
            "digest": digest_of(state),
            "outcome": outcome_of(state),
        }
        records.append(record)
        if record["outcome"] != "-":
            break
        actions, succ = all_moves(state)
        n = len(actions)
        succ_terminal = np.asarray(_terminal(succ))[:n]
        succ_rewards = np.asarray(_rewards(succ))[:n]
        record["legal"] = mask_hex(actions)
        each = digests(
            np.asarray(succ.board)[:n], np.asarray(succ.color)[:n], np.asarray(succ.ko)[:n]
        )
        record["successors"] = f"{fnv(''.join(each).encode()):08x}"
        record["successor_outcomes"] = "".join(
            outcome_code(bool(succ_terminal[i]), succ_rewards[i]) for i in range(n)
        )
        if len(played) < len(script):
            i = int(np.flatnonzero(actions == script[len(played)])[0])
        else:
            captures = np.asarray(succ.num_captured).sum(axis=1)[:n] > int(
                np.asarray(state.num_captured).sum()
            )
            i = pick(policy, len(played), rng, actions, captures)
        played.append(int(actions[i]))
        state = _row(succ, i)
    print(
        f"game {g} ({policy}): {len(played)} moves, ended {records[-1]['outcome']!r}",
        file=sys.stderr,
        flush=True,
    )
    return [{"kind": "game", "game": g, "policy": policy, "actions": played}] + [
        {"game": g, **r} for r in records
    ]


def perft(state: GameState, depth: int) -> int:
    if depth == 0:
        return 1
    if bool(_terminal(_stack(state))[0]):
        return 0
    actions, succ = all_moves(state)
    n = len(actions)
    if depth == 1:
        return n
    if depth == 2:
        # pgx's mask ignores termination, so a finished successor has to be left out by hand.
        live = ~np.asarray(_terminal(succ))[:n]
        return int(np.asarray(_mask(succ))[:n][live].sum())
    return sum(perft(_row(succ, i), depth - 1) for i in range(n))


def replay(actions: list[int]) -> GameState:
    state = GAME.init()
    for a in actions:
        state = GAME.step(state, jnp.int32(a))
    return state


def pgx_provenance() -> dict[str, Any]:
    root = Path(pgx.__file__).resolve().parent.parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    env = pgx.make("go_19x19")
    return {
        "pgx_env_version": env.version,
        "pgx_commit": git("rev-parse", "HEAD"),
        "pgx_rules_dirty": bool(git("status", "--porcelain", "pgx/_src/games/go.py", "pgx/go.py")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games", type=int, default=8, help="random games, on top of the scripted")
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument(
        "--perft-cases", type=int, default=3, help="later positions to take perft(2) from"
    )
    ap.add_argument(
        "--only", choices=("positions", "perft"), help="regenerate just one of the two files"
    )
    args = ap.parse_args()

    meta = {"kind": "meta", **pgx_provenance(), "seed": args.seed, "games": args.games}
    print(json.dumps(meta), file=sys.stderr)

    positions = HERE / "go.jsonl.gz"
    if args.only != "perft":
        games = [(name, script) for name, script in SCRIPTED.items()]
        games += [(POLICIES[g % len(POLICIES)], []) for g in range(args.games)]
        with gzip.open(positions, "wt", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
            for g, (policy, script) in enumerate(games):
                for r in play_game(g, policy, script, args.seed):
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")
        print(f"wrote {len(games)} games", file=sys.stderr)
    if args.only == "positions":
        return
    with gzip.open(positions, "rt", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f][1:]
    games_played = [r["actions"] for r in lines if r.get("kind") == "game"]

    perfts: list[dict[str, Any]] = []
    start = GAME.init()
    for d in (1, 2, 3):
        perfts.append({"actions": [], "depth": d, "nodes": perft(start, d)})
        print(f"start perft({d}) = {perfts[-1]['nodes']}", file=sys.stderr, flush=True)
    # Crowded positions, where captures, suicide and ko all bite. Take them from the random games.
    for actions in [a for a in games_played if len(a) > 400][: args.perft_cases]:
        prefix = actions[:400]
        nodes = perft(replay(prefix), 2)
        perfts.append({"actions": prefix, "depth": 2, "nodes": nodes})
        print(f"perft(2) = {nodes}", file=sys.stderr, flush=True)
    (HERE / "go_perft.json").write_text(
        json.dumps({**meta, "kind": "perft", "cases": perfts}, separators=(",", ":")) + "\n"
    )
    print("wrote perft", file=sys.stderr)


if __name__ == "__main__":
    main()

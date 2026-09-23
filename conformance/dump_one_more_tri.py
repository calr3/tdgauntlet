"""Dump One More Tri positions from pgx for the Rust conformance test.

Run in the pgx environment (it imports pgx; it lives here so the pgx repo is untouched):

    JAX_PLATFORMS=cpu python conformance/dump_one_more_tri.py

One More Tri is pgx's `g_hex` env. Writes two files next to this script:

  one_more_tri.jsonl.gz    one line per position reached in a set of seeded games. Each carries
                           the wire state, every legal move (triangle, tile), and for each move a
                           digest of the successor state and that successor's outcome.
  one_more_tri_perft.json  move-tree node counts from the start (depths 1 to 3) and depth 2 from
                           a few later positions.

A move is one pgx action, `triangle * 10 + tile - 1`, so the moves of a position are its action
mask read off directly.

Every game lasts exactly 20 moves, so random play reaches the end every time. What it rarely
reaches is the end where triangle 4 or 10 is the one left empty, and those two are where pgx's
adjacency is one-sided (4 counts 10's tile, 10 does not count 4's). So half the games are played
by a policy that fills everything else first, which leaves 4 and 10 as the last two empties: the
final move then decides between the two cases.

Regenerate whenever pgx's g_hex rules change, and commit the result.
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
from pgx._src.games.g_hex import Game, GameState  # noqa: E402

GAME = Game()
HERE = Path(__file__).resolve().parent
TRIANGLES = 21
TILES = 10
#: The two triangles whose adjacency is one-sided in pgx.
LOPSIDED = (4, 10)

_mask = jax.jit(jax.vmap(GAME.legal_action_mask))
_step = jax.jit(jax.vmap(GAME.step))
_terminal = jax.jit(jax.vmap(GAME.is_terminal))
_rewards = jax.jit(jax.vmap(GAME.rewards))


def _bucket(n: int) -> int:
    """Pad batches to a power of two so jit compiles a handful of shapes, not one per size."""
    size = 1
    while size < n:
        size *= 2
    return size


def _stack(state: GameState) -> GameState:
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], state)


def _row(batch: GameState, i: int) -> GameState:
    return jax.tree_util.tree_map(lambda x: x[i], batch)


def all_moves(state: GameState) -> tuple[list[tuple[int, int]], GameState]:
    """Every legal (triangle, tile) from a state, and the batch of successors in the same order."""
    one = _stack(state)
    actions = np.flatnonzero(np.asarray(_mask(one)[0]))
    if len(actions) == 0:
        return [], one
    n = len(actions)
    padded = np.concatenate([actions, np.full(_bucket(n) - n, actions[0])])
    batch = jax.tree_util.tree_map(lambda x: jnp.repeat(x, len(padded), axis=0), one)
    succ = _step(batch, jnp.asarray(padded, dtype=jnp.int32))
    moves = [(int(a) // TILES, int(a) % TILES + 1) for a in actions]
    return moves, succ


def wire(state: GameState) -> dict[str, Any]:
    return {
        "board": [int(v) for v in np.asarray(state.board)[:TRIANGLES]],
        "to_move": int(state.color),
    }


def digest(w: dict[str, Any]) -> str:
    """FNV-1a (32-bit) of a canonical rendering of a wire state. Mirrored in the Rust test."""
    text = ",".join(str(v) for v in w["board"]) + f"|{w['to_move']}"
    h = 0x811C9DC5
    for b in text.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def outcome_code(terminal: bool, rewards: np.ndarray) -> str:
    """'-' ongoing, '0' black won, '1' white won, 'd' drawn. pgx's game-level rewards are indexed
    by colour, and colour 0 (black) is the side that moves first."""
    if not terminal:
        return "-"
    if float(rewards[0]) == float(rewards[1]):
        return "d"
    return str(int(np.argmax(rewards)))


def outcome_of(state: GameState) -> str:
    one = _stack(state)
    return outcome_code(bool(_terminal(one)[0]), np.asarray(_rewards(one)[0]))


POLICIES = ("uniform", "lopsided")


def pick(policy: str, rng: random.Random, moves: list[tuple[int, int]]) -> int:
    """Index of the move to play: anything at all, or anything that keeps 4 and 10 empty while
    another triangle is free."""
    if policy == "lopsided":
        elsewhere = [i for i, (tri, _) in enumerate(moves) if tri not in LOPSIDED]
        if elsewhere:
            return rng.choice(elsewhere)
    return rng.randrange(len(moves))


def play_games(n_games: int, seed: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for g in range(n_games):
        rng = random.Random(seed * 1000 + g)
        policy = POLICIES[g % len(POLICIES)]
        state = GAME.init()
        while True:
            moves, succ = all_moves(state)
            record: dict[str, Any] = {
                "game": g,
                "policy": policy,
                "state": wire(state),
                "outcome": outcome_of(state),
            }
            if bool(_terminal(_stack(state))[0]):
                out.append(record)
                break
            n = len(moves)
            succ_terminal = np.asarray(_terminal(succ))[:n]
            succ_rewards = np.asarray(_rewards(succ))[:n]
            order = sorted(range(n), key=lambda i: moves[i])
            record["moves"] = [list(moves[i]) for i in order]
            record["successors"] = [digest(wire(_row(succ, i))) for i in order]
            record["successor_outcomes"] = "".join(
                outcome_code(bool(succ_terminal[i]), succ_rewards[i]) for i in order
            )
            out.append(record)
            if n == 0:
                break
            state = _row(succ, pick(policy, rng, moves))
        print(
            f"game {g} ({policy}): {sum(r['game'] == g for r in out)} positions, "
            f"ended {out[-1]['outcome']!r}",
            file=sys.stderr,
            flush=True,
        )
    return out


def perft(state: GameState, depth: int) -> int:
    if depth == 0:
        return 1
    if bool(_terminal(_stack(state))[0]):
        return 0
    moves, succ = all_moves(state)
    if depth == 1:
        return len(moves)
    if depth == 2:
        # A finished game's mask is all false at the game level, so this counts nothing for it.
        return int(np.asarray(_mask(succ))[: len(moves)].sum())
    return sum(perft(_row(succ, i), depth - 1) for i in range(len(moves)))


def state_from_wire(w: dict[str, Any]) -> GameState:
    board = np.zeros(TRIANGLES + 1, dtype=np.int32)
    board[:TRIANGLES] = w["board"]
    tiles = np.ones((2, TILES), dtype=bool)
    for v in w["board"]:
        if v:
            tiles[int(v < 0), abs(v) - 1] = False
    return GameState(
        board=jnp.asarray(board),
        tiles=jnp.asarray(tiles),
        color=jnp.int32(w["to_move"]),
        winner=jnp.int32(-1),
    )


def pgx_provenance() -> dict[str, Any]:
    root = Path(pgx.__file__).resolve().parent.parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    env = pgx.make("g_hex")
    return {
        "pgx_env_version": env.version,
        "pgx_commit": git("rev-parse", "HEAD"),
        "pgx_rules_dirty": bool(git("status", "--porcelain", "pgx/_src/games/g_hex.py")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20262)
    ap.add_argument(
        "--perft-cases", type=int, default=3, help="later positions to take perft(2) from"
    )
    ap.add_argument(
        "--only", choices=("positions", "perft"), help="regenerate just one of the two files"
    )
    args = ap.parse_args()

    meta = {"kind": "meta", **pgx_provenance(), "seed": args.seed, "games": args.games}
    print(json.dumps(meta), file=sys.stderr)

    positions = HERE / "one_more_tri.jsonl.gz"
    if args.only != "perft":
        records = play_games(args.games, args.seed)
        with gzip.open(positions, "wt", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
            for r in records:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        print(f"wrote {len(records)} positions", file=sys.stderr)
    if args.only == "positions":
        return
    with gzip.open(positions, "rt", encoding="utf-8") as f:
        records = [json.loads(line) for line in f][1:]

    perfts: list[dict[str, Any]] = []
    start = GAME.init()
    for d in (1, 2, 3):
        perfts.append({"state": wire(start), "depth": d, "nodes": perft(start, d)})
        print(f"start perft({d}) = {perfts[-1]['nodes']}", file=sys.stderr, flush=True)
    # Positions a few moves from the end, where perft reaches the result.
    late = [r for r in records if r["outcome"] == "-" and 15 <= sum(map(bool, r["state"]["board"]))]
    for r in late[: args.perft_cases]:
        nodes = perft(state_from_wire(r["state"]), 2)
        perfts.append({"state": r["state"], "depth": 2, "nodes": nodes})
        print(f"perft(2) = {nodes}", file=sys.stderr, flush=True)
    (HERE / "one_more_tri_perft.json").write_text(
        json.dumps({**meta, "kind": "perft", "cases": perfts}, indent=1) + "\n"
    )
    print("wrote perft", file=sys.stderr)


if __name__ == "__main__":
    main()

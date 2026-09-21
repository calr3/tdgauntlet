"""Dump Gess positions from pgx for the Rust conformance test.

Run in the pgx environment (it imports pgx; it lives here so the pgx repo is untouched):

    JAX_PLATFORMS=cpu python conformance/dump_gess.py

Writes two files next to this script:

  gess.jsonl.gz    one line per position reached in a set of seeded games. Each carries the wire
                   state, every complete legal move (source centre, destination centre), and for
                   each move a digest of the successor state and that successor's outcome.
  gess_perft.json  move-tree node counts from the start (depths 1 and 2, the second of which
                   takes a few minutes) and depth 2 from a few later positions.

pgx plays a move as two actions, so the complete moves of a position are found by walking its
action masks: every legal source, then every legal destination for that source.

pgx's state has no move counter, so this script keeps its own and puts it in the wire state, which
is where tdgauntlet carries it.

Regenerate whenever pgx's Gess rules change, and commit the result.
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
from pgx._src.games.gess import Game, GameState  # noqa: E402

GAME = Game()
HERE = Path(__file__).resolve().parent

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


def _take(batch: GameState, idx: np.ndarray) -> GameState:
    """Rows `idx` of a batched state, padded (by repeating row 0 of idx) to a bucket size."""
    n = len(idx)
    padded = np.concatenate([idx, np.full(_bucket(n) - n, idx[0] if n else 0)])
    return jax.tree_util.tree_map(lambda x: x[padded], batch)


def _stack(state: GameState) -> GameState:
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], state)


def _row(batch: GameState, i: int) -> GameState:
    return jax.tree_util.tree_map(lambda x: x[i], batch)


def _pad(values: np.ndarray) -> jnp.ndarray:
    return jnp.asarray(np.pad(values, (0, _bucket(len(values)) - len(values))))


def complete_moves(state: GameState) -> tuple[list[tuple[int, int]], GameState]:
    """Every complete move from a stage-0 state, and the batch of successors in the same order."""
    one = _stack(state)
    sources = np.flatnonzero(np.asarray(_mask(one)[0]))
    if len(sources) == 0:
        return [], one

    # Stage 1: one row per source centre.
    s1 = _step(_take(one, np.zeros(len(sources), dtype=np.int64)), _pad(sources))
    dest_masks = np.asarray(_mask(s1))[: len(sources)]
    pairs = [(i, int(d)) for i in range(len(sources)) for d in np.flatnonzero(dest_masks[i])]
    if not pairs:
        return [], one

    # Stage 2: one row per complete move.
    rows = np.array([i for i, _ in pairs])
    dests = np.array([d for _, d in pairs])
    s2 = _step(_take(s1, rows), _pad(dests))
    moves = [(int(sources[i]), d) for i, d in pairs]
    return moves, s2


def wire(state: GameState, moves_played: int) -> dict[str, Any]:
    board = "".join(".BW"[int(c)] for c in np.asarray(state.board))
    return {
        "board": board,
        "to_move": int(state.color),
        "moves": moves_played,
        "quiet_moves": int(state.no_capture_turns),
    }


def digest(w: dict[str, Any]) -> str:
    """FNV-1a (32-bit) of a canonical rendering of a wire state. Mirrored in the Rust test."""
    text = f"{w['board']}|{w['to_move']}|{w['moves']}|{w['quiet_moves']}"
    h = 0x811C9DC5
    for b in text.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def outcome_code(terminal: bool, rewards: np.ndarray) -> str:
    """'-' ongoing, '0' black won, '1' white won, 'd' drawn."""
    if not terminal:
        return "-"
    if float(rewards[0]) == float(rewards[1]):
        return "d"
    return str(int(np.argmax(rewards)))


def outcome_of(state: GameState) -> str:
    one = _stack(state)
    return outcome_code(bool(_terminal(one)[0]), np.asarray(_rewards(one)[0]))


POLICIES = ("uniform", "quiet", "greedy")


def pick(
    policy: str,
    rng: random.Random,
    state: GameState,
    moves: list[tuple[int, int]],
    succ: GameState,
) -> int:
    """Index of the move to play.

    'uniform' is plain random play, which in Gess is mostly suicide: a piece taken from around
    your own ring destroys it and loses on the spot, so those games are two or three moves long
    and are the coverage for that rule. The other two only play moves that keep their own ring,
    which is what makes a game long enough to reach anything else. 'quiet' then avoids taking
    stones, so it runs into the 20-move quiet clock and the count-back that decides it there;
    'greedy' takes as many stones as it can, which is what breaks the other side's ring.
    """
    before = int((np.asarray(state.board) != 0).sum())
    after = (np.asarray(succ.board)[: len(moves)] != 0).sum(axis=1)
    taken = before - after
    # A move that leaves the mover without a ring hands the game to the opponent.
    winners = np.asarray(succ.winner)[: len(moves)]
    safe = [i for i in range(len(moves)) if winners[i] != 1 - int(state.color)]
    if policy == "uniform" or not safe:
        return rng.randrange(len(moves))
    if policy == "quiet":
        calm = [i for i in safe if taken[i] == 0]
        if calm:
            return rng.choice(calm)
    elif policy == "greedy":
        best = int(taken[safe].max())
        if best > 0 and rng.random() < 0.8:
            return rng.choice([i for i in safe if taken[i] == best])
    return rng.choice(safe)


def play_games(n_games: int, max_positions: int, seed: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for g in range(n_games):
        rng = random.Random(seed * 1000 + g)
        policy = POLICIES[g % len(POLICIES)]
        state = GAME.init()
        for played in range(max_positions):
            moves, succ = complete_moves(state)
            terminal = bool(_terminal(_stack(state))[0])
            record = {
                "game": g,
                "policy": policy,
                "state": wire(state, played),
                "outcome": outcome_of(state),
            }
            if terminal:
                out.append(record)
                break
            n = len(moves)
            succ_terminal = np.asarray(_terminal(succ))[:n]
            succ_rewards = np.asarray(_rewards(succ))[:n]
            order = sorted(range(n), key=lambda i: moves[i])
            record["moves"] = [list(moves[i]) for i in order]
            record["successors"] = [digest(wire(_row(succ, i), played + 1)) for i in order]
            record["successor_outcomes"] = "".join(
                outcome_code(bool(succ_terminal[i]), succ_rewards[i]) for i in order
            )
            out.append(record)
            if n == 0:
                break
            state = _row(succ, pick(policy, rng, state, moves, succ))
        ends = out[-1]["outcome"]
        print(
            f"game {g} ({policy}): {sum(r['game'] == g for r in out)} positions, ended {ends!r}",
            file=sys.stderr,
            flush=True,
        )
    return out


def perft(state: GameState, depth: int) -> int:
    if depth == 0:
        return 1
    if bool(_terminal(_stack(state))[0]):
        return 0
    moves, succ = complete_moves(state)
    if depth == 1:
        return len(moves)
    return sum(perft(_row(succ, i), depth - 1) for i in range(len(moves)))


def state_from_wire(w: dict[str, Any]) -> GameState:
    board = np.array([".BW".index(c) for c in w["board"]], dtype=np.int8)
    return GameState(
        color=jnp.int32(w["to_move"]),
        board=jnp.asarray(board),
        stage=jnp.int32(0),
        source=jnp.int32(0),
        winner=jnp.int32(-1),
        no_capture_turns=jnp.int32(w["quiet_moves"]),
    )


def pgx_provenance() -> dict[str, Any]:
    root = Path(pgx.__file__).resolve().parent.parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    env = pgx.make("gess")
    return {
        "pgx_env_version": env.version,
        "pgx_commit": git("rev-parse", "HEAD"),
        "pgx_rules_dirty": bool(git("status", "--porcelain", "pgx/_src/games/gess.py")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--max-positions", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261)
    ap.add_argument(
        "--perft-cases", type=int, default=3, help="late positions to take perft(2) from"
    )
    ap.add_argument(
        "--only", choices=("positions", "perft"), help="regenerate just one of the two files"
    )
    args = ap.parse_args()

    meta = {"kind": "meta", **pgx_provenance(), "seed": args.seed, "games": args.games}
    print(json.dumps(meta), file=sys.stderr)

    positions = HERE / "gess.jsonl.gz"
    if args.only != "perft":
        records = play_games(args.games, args.max_positions, args.seed)
        with gzip.open(positions, "wt", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
            for r in records:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        print(f"wrote {len(records)} positions", file=sys.stderr)
    if args.only == "positions":
        return
    with gzip.open(positions, "rt", encoding="utf-8") as f:
        records = [json.loads(line) for line in f][1:]

    # Gess branches into the low thousands, so perft(2) is only affordable from positions that
    # have been stripped down: take the emptiest ones there are.
    perfts: list[dict[str, Any]] = []
    start = GAME.init()
    for d in (1, 2):
        perfts.append({"state": wire(start, 0), "depth": d, "nodes": perft(start, d)})
        print(f"start perft({d}) = {perfts[-1]['nodes']}", file=sys.stderr, flush=True)
    playable = [r for r in records if r["outcome"] == "-" and r["state"]["moves"] > 0]
    playable.sort(key=lambda r: r["state"]["board"].count("."))
    for r in playable[-args.perft_cases :]:
        nodes = perft(state_from_wire(r["state"]), 2)
        perfts.append({"state": r["state"], "depth": 2, "nodes": nodes})
        print(f"perft(2) = {nodes}", file=sys.stderr, flush=True)
    (HERE / "gess_perft.json").write_text(
        json.dumps({**meta, "kind": "perft", "cases": perfts}, indent=1) + "\n"
    )
    print("wrote perft", file=sys.stderr)


if __name__ == "__main__":
    main()

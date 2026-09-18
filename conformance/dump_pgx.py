"""Dump Epaminondas positions from pgx for the Rust conformance test.

Run in the pgx environment (it imports pgx; it lives here so the pgx repo is untouched):

    JAX_PLATFORMS=cpu python conformance/dump_pgx.py

Writes two files next to this script:

  epaminondas.jsonl.gz  one line per position reached in a set of seeded games. Each carries the
                        wire state, every complete legal move (lead, rear, dest), and for each
                        move a digest of the successor state and that successor's outcome.
  epaminondas_perft.json  move-tree node counts from the start and from a few midgame positions.

pgx plays a move as three actions, so the complete moves of a position are found by walking its
action masks: every legal lead, then every legal rear for that lead, then every legal destination
for that pair. A (lead, rear) whose destination mask is empty contributes nothing, exactly as a
player could never finish that move.

Regenerate whenever pgx's Epaminondas rules change, and commit the result.
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
from pgx._src.games.epaminondas import Game, GameState  # noqa: E402

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


def complete_moves(state: GameState) -> tuple[list[tuple[int, int, int]], GameState]:
    """Every complete move from a stage-0 state, and the batch of successors in the same order."""
    one = _stack(state)
    leads = np.flatnonzero(np.asarray(_mask(one)[0]))
    if len(leads) == 0:
        return [], one

    # Stage 1: one row per lead.
    s1 = _step(
        _take(one, np.zeros(len(leads), dtype=np.int64)),
        jnp.asarray(np.pad(leads, (0, _bucket(len(leads)) - len(leads)))),
    )
    rear_masks = np.asarray(_mask(s1))[: len(leads)]
    pairs = [(i, int(r)) for i in range(len(leads)) for r in np.flatnonzero(rear_masks[i])]
    if not pairs:
        return [], one

    # Stage 2: one row per (lead, rear).
    rows = np.array([i for i, _ in pairs])
    rears = np.array([r for _, r in pairs])
    s2 = _step(_take(s1, rows), jnp.asarray(np.pad(rears, (0, _bucket(len(rears)) - len(rears)))))
    dest_masks = np.asarray(_mask(s2))[: len(pairs)]
    triples = [(j, int(d)) for j in range(len(pairs)) for d in np.flatnonzero(dest_masks[j])]
    if not triples:
        return [], one

    # Stage 3: one row per complete move.
    rows3 = np.array([j for j, _ in triples])
    dests = np.array([d for _, d in triples])
    s3 = _step(_take(s2, rows3), jnp.asarray(np.pad(dests, (0, _bucket(len(dests)) - len(dests)))))
    moves = [(int(leads[pairs[j][0]]), pairs[j][1], d) for j, d in triples]
    return moves, s3


def wire(state: GameState) -> dict[str, Any]:
    board = "".join(".WB"[int(c)] for c in np.asarray(state.board))
    return {
        "board": board,
        "to_move": int(state.color),
        "moves": int(state.moves),
        "quiet_moves": int(state.quiet_moves),
        "last_capturer": int(state.last_capturer),
    }


def digest(w: dict[str, Any]) -> str:
    """FNV-1a (32-bit) of a canonical rendering of a wire state. Mirrored in the Rust test."""
    text = f"{w['board']}|{w['to_move']}|{w['moves']}|{w['quiet_moves']}|{w['last_capturer']}"
    h = 0x811C9DC5
    for b in text.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def outcome_code(terminal: bool, rewards: np.ndarray) -> str:
    """'-' ongoing, '0' white won, '1' black won. pgx Epaminondas has no draws."""
    if not terminal:
        return "-"
    return str(int(np.argmax(rewards)))


def outcome_of(state: GameState) -> str:
    one = _stack(state)
    return outcome_code(bool(_terminal(one)[0]), np.asarray(_rewards(one)[0]))


POLICIES = ("uniform", "quiet", "aggressive")


def pick(
    policy: str,
    rng: random.Random,
    state: GameState,
    moves: list[tuple[int, int, int]],
    succ: GameState,
) -> int:
    """Index of the move to play.

    'uniform' is plain random play. 'quiet' avoids captures and incursions where it can, so games
    run long enough to reach the quiet-move clock and its advancement tiebreak. 'aggressive'
    captures whenever it can and otherwise advances, which is what exercises the capture rule and
    back-rank incursions: random play almost never sets either up.
    """
    before = int((np.asarray(state.board) != 0).sum())
    boards = np.asarray(succ.board)[: len(moves)]
    captures = [i for i in range(len(moves)) if int((boards[i] != 0).sum()) < before]
    white = int(state.color) == 0
    opp_back = 11 if white else 0
    if policy == "quiet":
        calm = [
            i
            for i, (_, _, dest) in enumerate(moves)
            if i not in captures and dest // 14 not in (opp_back, 11 - opp_back)
        ]
        if calm:
            return rng.choice(calm)
    elif policy == "aggressive":
        if captures:
            return rng.choice(captures)
        # Otherwise favour moves whose lead ends up furthest towards the opponent.
        progress = [(d // 14) if white else (11 - d // 14) for _, _, d in moves]
        best = max(progress)
        forward = [i for i, p in enumerate(progress) if p >= best - 1]
        if rng.random() < 0.7:
            return rng.choice(forward)
    return rng.randrange(len(moves))


def play_games(n_games: int, max_positions: int, seed: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for g in range(n_games):
        rng = random.Random(seed * 1000 + g)
        policy = POLICIES[g % len(POLICIES)]
        state = GAME.init()
        for _ in range(max_positions):
            moves, succ = complete_moves(state)
            terminal = bool(_terminal(_stack(state))[0])
            record = {
                "game": g,
                "policy": policy,
                "state": wire(state),
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
            record["successors"] = [digest(wire(_row(succ, i))) for i in order]
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


def pgx_provenance() -> dict[str, Any]:
    root = Path(pgx.__file__).resolve().parent.parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    env = pgx.make("epaminondas")
    return {
        "pgx_env_version": env.version,
        "pgx_commit": git("rev-parse", "HEAD"),
        "pgx_rules_dirty": bool(git("status", "--porcelain", "pgx/_src/games/epaminondas.py")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--max-positions", type=int, default=120)
    ap.add_argument("--seed", type=int, default=77659)
    ap.add_argument("--perft-depth", type=int, default=3)
    ap.add_argument(
        "--only", choices=("positions", "perft"), help="regenerate just one of the two files"
    )
    args = ap.parse_args()

    meta = {"kind": "meta", **pgx_provenance(), "seed": args.seed, "games": args.games}
    print(json.dumps(meta), file=sys.stderr)

    positions = HERE / "epaminondas.jsonl.gz"
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

    # Perft from the start, and to depth 2 from a spread of midgame positions.
    perfts: list[dict[str, Any]] = []
    start = GAME.init()
    for d in range(1, args.perft_depth + 1):
        perfts.append({"state": wire(start), "depth": d, "nodes": perft(start, d)})
        print(f"start perft({d}) = {perfts[-1]['nodes']}", file=sys.stderr, flush=True)
    midgame = [r for r in records if r["outcome"] == "-" and r["state"]["moves"] >= 20]
    for r in midgame[:: max(1, len(midgame) // 6)][:6]:
        board = np.array([".WB".index(c) for c in r["state"]["board"]], dtype=np.int8)
        s = GameState(
            color=jnp.int32(r["state"]["to_move"]),
            board=jnp.asarray(board),
            moves=jnp.int32(r["state"]["moves"]),
            quiet_moves=jnp.int32(r["state"]["quiet_moves"]),
            last_capturer=jnp.int32(r["state"]["last_capturer"]),
        )
        perfts.append({"state": r["state"], "depth": 2, "nodes": perft(s, 2)})
    (HERE / "epaminondas_perft.json").write_text(
        json.dumps({**meta, "kind": "perft", "cases": perfts}, indent=1) + "\n"
    )
    print("wrote perft", file=sys.stderr)


if __name__ == "__main__":
    main()

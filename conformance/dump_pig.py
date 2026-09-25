"""Dump Pig positions from pgx for the Rust conformance test.

Run in the pgx environment (it imports pgx; it lives here so the pgx repo is untouched):

    JAX_PLATFORMS=cpu python conformance/dump_pig.py

Writes `pig.jsonl.gz` next to this script: a meta line, which also carries the start position
after each first roll, then one line per position reached in a set of seeded games. Each
position carries the wire state, its outcome, its legal moves (pgx actions: 0 stop, 1 roll), and
for each move and each face of the die a digest of the successor state and its outcome.

pgx draws the die from the key passed to `step` (and `init`), and uses it directly, so a key
that pgx turns into a given face forces that face. The script finds one key per face up front,
which is what lets it enumerate every roll rather than sample one.

Pig has no fixed length and random play decides little, so the games mix three policies:
uniform random; hold at 20 (the classic simple strategy, which reaches 100 quickly); and hold at
20 until the target is in reach, then keep rolling to 110 - which is what gets a turn total past
100 without banking it, and a 1 rolled there.

Regenerate whenever pgx's pig rules change, and commit the result.
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
from pgx._src.games.pig import TARGET, Game, GameState, _roll  # noqa: E402

GAME = Game()
HERE = Path(__file__).resolve().parent
FACES = 6
ACTIONS = 2


def face_keys() -> list[jax.Array]:
    """`keys[r - 1]` makes pgx roll `r`."""
    keys: dict[int, jax.Array] = {}
    i = 0
    while len(keys) < FACES:
        key = jax.random.PRNGKey(i)
        keys.setdefault(int(_roll(key)), key)
        i += 1
    return [keys[r] for r in range(1, FACES + 1)]


KEYS = jnp.stack(face_keys())
# Every (action, face) pair, action-major: the successors of one position in one batch.
PAIR_ACTIONS = jnp.repeat(jnp.arange(ACTIONS, dtype=jnp.int32), FACES)
PAIR_KEYS = jnp.tile(KEYS, (ACTIONS, 1))

_init = jax.jit(jax.vmap(GAME.init))
_mask = jax.jit(GAME.legal_action_mask)


@jax.jit
def _successors(state: GameState) -> GameState:
    batch = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None], ACTIONS * FACES, axis=0), state)
    return jax.vmap(GAME.step)(batch, PAIR_ACTIONS, PAIR_KEYS)


def _row(batch: GameState, i: int) -> GameState:
    return jax.tree_util.tree_map(lambda x: x[i], batch)


def wire(state: GameState) -> dict[str, Any]:
    return {
        "totals": [int(v) for v in np.asarray(state.totals)],
        "turn_total": int(state.turn_total),
        "last_roll": int(state.last_roll),
        "to_move": int(state.color),
    }


def digest(w: dict[str, Any]) -> str:
    """FNV-1a (32-bit) of a canonical rendering of a wire state. Mirrored in the Rust test."""
    text = f"{w['totals'][0]},{w['totals'][1]}|{w['turn_total']}|{w['last_roll']}|{w['to_move']}"
    h = 0x811C9DC5
    for b in text.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def outcome_of(state: GameState) -> str:
    """'-' ongoing, or the side that won. Pig has no draws."""
    winner = int(state.winner)
    return "-" if winner < 0 else str(winner)


POLICIES = ("uniform", "hold_at_20", "overshoot")


def wants_roll(policy: str, rng: random.Random, w: dict[str, Any]) -> bool:
    banked = w["totals"][w["to_move"]] + w["turn_total"]
    if policy == "uniform":
        return rng.random() < 0.5
    if policy == "overshoot" and banked >= TARGET:
        return banked < TARGET + 10
    return w["turn_total"] < 20 and banked < TARGET


def play_games(n_games: int, seed: int, max_positions: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for g in range(n_games):
        rng = random.Random(seed * 1000 + g)
        policy = POLICIES[g % len(POLICIES)]
        state = GAME.init(KEYS[rng.randrange(FACES)])
        n = 0
        while True:
            w = wire(state)
            record: dict[str, Any] = {"game": g, "policy": policy, "state": w}
            record["outcome"] = outcome_of(state)
            n += 1
            if record["outcome"] != "-" or n > max_positions:
                out.append(record)
                break
            legal = [a for a in range(ACTIONS) if bool(_mask(state)[a])]
            succ = jax.tree_util.tree_map(np.asarray, _successors(state))
            record["moves"] = legal
            record["successors"] = [
                [digest(wire(_row(succ, a * FACES + f))) for f in range(FACES)] for a in legal
            ]
            record["successor_outcomes"] = [
                "".join(outcome_of(_row(succ, a * FACES + f)) for f in range(FACES)) for a in legal
            ]
            out.append(record)
            action = 1 if 1 in legal and wants_roll(policy, rng, w) else 0
            state = _row(succ, action * FACES + rng.randrange(FACES))
        print(
            f"game {g} ({policy}): {n} positions, ended {out[-1]['outcome']!r} "
            f"at {out[-1]['state']['totals']}",
            file=sys.stderr,
            flush=True,
        )
    return out


def pgx_provenance() -> dict[str, Any]:
    root = Path(pgx.__file__).resolve().parent.parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    env = pgx.make("pig")
    return {
        "pgx_env_version": env.version,
        "pgx_commit": git("rev-parse", "HEAD"),
        "pgx_rules_dirty": bool(git("status", "--porcelain", "pgx/_src/games/pig.py")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games", type=int, default=36)
    ap.add_argument("--seed", type=int, default=20263)
    ap.add_argument(
        "--max-positions", type=int, default=2000, help="cut a game off after this many"
    )
    args = ap.parse_args()

    starts = _init(KEYS)
    meta = {
        "kind": "meta",
        **pgx_provenance(),
        "seed": args.seed,
        "games": args.games,
        "starts": [wire(_row(starts, f)) for f in range(FACES)],
    }
    print(json.dumps(meta), file=sys.stderr)

    records = play_games(args.games, args.seed, args.max_positions)
    with gzip.open(HERE / "pig.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(f"wrote {len(records)} positions", file=sys.stderr)


if __name__ == "__main__":
    main()

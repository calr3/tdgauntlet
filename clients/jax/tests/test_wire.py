"""The JAX client's view of a position must be exactly pgx's own.

Skipped where pgx is not installed (CI); run in the pgx environment:

    JAX_PLATFORMS=cpu python -m pytest -p no:xdist clients/jax
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pgx")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from tdg_jax.wire import ENV, actions_to_move, from_pgx, move_to_actions, to_pgx  # noqa: E402

VECTORS = Path(__file__).resolve().parents[3] / "conformance" / "epaminondas.jsonl.gz"
_step = jax.jit(ENV.step)


def vectors() -> list[dict[str, Any]]:
    with gzip.open(VECTORS, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f][1:]


def digest(w: dict[str, Any]) -> str:
    """The conformance suite's successor digest (FNV-1a over a canonical rendering)."""
    text = f"{w['board']}|{w['to_move']}|{w['moves']}|{w['quiet_moves']}|{w['last_capturer']}"
    h = 0x811C9DC5
    for b in text.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def assert_same_state(ours: Any, theirs: Any) -> None:
    for name in ("current_player", "observation", "legal_action_mask", "terminated", "rewards"):
        np.testing.assert_array_equal(getattr(ours, name), getattr(theirs, name), err_msg=name)
    for name in (
        "color",
        "board",
        "stage",
        "lead",
        "rear",
        "winner",
        "moves",
        "quiet_moves",
        "last_capturer",
    ):
        np.testing.assert_array_equal(
            getattr(ours._x, name), getattr(theirs._x, name), err_msg=f"_x.{name}"
        )


def test_rebuilt_states_match_states_pgx_reached_itself() -> None:
    """Play games in pgx; at every move boundary, going to the wire and back loses nothing."""
    rng = random.Random(3)
    checked = 0
    for game in range(4):
        state = ENV.init(jax.random.PRNGKey(game))
        while not bool(state.terminated) and int(state._x.moves) < 60:
            if int(state._x.stage) == 0:
                rebuilt = to_pgx(from_pgx(state))
                assert_same_state(rebuilt, state)
                checked += 1
            legal = np.flatnonzero(np.asarray(state.legal_action_mask))
            state = _step(state, jnp.int32(rng.choice(list(legal))))
    assert checked > 100


def test_wire_states_round_trip() -> None:
    for r in vectors()[::25]:
        assert from_pgx(to_pgx(r["state"])) == r["state"]


def test_moves_step_pgx_to_the_servers_successor() -> None:
    """Every move the server considers legal, played as three pgx actions, reaches the state the
    server's own rules reach: the conversion agrees with the conformance vectors."""
    vstep = jax.jit(jax.vmap(ENV.step))
    checked = 0
    for r in [r for r in vectors() if "moves" in r][::60]:
        moves = np.array(r["moves"])
        n = len(moves)
        size = 1 << (n - 1).bit_length()  # pad to a power of two: few compilations
        idx = np.concatenate([np.arange(n), np.zeros(size - n, dtype=int)])
        state = to_pgx(r["state"])
        batch = jax.tree_util.tree_map(lambda x, size=size: jnp.stack([x] * size), state)
        for stage in range(3):
            actions = jnp.asarray(moves[idx, stage])
            assert bool(batch.legal_action_mask[jnp.arange(size), actions].all()), (
                f"pgx refuses a stage-{stage} action from {r['state']}"
            )
            batch = vstep(batch, actions)
        for i in range(n):
            got = from_pgx(jax.tree_util.tree_map(lambda x, i=i: x[i], batch))
            assert digest(got) == r["successors"][i], f"{moves[i]} from {r['state']}"
            checked += 1
    assert checked > 2000


def test_move_actions_round_trip() -> None:
    assert actions_to_move(move_to_actions({"lead": 17, "rear": 3, "dest": 45, "san": "x"})) == {
        "lead": 17,
        "rear": 3,
        "dest": 45,
    }


@pytest.mark.parametrize(
    "bad",
    [
        {"board": "." * 167, "to_move": 0, "moves": 0, "quiet_moves": 0, "last_capturer": -1},
        {"board": "x" * 168, "to_move": 0, "moves": 0, "quiet_moves": 0, "last_capturer": -1},
        {"board": "." * 168, "to_move": 2, "moves": 0, "quiet_moves": 0, "last_capturer": -1},
        {"board": "." * 168, "to_move": 0, "moves": 0, "last_capturer": -1},
    ],
)
def test_malformed_states_are_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises((ValueError, KeyError)):
        to_pgx(bad)

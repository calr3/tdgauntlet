"""The JAX client's view of a position must be exactly pgx's own, in every game it plays.

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
from tdg_jax.wire import WIRES, Wire, for_game_id  # noqa: E402

CONFORMANCE = Path(__file__).resolve().parents[3] / "conformance"
#: The fields the conformance dumps hash, in order, per game.
DIGEST_FIELDS = {
    "epaminondas/1": ("board", "to_move", "moves", "quiet_moves", "last_capturer"),
    "gess/1": ("board", "to_move", "moves", "quiet_moves"),
}
VECTORS = {"epaminondas/1": "epaminondas.jsonl.gz", "gess/1": "gess.jsonl.gz"}
GAME_IDS = [w.game_id for w in WIRES]
#: `_x` fields that mean nothing between moves, so the wire does not carry them. Gess leaves the
#: last move's source in place; `test_a_stale_gess_source_is_invisible` is why that is harmless.
SCRATCH = {"epaminondas/1": (), "gess/1": ("source",)}
#: Games to walk, and the least move boundaries that must produce. Random Gess is short: a piece
#: lifted from around your own ring destroys it and loses on the spot.
WALKS = {"epaminondas/1": (4, 100), "gess/1": (20, 30)}


@pytest.fixture(params=GAME_IDS)
def wire(request: Any) -> Wire:
    return for_game_id(request.param)


def vectors(wire: Wire) -> list[dict[str, Any]]:
    with gzip.open(CONFORMANCE / VECTORS[wire.game_id], "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f][1:]


def digest(wire: Wire, w: dict[str, Any]) -> str:
    """The conformance suite's successor digest (FNV-1a over a canonical rendering)."""
    text = "|".join(str(w[f]) for f in DIGEST_FIELDS[wire.game_id])
    h = 0x811C9DC5
    for b in text.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def assert_same_state(wire: Wire, ours: Any, theirs: Any) -> None:
    # Everything the network and the rules read.
    for name in ("current_player", "observation", "legal_action_mask", "terminated", "rewards"):
        np.testing.assert_array_equal(getattr(ours, name), getattr(theirs, name), err_msg=name)
    # `GameState` is a NamedTuple, so this covers whatever fields the game happens to have.
    for name in theirs._x._fields:
        if name in SCRATCH[wire.game_id]:
            continue
        np.testing.assert_array_equal(
            getattr(ours._x, name), getattr(theirs._x, name), err_msg=f"_x.{name}"
        )


def test_rebuilt_states_match_states_pgx_reached_itself(wire: Wire) -> None:
    """Play games in pgx; at every move boundary, going to the wire and back loses nothing."""
    rng = random.Random(3)
    step = jax.jit(wire.env.step)
    games, least = WALKS[wire.game_id]
    checked = 0
    for game in range(games):
        state = wire.env.init(jax.random.PRNGKey(game))
        moves = 0
        while not bool(state.terminated) and moves < 60:
            if int(state._x.stage) == 0:
                assert_same_state(wire, wire.to_pgx(wire.from_pgx(state)), state)
                checked += 1
                moves += 1
            legal = np.flatnonzero(np.asarray(state.legal_action_mask))
            state = step(state, jnp.int32(rng.choice(list(legal))))
    assert checked >= least


def test_a_stale_gess_source_is_invisible() -> None:
    """pgx leaves the last move's source in the state between moves, and the wire drops it. That
    is only sound while nothing at stage 0 reads it: this is the check that nothing does."""
    wire = for_game_id("gess/1")
    step = jax.jit(wire.env.step)
    rng = np.random.default_rng(0)
    state = wire.env.init(jax.random.PRNGKey(0))
    for _ in range(2):  # one full move, which leaves a source behind
        legal = np.flatnonzero(np.asarray(state.legal_action_mask))
        state = step(state, jnp.int32(rng.choice(legal)))
    assert int(state._x.stage) == 0 and int(state._x.source) != 0, "no stale source to test with"
    zeroed = state.replace(_x=state._x._replace(source=jnp.int32(0)))
    zeroed = zeroed.replace(observation=wire.env.observe(zeroed))
    np.testing.assert_array_equal(zeroed.observation, state.observation)
    np.testing.assert_array_equal(
        wire.game.legal_action_mask(zeroed._x), wire.game.legal_action_mask(state._x)
    )


def test_wire_states_round_trip(wire: Wire) -> None:
    for r in vectors(wire)[::25]:
        assert wire.from_pgx(wire.to_pgx(r["state"])) == r["state"]


def test_moves_step_pgx_to_the_servers_successor(wire: Wire) -> None:
    """Every move the server considers legal, played as pgx actions, reaches the state the
    server's own rules reach: the conversion agrees with the conformance vectors."""
    vstep = jax.jit(jax.vmap(wire.env.step))
    checked = 0
    for r in [r for r in vectors(wire) if "moves" in r][::60]:
        moves = np.array(r["moves"])
        n = len(moves)
        if n == 0:
            continue
        size = 1 << (n - 1).bit_length()  # pad to a power of two: few compilations
        idx = np.concatenate([np.arange(n), np.zeros(size - n, dtype=int)])
        state = wire.to_pgx(r["state"])
        batch = jax.tree_util.tree_map(lambda x, size=size: jnp.stack([x] * size), state)
        for stage in range(wire.actions_per_move):
            actions = jnp.asarray(moves[idx, stage])
            assert bool(batch.legal_action_mask[jnp.arange(size), actions].all()), (
                f"pgx refuses a stage-{stage} action from {r['state']}"
            )
            batch = vstep(batch, actions)
        for i in range(n):
            got = wire.from_pgx(jax.tree_util.tree_map(lambda x, i=i: x[i], batch))
            assert digest(wire, got) == r["successors"][i], f"{moves[i]} from {r['state']}"
            checked += 1
    # The two games' vector sets are different sizes; both check a few thousand moves.
    assert checked > (2000 if wire.game_id == "epaminondas/1" else 1500)


def test_move_actions_round_trip() -> None:
    epaminondas, gess = for_game_id("epaminondas/1"), for_game_id("gess/1")
    move = {"lead": 17, "rear": 3, "dest": 45, "san": "x"}
    assert epaminondas.actions_to_move(epaminondas.move_to_actions(move)) == {
        "lead": 17,
        "rear": 3,
        "dest": 45,
    }
    assert gess.move_to_actions({"from": 126, "to": 166, "san": "x"}) == [126, 166]
    assert gess.actions_to_move([126, 166]) == {"from": 126, "to": 166}
    with pytest.raises(ValueError, match="2 actions"):
        gess.actions_to_move([126, 166, 3])


def epaminondas_state(**changes: Any) -> dict[str, Any]:
    base = {
        "board": "." * 168,
        "to_move": 0,
        "moves": 0,
        "quiet_moves": 0,
        "last_capturer": -1,
    }
    return {**base, **changes}


def gess_state(**changes: Any) -> dict[str, Any]:
    return {**{"board": "." * 400, "to_move": 0, "moves": 0, "quiet_moves": 0}, **changes}


@pytest.mark.parametrize(
    ("game_id", "bad"),
    [
        ("epaminondas/1", epaminondas_state(board="." * 167)),
        ("epaminondas/1", epaminondas_state(board="x" * 168)),
        ("epaminondas/1", epaminondas_state(to_move=2)),
        ("epaminondas/1", {k: v for k, v in epaminondas_state().items() if k != "quiet_moves"}),
        ("gess/1", gess_state(board="." * 399)),
        ("gess/1", gess_state(board="x" * 400)),
        # Gess writes black first, so an Epaminondas-length board is wrong twice over.
        ("gess/1", gess_state(board="." * 168)),
        ("gess/1", gess_state(to_move=2)),
        ("gess/1", {k: v for k, v in gess_state().items() if k != "quiet_moves"}),
    ],
)
def test_malformed_states_are_rejected(game_id: str, bad: dict[str, Any]) -> None:
    with pytest.raises((ValueError, KeyError)):
        for_game_id(game_id).to_pgx(bad)


def test_an_unknown_game_has_no_wire() -> None:
    with pytest.raises(ValueError, match="this client plays"):
        for_game_id("go/1")

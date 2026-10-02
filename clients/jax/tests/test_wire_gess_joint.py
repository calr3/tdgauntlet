"""The wire for checkpoints trained on pgx's ``gess_joint`` (one action per move): the same
game as ``gess/1``, so it is held to the same conformance vectors as the two-step wire."""

from __future__ import annotations

import pytest

pytest.importorskip("pgx")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from tdg_jax.wire import GESS_JOINT, for_env_id  # noqa: E402
from test_wire import digest, vectors  # noqa: E402

N = 400


def test_found_by_env_id() -> None:
    assert for_env_id("gess_joint") is GESS_JOINT
    assert GESS_JOINT.game_id == "gess/1"
    assert GESS_JOINT.actions_per_move == 1


def test_wire_states_round_trip() -> None:
    for r in vectors(GESS_JOINT)[::25]:
        assert GESS_JOINT.from_pgx(GESS_JOINT.to_pgx(r["state"])) == r["state"]


def test_move_lists_are_the_servers_legal_moves_and_reach_its_successors() -> None:
    """For each vector position: pgx's move list is exactly the server's legal moves, and
    playing each listed move reaches the server's successor."""
    step = jax.jit(jax.vmap(GESS_JOINT.env.step))
    checked = 0
    for r in [r for r in vectors(GESS_JOINT) if "moves" in r][::60]:
        state = GESS_JOINT.to_pgx(r["state"])
        listed = np.asarray(state._moves)
        listed = listed[listed >= 0]
        server = [m[0] * N + m[1] for m in r["moves"]]
        assert sorted(server) == listed.tolist(), r["state"]
        if not server:
            continue
        slot = {int(m): i for i, m in enumerate(listed)}
        n = len(server)
        size = 1 << (n - 1).bit_length()
        idx = np.concatenate([np.arange(n), np.zeros(size - n, dtype=int)])
        actions = jnp.asarray([slot[server[i]] for i in idx])
        batch = jax.tree_util.tree_map(lambda x, size=size: jnp.stack([x] * size), state)
        batch = step(batch, actions)
        for i in range(n):
            got = GESS_JOINT.from_pgx(jax.tree_util.tree_map(lambda x, i=i: x[i], batch))
            assert digest(GESS_JOINT, got) == r["successors"][i], f"{r['moves'][i]}"
            checked += 1
    assert checked > 1500


def test_move_actions_round_trip() -> None:
    """A wire move and its slot in the position's move list translate both ways."""
    for r in [r for r in vectors(GESS_JOINT) if r.get("moves")][::200]:
        state = GESS_JOINT.to_pgx(r["state"])
        for m in r["moves"][:20]:
            move = {"from": m[0], "to": m[1]}
            actions = GESS_JOINT.move_to_actions(move, state)
            assert bool(state.legal_action_mask[actions[0]])
            assert GESS_JOINT.actions_to_move(actions, state) == move
    with pytest.raises(ValueError):
        GESS_JOINT.actions_to_move([0])

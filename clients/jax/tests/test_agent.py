"""The JAX agent: loading, legality, and batching concurrent requests from unrelated games.

The model tests need a checkpoint: set TDG_JAX_CHECKPOINT, or they use the newest Epaminondas
checkpoint under ~/calr3gh/pgx/checkpoints trained on the current rules, and skip if there is none.
A checkpoint named in TDG_JAX_CHECKPOINT is taken as meant, rules label and all, so a newer-rules
one can be used to exercise the agent against pgx's current network.
"""

from __future__ import annotations

import os
import pickle
import threading
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pgx")

import sys  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from tdg_jax.agent import Agent, alphazero_dir, load_checkpoint  # noqa: E402
from tdg_jax.wire import ENV, RULES_VERSION, from_pgx, to_pgx  # noqa: E402


def find_checkpoint() -> Path | None:
    if os.environ.get("TDG_JAX_CHECKPOINT"):
        return Path(os.environ["TDG_JAX_CHECKPOINT"])
    # Checkpoints pickle pgx's training Config, so its module must be importable to read one.
    # Take it off the path again afterwards, so the agent is tested setting it up for itself.
    sys.path.insert(0, str(alphazero_dir()))
    try:
        root = Path.home() / "calr3gh" / "pgx" / "checkpoints"
        for run in sorted(root.glob("epaminondas_*"), reverse=True):
            for ckpt in sorted(run.glob("*.ckpt"), reverse=True):
                with open(ckpt, "rb") as f:
                    meta = pickle.load(f)
                if meta.get("env_version") == RULES_VERSION:
                    return ckpt
                break  # every checkpoint of a run shares its rules; try the next run
        return None
    finally:
        sys.path.remove(str(alphazero_dir()))
        for module in ("config", "network"):
            sys.modules.pop(module, None)


CHECKPOINT = find_checkpoint()
ALLOW_MISMATCH = bool(os.environ.get("TDG_JAX_CHECKPOINT"))
needs_model = pytest.mark.skipif(CHECKPOINT is None, reason="no current-rules checkpoint")


class Stub:
    """Just enough of a pickled config for load_checkpoint to get as far as the version check."""


def fake_checkpoint(tmp_path: Path, **fields: Any) -> Path:
    path = tmp_path / "fake.ckpt"
    with open(path, "wb") as f:
        pickle.dump({"config": Stub(), "model": None, **fields}, f)
    return path


def test_a_checkpoint_for_other_rules_is_refused(tmp_path: Path) -> None:
    path = fake_checkpoint(tmp_path, env_id="epaminondas", env_version="v5")
    with pytest.raises(ValueError, match="trained on pgx epaminondas v5"):
        load_checkpoint(path)


def test_a_checkpoint_for_another_game_is_refused(tmp_path: Path) -> None:
    path = fake_checkpoint(tmp_path, env_id="gess", env_version="v1")
    with pytest.raises(ValueError, match="gess checkpoint"):
        load_checkpoint(path)


@pytest.fixture(scope="module")
def agent() -> Agent:
    assert CHECKPOINT is not None
    a = Agent(CHECKPOINT, num_simulations=8, max_batch=8, allow_rules_mismatch=ALLOW_MISMATCH)
    a.warm_up()
    return a


def some_positions(n: int) -> list[dict[str, Any]]:
    """Wire states from a random pgx game, at move boundaries."""
    step = jax.jit(ENV.step)
    rng = np.random.default_rng(1)
    state = ENV.init(jax.random.PRNGKey(0))
    out = []
    while len(out) < n and not bool(state.terminated):
        if int(state._x.stage) == 0:
            out.append(from_pgx(state))
        legal = np.flatnonzero(np.asarray(state.legal_action_mask))
        state = step(state, jnp.int32(rng.choice(legal)))
    return out


def legal_move(wire: dict[str, Any], actions: list[int]) -> bool:
    state = to_pgx(wire)
    for a in actions:
        if not bool(state.legal_action_mask[a]):
            return False
        state = ENV.step(state, jnp.int32(a))
    return int(state._x.stage) == 0


@needs_model
def test_it_plays_legal_moves(agent: Agent) -> None:
    for wire in some_positions(4):
        actions, info = agent.choose(to_pgx(wire))
        assert legal_move(wire, actions), (wire, actions)
        assert info["sims"] == 8 and "eval" in info


@needs_model
def test_policy_only_plays_legal_moves_without_searching() -> None:
    assert CHECKPOINT is not None
    a = Agent(
        CHECKPOINT,
        num_simulations=8,
        max_batch=4,
        policy_only=True,
        allow_rules_mismatch=ALLOW_MISMATCH,
    )
    a.warm_up()
    for wire in some_positions(3):
        actions, info = a.choose(to_pgx(wire))
        assert legal_move(wire, actions), (wire, actions)
        assert info["sims"] == 0 and info["policy_only"] is True
    # No search means no randomness: the same position always gets the same move.
    wire = some_positions(1)[0]
    assert a.choose(to_pgx(wire))[0] == a.choose(to_pgx(wire))[0]


@needs_model
def test_concurrent_requests_from_different_games_are_batched_and_kept_apart(agent: Agent) -> None:
    positions = some_positions(8)
    results: dict[int, list[int]] = {}
    agent.batches.clear()

    def ask(i: int) -> None:
        results[i] = agent.choose(to_pgx(positions[i]))[0]

    threads = [threading.Thread(target=ask, args=(i,)) for i in range(len(positions))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Each answer is legal for its own position, not some other request's.
    for i, wire in enumerate(positions):
        assert legal_move(wire, results[i]), f"request {i} got a move for another position"
    assert max(agent.batches) > 1, f"requests were never batched: {agent.batches}"

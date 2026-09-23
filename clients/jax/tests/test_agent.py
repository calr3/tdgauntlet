"""The JAX agent: loading, legality, and batching concurrent requests from unrelated games.

The model tests need a checkpoint per game: set TDG_JAX_CHECKPOINT, or they use the newest
checkpoint under ~/calr3gh/pgx/checkpoints trained on the rules that game's wire id stands for,
and skip the games with none. A checkpoint named in TDG_JAX_CHECKPOINT is taken as meant, rules
label and all, so a newer-rules one can be used to exercise the agent against pgx's current
network; which game it is for is its own to say.
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
from tdg_jax.agent import (  # noqa: E402
    Agent,
    alphazero_dir,
    load_checkpoint,
    narrow,
    stem_inputs,
)
from tdg_jax.wire import WIRES, Wire, for_env_id, for_game_id  # noqa: E402


def read_meta(path: Path) -> dict[str, Any]:
    """A checkpoint's metadata. Checkpoints pickle pgx's training Config, so its module must be
    importable to read one; take it off the path again afterwards, so the agent is tested setting
    it up for itself."""
    sys.path.insert(0, str(alphazero_dir()))
    try:
        with open(path, "rb") as f:
            meta: dict[str, Any] = pickle.load(f)
        return meta
    finally:
        sys.path.remove(str(alphazero_dir()))
        for module in ("config", "network"):
            sys.modules.pop(module, None)


def find_checkpoints() -> tuple[dict[str, Path], dict[str, str]]:
    """The newest checkpoint for each game, if it was trained on the rules the server plays, and
    why it was not usable otherwise.

    Only the newest run is considered. An older one that happens to carry the right rules label
    is no better: it plays the same rules, and the agent narrows the observation for it either
    way."""
    if os.environ.get("TDG_JAX_CHECKPOINT"):
        path = Path(os.environ["TDG_JAX_CHECKPOINT"])
        wire = for_env_id(read_meta(path).get("env_id", "epaminondas"))
        return {wire.game_id: path}, {}
    found: dict[str, Path] = {}
    why: dict[str, str] = {}
    root = Path.home() / "calr3gh" / "pgx" / "checkpoints"
    for wire in WIRES:
        runs = sorted(root.glob(f"{wire.env_id}_*"), reverse=True)
        newest = next((c for run in runs for c in sorted(run.glob("*.ckpt"), reverse=True)), None)
        if newest is None:
            why[wire.game_id] = f"no {wire.env_id} checkpoint under {root}"
            continue
        version = read_meta(newest).get("env_version")
        if version in wire.rules_versions:
            found[wire.game_id] = newest
        else:
            why[wire.game_id] = (
                f"the newest {wire.env_id} run is pgx {version}, and {wire.game_id} plays "
                f"{', '.join(wire.rules_versions)}"
            )
    return found, why


CHECKPOINTS, NO_CHECKPOINT = find_checkpoints()
ALLOW_MISMATCH = bool(os.environ.get("TDG_JAX_CHECKPOINT"))
_AGENTS: dict[tuple[str, bool, bool], Agent] = {}


@pytest.fixture(params=[w.game_id for w in WIRES])
def wire(request: Any) -> Wire:
    if request.param not in CHECKPOINTS:
        pytest.skip(NO_CHECKPOINT.get(request.param, f"no checkpoint for {request.param}"))
    return for_game_id(request.param)


def agent_for(wire: Wire, policy_only: bool = False, value_only: bool = False) -> Agent:
    """One agent per game and mode, kept for the whole session: loading is the slow part."""
    key = (wire.game_id, policy_only, value_only)
    if key not in _AGENTS:
        a = Agent(
            CHECKPOINTS[wire.game_id],
            num_simulations=8,
            max_batch=8,
            policy_only=policy_only,
            value_only=value_only,
            allow_rules_mismatch=ALLOW_MISMATCH,
        )
        a.warm_up()
        _AGENTS[key] = a
    return _AGENTS[key]


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
    path = fake_checkpoint(tmp_path, env_id="gess", env_version="v0")
    with pytest.raises(ValueError, match="trained on pgx gess v0"):
        load_checkpoint(path)


def test_a_checkpoint_for_a_game_the_server_lacks_is_refused(tmp_path: Path) -> None:
    path = fake_checkpoint(tmp_path, env_id="go", env_version="v0")
    with pytest.raises(ValueError, match="go checkpoint"):
        load_checkpoint(path)


def test_the_checkpoint_says_which_game_is_played(tmp_path: Path) -> None:
    for wire in WIRES:
        for version in wire.rules_versions:
            path = fake_checkpoint(tmp_path, env_id=wire.env_id, env_version=version)
            assert load_checkpoint(path).wire.game_id == wire.game_id


def test_every_version_with_a_game_ids_rules_is_accepted(tmp_path: Path) -> None:
    """pgx bumps its env version for observation changes too, and those leave the rules alone,
    so a wire id stands for a family rather than one version. `epaminondas/1` is pgx v6, and v7
    to v11 only grew the planes: regenerating conformance/epaminondas.jsonl.gz from pgx v11
    reproduces the committed v6 vectors byte for byte."""
    for version in ("v6", "v11"):
        path = fake_checkpoint(tmp_path, env_id="epaminondas", env_version=version)
        assert load_checkpoint(path).wire.game_id == "epaminondas/1"


def test_stem_inputs_reads_the_stem_and_narrow_drops_the_rest() -> None:
    assert stem_inputs({"board_former/stem_conv": {"w": np.zeros((3, 3, 7, 64))}}) == 7
    assert stem_inputs({"mlp/linear": {"w": np.zeros((4, 4))}}) is None

    obs = jnp.arange(2 * 3 * 19, dtype=jnp.float32).reshape(2, 3, 19)
    np.testing.assert_array_equal(narrow(obs, 7), obs[..., :7])
    np.testing.assert_array_equal(narrow(obs, 19), obs)
    np.testing.assert_array_equal(narrow(obs, None), obs)


def test_a_checkpoints_plane_count_is_measured_against_its_own_architecture(wire: Wire) -> None:
    """A network may feed its stem more than the observation - GessFormer adds an is_border
    plane - so the count has to come from the gap against the same architecture built here, not
    from the stem's width alone. A current checkpoint was trained on the env's own planes."""
    agent = agent_for(wire)
    assert agent.obs_channels == int(wire.env.observation_shape[-1])


def test_the_agent_is_fed_the_planes_its_checkpoint_was_trained_on(wire: Wire) -> None:
    """Whatever the checkpoint's vintage, what reaches the network has the channel count its
    stem was built for - which for a current checkpoint is the env's own."""
    agent = agent_for(wire)
    channels = agent.obs_channels
    assert channels is not None
    assert channels <= int(wire.env.observation_shape[-1])
    state = wire.env.init(jax.random.PRNGKey(0))
    assert narrow(state.observation, channels).shape[-1] == channels


def some_positions(wire: Wire, n: int) -> list[dict[str, Any]]:
    """Wire states from random pgx games, at move boundaries."""
    step = jax.jit(wire.env.step)
    rng = np.random.default_rng(1)
    out = []
    for game in range(100):
        state = wire.env.init(jax.random.PRNGKey(game))
        while len(out) < n and not bool(state.terminated):
            if int(state._x.stage) == 0:
                out.append(wire.from_pgx(state))
            legal = np.flatnonzero(np.asarray(state.legal_action_mask))
            state = step(state, jnp.int32(rng.choice(legal)))
        if len(out) >= n:
            break
    return out


def legal_move(wire: Wire, state_wire: dict[str, Any], actions: list[int]) -> bool:
    state = wire.to_pgx(state_wire)
    for a in actions:
        if not bool(state.legal_action_mask[a]):
            return False
        state = wire.env.step(state, jnp.int32(a))
    return int(state._x.stage) == 0


def test_it_plays_legal_moves(wire: Wire) -> None:
    agent = agent_for(wire)
    assert agent.wire is wire
    for state in some_positions(wire, 4):
        actions, info = agent.choose(wire.to_pgx(state))
        assert len(actions) == wire.actions_per_move
        assert legal_move(wire, state, actions), (state, actions)
        assert info["sims"] == 8 and "eval" in info


def test_policy_only_plays_legal_moves_without_searching(wire: Wire) -> None:
    agent = agent_for(wire, policy_only=True)
    for state in some_positions(wire, 3):
        actions, info = agent.choose(wire.to_pgx(state))
        assert legal_move(wire, state, actions), (state, actions)
        assert info["sims"] == 0 and info["policy_only"] is True
    # No search means no randomness: the same position always gets the same move.
    state = some_positions(wire, 1)[0]
    assert agent.choose(wire.to_pgx(state))[0] == agent.choose(wire.to_pgx(state))[0]


def all_moves(wire: Wire, state: Any) -> list[list[int]]:
    """Every legal move from a position between moves, walked one action at a time."""
    step = jax.jit(wire.env.step)
    moves: list[list[int]] = []

    def walk(state: Any, prefix: list[int]) -> None:
        if len(prefix) == wire.actions_per_move:
            moves.append(prefix)
            return
        for a in np.flatnonzero(np.asarray(state.legal_action_mask)):
            walk(step(state, jnp.int32(a)), [*prefix, int(a)])

    walk(state, [])
    return moves


def score(agent: Agent, state: Any, move: list[int]) -> float:
    """What the value head makes of the position a move leaves, for the player who made it."""
    mover = int(state.current_player)
    for a in move:
        state = agent.wire.env.step(state, jnp.int32(a))
    if bool(state.terminated):
        return float(state.rewards[mover])
    params, net_state = agent.model
    obs = narrow(state.observation[None], agent.obs_channels)
    (_, value), _ = agent.forward.apply(params, net_state, obs, is_eval=True)
    return float(value[0]) * (1 if int(state.current_player) == mover else -1)


def test_value_only_plays_the_move_the_value_head_likes_best(wire: Wire) -> None:
    agent = agent_for(wire, value_only=True)
    rng = np.random.default_rng(2)
    for state_wire in some_positions(wire, 2):
        state = wire.to_pgx(state_wire)
        actions, info = agent.choose(state)
        assert legal_move(wire, state_wire, actions), (state_wire, actions)
        assert info["sims"] == 0 and info["value_only"] is True
        moves = all_moves(wire, state)
        assert info["successors"] == len(moves)
        # Its own score is the eval it reports, and no other move scores better.
        assert score(agent, state, actions) == pytest.approx(info["eval"], abs=1e-4)
        for i in rng.choice(len(moves), size=min(8, len(moves)), replace=False):
            assert score(agent, state, moves[i]) <= info["eval"] + 1e-4, moves[i]
    state = wire.to_pgx(some_positions(wire, 1)[0])
    assert agent.choose(state)[0] == agent.choose(state)[0]


def test_value_only_batches_concurrent_requests(wire: Wire) -> None:
    agent = agent_for(wire, value_only=True)
    positions = some_positions(wire, 6)
    results: dict[int, list[int]] = {}
    agent.batches.clear()

    def ask(i: int) -> None:
        results[i] = agent.choose(wire.to_pgx(positions[i]))[0]

    threads = [threading.Thread(target=ask, args=(i,)) for i in range(len(positions))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i, state in enumerate(positions):
        assert legal_move(wire, state, results[i]), f"request {i} got a move for another position"
    assert max(agent.batches) > 1, f"requests were never batched: {agent.batches}"


def test_policy_only_and_value_only_are_exclusive(wire: Wire) -> None:
    with pytest.raises(ValueError, match="choose one"):
        Agent(
            CHECKPOINTS[wire.game_id],
            num_simulations=8,
            policy_only=True,
            value_only=True,
            allow_rules_mismatch=ALLOW_MISMATCH,
        )


def test_concurrent_requests_from_different_games_are_batched_and_kept_apart(wire: Wire) -> None:
    agent = agent_for(wire)
    positions = some_positions(wire, 8)
    results: dict[int, list[int]] = {}
    agent.batches.clear()

    def ask(i: int) -> None:
        results[i] = agent.choose(wire.to_pgx(positions[i]))[0]

    threads = [threading.Thread(target=ask, args=(i,)) for i in range(len(positions))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Each answer is legal for its own position, not some other request's.
    for i, state in enumerate(positions):
        assert legal_move(wire, state, results[i]), f"request {i} got a move for another position"
    assert max(agent.batches) > 1, f"requests were never batched: {agent.batches}"

"""Gumbel MCTS over a pgx AlphaZero checkpoint, batched across concurrent requests.

The search mirrors `ModelAgent` in pgx's `examples/alphazero/interactive_tournament.py`: the same
network, the same `recurrent_fn` (including the +1 discount between the stages of one move), the
same `gumbel_muzero_policy` with no Gumbel noise. A move is one full search per stage, exactly as
that tournament plays it: three for Epaminondas, two for Gess.

Which game a process plays is the checkpoint's to say. It names the pgx env it was trained on,
that picks the `Wire`, and the client advertises the one game id that goes with it. A wire id
covers every pgx version with its rules, so checkpoints from either side of an observation change
play each other here; each is fed the leading planes it was trained on.

With `policy_only`, no search runs at all: the move is the legal action with the largest policy
logit, one network evaluation per decision. That is the network's own opinion, unimproved.

With `value_only`, no search runs either, and the policy head is not consulted: every legal move
is played out in full (all of its actions) and the value head scores the position it leaves. The
move whose successor is best for the mover is played - a one-ply search over the value alone. In
a game of chance (Pig) a move leads to one of several positions, one per roll of the die, and the
score is their average: each roll is played, not sampled.

The search does sample. Every step it takes draws its roll from its own key, as pgx's self-play
did; mctx keeps one successor per edge, so an edge's value rests on that one roll.

What differs is that requests arrive concurrently from many games. Each stage of each request is
put on a queue; one worker thread takes whatever is waiting (up to `max_batch`), pads it to a
power of two so jit compiles only a handful of shapes, and runs one batched search for all of it.
The model is the only state held, and it is shared by every game.
"""

from __future__ import annotations

import pickle
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mctx
import numpy as np
import pgx

from .wire import Wire, for_env_id

Model = tuple[Any, Any]


def alphazero_dir() -> Path:
    """Where pgx keeps its AlphaZero training code (`config.py`, `network.py`)."""
    return Path(pgx.__file__).resolve().parent.parent / "examples" / "alphazero"


def _use_alphazero_modules() -> None:
    """Unpickling a checkpoint needs pgx's training `config` module, and building the network
    needs `network`: make their directory importable."""
    if str(alphazero_dir()) not in sys.path:
        sys.path.insert(0, str(alphazero_dir()))


@dataclass
class Checkpoint:
    """A loaded checkpoint, and the game it says it plays."""

    config: Any
    model: Model
    wire: Wire


class _Unpickler(pickle.Unpickler):
    """Older checkpoints, the g_hex ones among them, were written by a training script that
    defined `Config` itself, so they name it `__main__.Config`. That is pgx's `config.Config`."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("__main__", "Config"):
            module = "config"
        return super().find_class(module, name)


def read_checkpoint(path: Path) -> dict[str, Any]:
    """A checkpoint's contents, whichever script pickled its config."""
    _use_alphazero_modules()
    with open(path, "rb") as f:
        ckpt: dict[str, Any] = _Unpickler(f).load()
    return ckpt


def load_checkpoint(path: Path, allow_rules_mismatch: bool = False) -> Checkpoint:
    _use_alphazero_modules()
    from config import Config  # type: ignore[import-not-found]

    ckpt = read_checkpoint(path)
    env_id = ckpt.get("env_id", "epaminondas")
    try:
        wire = for_env_id(env_id)
    except ValueError as e:
        raise ValueError(f"{path} is a {env_id} checkpoint: {e}") from e
    version = ckpt.get("env_version")
    if version not in wire.rules_versions and not allow_rules_mismatch:
        known = ", ".join(wire.rules_versions)
        raise ValueError(
            f"{path} was trained on pgx {env_id} {version}, but {wire.game_id} is "
            f"{known}. Pass --allow-rules-mismatch to play it anyway."
        )
    # Rehydrate through the current Config so older checkpoints get newer fields' defaults.
    config = Config(**ckpt["config"].__dict__)
    return Checkpoint(config, ckpt["model"], wire)


def stem_inputs(params: Any) -> int | None:
    """Channels the stem convolution takes, or None if there is no `stem_conv` (the MLP nets).

    The weight's shape is (kh, kw, in, out), so its third axis is the count.
    """
    if not hasattr(params, "items"):
        return None
    for module, entries in params.items():
        if module.endswith("stem_conv"):
            w = entries.get("w")
            if getattr(w, "ndim", 0) == 4:
                return int(w.shape[2])
    return None


def _moves(state: Any, batched: bool = True) -> Any:
    """The move list a gess_joint network takes with the observation; None for other games."""
    moves = getattr(state, "_moves", None)
    if moves is None or batched:
        return moves
    return moves[None]


def obs_channels(model: Model, forward: Any, env: Any) -> int | None:
    """Observation planes the checkpoint was trained on, or None when it cannot be told.

    Read off the stem convolution, but not directly: a network may feed the stem more than the
    observation. GessFormer pads the board and adds an is_border plane, so its stem takes five
    channels for a four-plane observation. So build the same architecture against the env
    installed now, and measure the gap between what its stem takes and what the env provides -
    whatever the network adds, it adds to both.
    """
    trained = stem_inputs(model[0])
    if trained is None:
        return None
    start = env.init(jax.random.PRNGKey(0))
    reference, _ = forward.init(
        jax.random.PRNGKey(0),
        start.observation[None],
        is_eval=True,
        moves=_moves(start, batched=False),
    )
    provided = stem_inputs(reference)
    if provided is None:
        return None
    return int(env.observation_shape[-1]) - (provided - trained)


def narrow(observation: jnp.ndarray, channels: int | None) -> jnp.ndarray:
    """The leading `channels` planes of an observation: what a checkpoint from before later
    planes were added saw in training. `None`, or a count that already matches, passes it
    through."""
    if channels is None or channels == observation.shape[-1]:
        return observation
    return observation[..., :channels]


def _bucket(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 1 else 1


#: The most successor positions stepped or scored in one call, in value-only mode. Chunks are
#: padded to a power of two between `_MIN_CHUNK` and this, so few shapes are ever compiled.
VALUE_CHUNK = 1024
_MIN_CHUNK = 16


def _chunk_size(n: int) -> int:
    return min(max(_bucket(n), _MIN_CHUNK), VALUE_CHUNK)


@dataclass
class _Job:
    state: Any  # unbatched pgx State
    done: threading.Event = field(default_factory=threading.Event)
    action: int = -1
    value: float = 0.0
    # Value-only mode decides a whole move at once: all of its actions, and how many moves it
    # chose between.
    actions: list[int] = field(default_factory=list)
    successors: int = 0
    error: BaseException | None = None


class Agent:
    def __init__(
        self,
        checkpoint: Path | Checkpoint,
        num_simulations: int,
        max_considered: int = 16,
        max_batch: int = 16,
        seed: int = 0,
        allow_rules_mismatch: bool = False,
        policy_only: bool = False,
        value_only: bool = False,
    ) -> None:
        _use_alphazero_modules()
        from network import (  # type: ignore[import-not-found]
            make_forward,
            mlp_input_features,
        )

        if isinstance(checkpoint, Path):
            checkpoint = load_checkpoint(checkpoint, allow_rules_mismatch)
        self.wire = checkpoint.wire
        self.env = self.wire.env
        self.config, self.model = checkpoint.config, checkpoint.model
        self.forward = make_forward(self.env.num_actions, self.config)
        # pgx appends observation planes and never reorders them, so a checkpoint from before a
        # plane was added is shown exactly what it saw in training by dropping the trailing ones.
        # That is what keeps the whole v6-to-v11 family playable side by side (pgx does the same
        # in `examples/alphazero/model_tournament.py::narrow_observation`). It is sound only while
        # the append-only invariant holds: if a plane ever changes meaning, pgx bumps the rules
        # version and `wire.rules_versions` drops it.
        self.obs_channels = obs_channels(self.model, self.forward, self.env)
        if self.obs_channels is None:
            # An MLP (Pig's) takes a flat observation that grows at the end in the same way: v0
            # had six features and v1 seven. Its first layer says how many it was built for.
            self.obs_channels = mlp_input_features(self.model[0], self.config)
        env_channels = int(self.env.observation_shape[-1])
        if self.obs_channels is not None and self.obs_channels > env_channels:
            raise ValueError(
                f"the checkpoint was trained on {self.obs_channels} observation planes but "
                f"pgx {self.wire.env_id} now provides {env_channels}: planes it needs are gone, "
                "so it cannot play against this env."
            )
        if policy_only and value_only:
            raise ValueError("policy-only and value-only are alternatives: choose one")
        self.policy_only = policy_only
        self.value_only = value_only
        self.num_simulations = 0 if policy_only or value_only else num_simulations
        self.max_considered = max_considered
        self.max_batch = max_batch
        self._key = jax.random.PRNGKey(seed)
        self._key_lock = threading.Lock()
        self._search = jax.jit(self._run, static_argnames=("num_simulations",))
        self._step = jax.jit(self.env.step)
        self._step_batch = jax.jit(self._step_all)
        self._leaf_scores = jax.jit(self._scores)
        self._jobs: queue.Queue[_Job] = queue.Queue()
        self.batches: list[int] = []  # sizes of the batches run, for tests and diagnostics
        threading.Thread(target=self._worker, daemon=True).start()

    def _recurrent_fn(
        self, model: Model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State
    ) -> tuple[mctx.RecurrentFnOutput, pgx.State]:
        params, net_state = model
        current_player = state.current_player
        # Each node draws its own chance outcome; a game without chance ignores the keys.
        keys = jax.random.split(rng_key, action.shape[0])
        state = jax.vmap(self.env.step)(state, action, keys)
        (logits, value), _ = self.forward.apply(
            params,
            net_state,
            narrow(state.observation, self.obs_channels),
            is_eval=True,
            moves=_moves(state),
        )
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
        value = jnp.where(state.terminated, 0.0, value)
        # +1 while the same player is still choosing the rest of a move, -1 when the turn has
        # passed, 0 at the end: one move is several actions by the same player.
        discount = jnp.where(state.current_player == current_player, 1.0, -1.0)
        discount = jnp.where(state.terminated, 0.0, discount)
        return (
            # mctx's chex dataclasses defeat mypy's view of their constructors.
            mctx.RecurrentFnOutput(  # type: ignore[call-arg]
                reward=reward, discount=discount, prior_logits=logits, value=value
            ),
            state,
        )

    def _run(self, key: jnp.ndarray, states: pgx.State, num_simulations: int):
        params, net_state = self.model
        (logits, value), _ = self.forward.apply(
            params,
            net_state,
            narrow(states.observation, self.obs_channels),
            is_eval=True,
            moves=_moves(states),
        )
        if self.policy_only:
            masked = jnp.where(states.legal_action_mask, logits, -jnp.inf)
            return jnp.argmax(masked, axis=-1), value
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=states)  # type: ignore[call-arg]
        out = mctx.gumbel_muzero_policy(
            params=self.model,
            rng_key=key,
            root=root,
            invalid_actions=~states.legal_action_mask,
            recurrent_fn=self._recurrent_fn,
            num_simulations=num_simulations,
            max_num_considered_actions=self.max_considered,
            gumbel_scale=0.0,
        )
        return out.action, value

    def _step_all(self, parents: pgx.State, actions: jnp.ndarray) -> pgx.State:
        """Step each parent by its action. Only a move's last action may involve chance (Pig's
        only has one), so the earlier ones take the first outcome's key, whatever it is."""
        return jax.vmap(self.env.step, in_axes=(0, 0, None))(
            parents, actions, self.wire.chance_keys[0]
        )

    def _scores(self, parents: pgx.State, actions: jnp.ndarray) -> jnp.ndarray:
        """Take each parent's last action of a move, and score the position it leaves from the
        point of view of the player who moved: the result if the game is over, and otherwise
        the value head's opinion, negated when it is the opponent's turn. Where chance decides
        the position, the average over its equally likely outcomes."""
        params, net_state = self.model
        mover = parents.current_player

        def score(key: jnp.ndarray) -> jnp.ndarray:
            states = jax.vmap(self.env.step, in_axes=(0, 0, None))(parents, actions, key)
            (_, value), _ = self.forward.apply(
                params,
                net_state,
                narrow(states.observation, self.obs_channels),
                is_eval=True,
                moves=_moves(states),
            )
            value = jnp.where(states.current_player == mover, value, -value)
            reward = states.rewards[jnp.arange(mover.shape[0]), mover]
            return jnp.where(states.terminated, reward, value)

        return jax.vmap(score)(self.wire.chance_keys).mean(axis=0)

    def _in_chunks(self, fn: Any, states: Any, rows: np.ndarray, actions: np.ndarray) -> list[Any]:
        """`fn(states[rows], actions)`, in padded chunks of at most VALUE_CHUNK, each trimmed
        back to its real length and brought to the host."""
        out = []
        for start in range(0, len(rows), VALUE_CHUNK):
            r, a = rows[start : start + VALUE_CHUNK], actions[start : start + VALUE_CHUNK]
            pad = _chunk_size(len(r)) - len(r)
            r, a = (
                np.concatenate([r, np.repeat(r[:1], pad)]),
                np.concatenate([a, np.repeat(a[:1], pad)]),
            )
            parents = jax.tree_util.tree_map(lambda x, r=r: x[r], states)
            result = fn(parents, jnp.asarray(a, dtype=jnp.int32))
            n = len(r) - pad
            out.append(jax.tree_util.tree_map(lambda x, n=n: np.asarray(x)[:n], result))
        return out

    def _best_moves(self, roots: list[pgx.State]) -> list[tuple[list[int], float, int]]:
        """For each root, the move whose successor scores best for the mover, its score, and
        how many legal moves there were.

        Every root's moves are expanded together, one stage at a time: each state in the
        frontier is stepped by each of its legal actions, until the last action of the move,
        whose successors are scored instead of kept."""
        frontier = jax.tree_util.tree_map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *roots)
        owner = np.arange(len(roots))
        paths = np.zeros((len(roots), 0), dtype=np.int64)
        stages = self.wire.actions_per_move
        scores = np.zeros(0, dtype=np.float32)
        for stage in range(stages):
            # A finished game's mask is all ones in pgx; it has no moves to expand.
            mask = frontier.legal_action_mask & ~frontier.terminated[:, None]
            rows, actions = np.nonzero(mask)
            owner, paths = owner[rows], np.column_stack([paths[rows], actions])
            if stage < stages - 1:
                chunks = self._in_chunks(self._step_batch, frontier, rows, actions)
                frontier = jax.tree_util.tree_map(lambda *xs: np.concatenate(xs), *chunks)
            else:
                scores = np.concatenate(self._in_chunks(self._leaf_scores, frontier, rows, actions))
        results = []
        for i in range(len(roots)):
            mine = np.flatnonzero(owner == i)
            if len(mine) == 0:
                raise ValueError("the position has no legal move")
            best = mine[np.argmax(scores[mine])]
            results.append(([int(a) for a in paths[best]], float(scores[best]), len(mine)))
        return results

    def _next_key(self) -> jnp.ndarray:
        with self._key_lock:
            self._key, sub = jax.random.split(self._key)
            return sub

    def _worker(self) -> None:
        while True:
            jobs = [self._jobs.get()]
            # Whatever else is already waiting joins this batch; no deliberate delay.
            while len(jobs) < self.max_batch:
                try:
                    jobs.append(self._jobs.get_nowait())
                except queue.Empty:
                    break
            try:
                if self.value_only:
                    for job, (actions, score, n) in zip(
                        jobs, self._best_moves([j.state for j in jobs]), strict=True
                    ):
                        job.actions, job.value, job.successors = actions, score, n
                    self.batches.append(len(jobs))
                    for job in jobs:
                        job.done.set()
                    continue
                size = min(_bucket(len(jobs)), _bucket(self.max_batch))
                padded = [j.state for j in jobs] + [jobs[0].state] * (size - len(jobs))
                batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *padded)
                actions, values = self._search(
                    self._next_key(), batch, num_simulations=self.num_simulations
                )
                actions, values = np.asarray(actions), np.asarray(values)
                self.batches.append(len(jobs))
                for i, job in enumerate(jobs):
                    job.action, job.value = int(actions[i]), float(values[i])
            except BaseException as e:  # noqa: BLE001 - hand it to every waiting request
                for job in jobs:
                    job.error = e
            for job in jobs:
                job.done.set()

    def warm_up(self, report: Any = None) -> None:
        """Compile every batch size the worker can use, so no compile lands mid-tournament and
        blows a move's time limit."""
        start = self.env.init(jax.random.PRNGKey(0))
        if self.value_only:
            size = _MIN_CHUNK
            while size <= VALUE_CHUNK:
                batch = jax.tree_util.tree_map(lambda x, size=size: jnp.stack([x] * size), start)
                actions = jnp.zeros(size, dtype=jnp.int32)
                jax.block_until_ready(self._step_batch(batch, actions))
                jax.block_until_ready(self._leaf_scores(batch, actions))
                if report:
                    report(size)
                size *= 2
            self._best_moves([start])  # and whatever the rest of the path compiles
            return
        size = 1
        while size <= _bucket(self.max_batch):
            batch = jax.tree_util.tree_map(lambda x, size=size: jnp.stack([x] * size), start)
            jax.block_until_ready(
                self._search(self._next_key(), batch, num_simulations=self.num_simulations)
            )
            if report:
                report(size)
            size *= 2
        if self.wire.actions_per_move > 1:
            first_legal = int(np.flatnonzero(np.asarray(start.legal_action_mask))[0])
            jax.block_until_ready(self._step(start, jnp.int32(first_legal), self._next_key()))

    def _submit(self, state: pgx.State) -> _Job:
        job = _Job(state)
        self._jobs.put(job)
        job.done.wait()
        if job.error is not None:
            raise job.error
        return job

    def _decide(self, state: pgx.State) -> tuple[int, float | None]:
        legal = np.flatnonzero(np.asarray(state.legal_action_mask))
        if len(legal) == 1:
            return int(legal[0]), None  # forced: no search needed
        job = self._submit(state)
        return job.action, job.value

    def choose(self, state: pgx.State) -> tuple[list[int], dict[str, Any]]:
        """The actions of one move — one decision per stage — and what to record about it."""
        started = time.monotonic()
        if self.value_only:
            job = self._submit(state)
            return job.actions, {
                "sims": 0,
                "elapsed_ms": round(1000 * (time.monotonic() - started)),
                "value_only": True,
                "successors": job.successors,
                "eval": job.value,
            }
        actions: list[int] = []
        value: float | None = None
        for stage in range(self.wire.actions_per_move):
            action, v = self._decide(state)
            if value is None:
                value = v
            actions.append(action)
            if stage < self.wire.actions_per_move - 1:
                state = self._step(state, jnp.int32(action), self._next_key())
        info: dict[str, Any] = {
            "sims": self.num_simulations,
            "elapsed_ms": round(1000 * (time.monotonic() - started)),
        }
        if self.policy_only:
            info["policy_only"] = True
        if value is not None:
            info["eval"] = value
        return actions, info

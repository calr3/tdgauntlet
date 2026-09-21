"""Gumbel MCTS over a pgx AlphaZero checkpoint, batched across concurrent requests.

The search mirrors `ModelAgent` in pgx's `examples/alphazero/interactive_tournament.py`: the same
network, the same `recurrent_fn` (including the +1 discount between the stages of one move), the
same `gumbel_muzero_policy` with no Gumbel noise. A move is one full search per stage, exactly as
that tournament plays it: three for Epaminondas, two for Gess.

Which game a process plays is the checkpoint's to say. It names the pgx env it was trained on,
that picks the `Wire`, and the client advertises the one game id that goes with it.

With `policy_only`, no search runs at all: the move is the legal action with the largest policy
logit, one network evaluation per decision. That is the network's own opinion, unimproved.

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


def load_checkpoint(path: Path, allow_rules_mismatch: bool = False) -> Checkpoint:
    _use_alphazero_modules()
    from config import Config  # type: ignore[import-not-found]

    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    env_id = ckpt.get("env_id", "epaminondas")
    try:
        wire = for_env_id(env_id)
    except ValueError as e:
        raise ValueError(f"{path} is a {env_id} checkpoint: {e}") from e
    version = ckpt.get("env_version")
    if version != wire.rules_version and not allow_rules_mismatch:
        raise ValueError(
            f"{path} was trained on pgx {env_id} {version}, but {wire.game_id} is "
            f"{wire.rules_version}. Pass --allow-rules-mismatch to play it anyway."
        )
    # Rehydrate through the current Config so older checkpoints get newer fields' defaults.
    config = Config(**ckpt["config"].__dict__)
    return Checkpoint(config, ckpt["model"], wire)


def _bucket(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 1 else 1


@dataclass
class _Job:
    state: Any  # unbatched pgx State
    done: threading.Event = field(default_factory=threading.Event)
    action: int = -1
    value: float = 0.0
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
    ) -> None:
        _use_alphazero_modules()
        from network import make_forward  # type: ignore[import-not-found]

        if isinstance(checkpoint, Path):
            checkpoint = load_checkpoint(checkpoint, allow_rules_mismatch)
        self.wire = checkpoint.wire
        self.env = self.wire.env
        self.config, self.model = checkpoint.config, checkpoint.model
        self.forward = make_forward(self.env.num_actions, self.config)
        self.policy_only = policy_only
        self.num_simulations = 0 if policy_only else num_simulations
        self.max_considered = max_considered
        self.max_batch = max_batch
        self._key = jax.random.PRNGKey(seed)
        self._key_lock = threading.Lock()
        self._search = jax.jit(self._run, static_argnames=("num_simulations",))
        self._step = jax.jit(self.env.step)
        self._jobs: queue.Queue[_Job] = queue.Queue()
        self.batches: list[int] = []  # sizes of the batches run, for tests and diagnostics
        threading.Thread(target=self._worker, daemon=True).start()

    def _recurrent_fn(
        self, model: Model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State
    ) -> tuple[mctx.RecurrentFnOutput, pgx.State]:
        del rng_key
        params, net_state = model
        current_player = state.current_player
        state = jax.vmap(self.env.step)(state, action)
        (logits, value), _ = self.forward.apply(params, net_state, state.observation, is_eval=True)
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
        (logits, value), _ = self.forward.apply(params, net_state, states.observation, is_eval=True)
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
        size = 1
        while size <= _bucket(self.max_batch):
            batch = jax.tree_util.tree_map(lambda x, size=size: jnp.stack([x] * size), start)
            jax.block_until_ready(
                self._search(self._next_key(), batch, num_simulations=self.num_simulations)
            )
            if report:
                report(size)
            size *= 2
        first_legal = int(np.flatnonzero(np.asarray(start.legal_action_mask))[0])
        jax.block_until_ready(self._step(start, jnp.int32(first_legal)))

    def _decide(self, state: pgx.State) -> tuple[int, float | None]:
        legal = np.flatnonzero(np.asarray(state.legal_action_mask))
        if len(legal) == 1:
            return int(legal[0]), None  # forced: no search needed
        job = _Job(state)
        self._jobs.put(job)
        job.done.wait()
        if job.error is not None:
            raise job.error
        return job.action, job.value

    def choose(self, state: pgx.State) -> tuple[list[int], dict[str, Any]]:
        """The actions of one move — one decision per stage — and what to record about it."""
        started = time.monotonic()
        actions: list[int] = []
        value: float | None = None
        for _ in range(self.wire.actions_per_move):
            action, v = self._decide(state)
            if value is None:
                value = v
            actions.append(action)
            state = self._step(state, jnp.int32(action))
        info: dict[str, Any] = {
            "sims": self.num_simulations,
            "elapsed_ms": round(1000 * (time.monotonic() - started)),
        }
        if self.policy_only:
            info["policy_only"] = True
        if value is not None:
            info["eval"] = value
        return actions, info

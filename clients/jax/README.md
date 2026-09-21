# JAX client

A client that runs inside the **pgx environment** and reuses pgx's own code rather than copying
it. It installs nothing: the HTTP side is `clients/pysdk`, which is standard library only,
so the JAX/CUDA versions pgx depends on are left alone.

## `tdg_jax` — a checkpoint under Gumbel MCTS

```bash
cd clients/jax
JAX_PLATFORMS=cpu python -m tdg_jax --port 9313 --sims 256 \
    --checkpoint ~/calr3gh/pgx/checkpoints/<run>/<step>.ckpt
```

The search is `ModelAgent` from `examples/alphazero/interactive_tournament.py`: the same network,
the same `recurrent_fn`, `gumbel_muzero_policy` with no Gumbel noise, three searches per move (one
per action). Concurrent requests from different games are searched together in one batch.

- It refuses a checkpoint whose `env_version` is not the rules `epaminondas/1` plays (v6);
  `--allow-rules-mismatch` overrides that.
- Every batch size up to `--max-batch` is compiled before `/health` reports ready, so no
  compilation happens mid-tournament.
- The simulation count is fixed at startup (`--sims`); the millisecond budget is not used.
- Omit `JAX_PLATFORMS=cpu` to use the GPU, if a training run is not already on it.

### `--policy-only`

```bash
JAX_PLATFORMS=cpu python -m tdg_jax --port 9314 --policy-only \
    --checkpoint ~/calr3gh/pgx/checkpoints/<run>/<step>.ckpt
```

No search: each of the three actions is the legal action with the largest policy logit, one
network evaluation per decision. It is the network's raw opinion, so a gauntlet against the same
checkpoint under `--sims` measures what the search adds. `--sims` is ignored and reported as 0,
the move is deterministic for a given position, and the default client name gains a `-policy`
suffix. `examples/policy_v_mcts.toml` sets that comparison up.

## Tests

```bash
JAX_PLATFORMS=cpu python -m pytest -p no:xdist clients/jax
```

`test_wire.py` checks that a position rebuilt from the wire is pgx's own state field for field
(observation and legal mask included), and that every conformance-vector move, played as three
pgx actions, reaches the server's successor. `test_agent.py` uses the newest current-rules
checkpoint under `~/calr3gh/pgx/checkpoints` (or `TDG_JAX_CHECKPOINT`), and skips without one.

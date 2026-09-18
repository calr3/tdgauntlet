# JAX clients

Two clients that run inside the **pgx environment** and reuse pgx's own code rather than copying
it. Neither installs anything: the HTTP side is `clients/pysdk`, which is standard library only,
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

## `tdg_jax.negamax` — pgx's alpha-beta engine

```bash
JAX_PLATFORMS=cpu python -m tdg_jax.negamax --port 9314
```

`examples/alphazero/negamax.py`, the fixed reference in `EPAMINONDAS_EXPERIMENTS.md`, over the
protocol. A move's millisecond budget is split across its three actions. It searches one position
at a time.

## Tests

```bash
JAX_PLATFORMS=cpu python -m pytest -p no:xdist clients/jax
```

`test_wire.py` checks that a position rebuilt from the wire is pgx's own state field for field
(observation and legal mask included), and that every conformance-vector move, played as three
pgx actions, reaches the server's successor. `test_agent.py` uses the newest current-rules
checkpoint under `~/calr3gh/pgx/checkpoints` (or `TDG_JAX_CHECKPOINT`), and skips without one.

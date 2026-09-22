# JAX client

A client that runs inside the **pgx environment** and reuses pgx's own code rather than copying
it. It plays whichever game its checkpoint was trained for. It installs nothing: the HTTP side is `clients/pysdk`, which is standard library only,
so the JAX/CUDA versions pgx depends on are left alone.

## `tdg_jax` — a checkpoint under Gumbel MCTS

```bash
cd clients/jax
JAX_PLATFORMS=cpu python -m tdg_jax --port 9313 --sims 256 \
    --checkpoint ~/calr3gh/pgx/checkpoints/<run>/<step>.ckpt
```

**The checkpoint decides the game.** It records the pgx env it was trained on, and the client
advertises the one tdgauntlet game that goes with it and nothing else: `epaminondas` becomes
`epaminondas/1`, `gess` becomes `gess/1`. One process, one checkpoint, one game.

The search is `ModelAgent` from `examples/alphazero/interactive_tournament.py`: the same network,
the same `recurrent_fn`, `gumbel_muzero_policy` with no Gumbel noise, one search per action —
three for an Epaminondas move, two for a Gess move (source, then destination). Concurrent
requests from different games are searched together in one batch.

- It refuses a checkpoint whose `env_version` is not the rules its game plays (`epaminondas/1`
  is pgx v6 through v11, which are one set of rules with a growing observation; `gess/1` is pgx
  v1); `--allow-rules-mismatch` overrides that.
- Within that family the observation is narrowed to the planes each checkpoint's stem was built
  for, so models from either side of a plane being added play each other unchanged.
- Every batch size up to `--max-batch` is compiled before `/health` reports ready, so no
  compilation happens mid-tournament.
- The simulation count is fixed at startup (`--sims`); the millisecond budget is not used.
- Omit `JAX_PLATFORMS=cpu` to use the GPU, if a training run is not already on it.

### `--policy-only`

```bash
JAX_PLATFORMS=cpu python -m tdg_jax --port 9314 --policy-only \
    --checkpoint ~/calr3gh/pgx/checkpoints/<run>/<step>.ckpt
```

No search: each action of a move is the legal action with the largest policy logit, one network
evaluation per decision. It is the network's raw opinion, so a gauntlet against the same
checkpoint under `--sims` measures what the search adds. `--sims` is ignored and reported as 0,
the move is deterministic for a given position, and the default client name gains a `-policy`
suffix. `examples/policy_v_mcts.toml` sets that comparison up.

## Tests

```bash
JAX_PLATFORMS=cpu python -m pytest -p no:xdist clients/jax
```

`test_wire.py` runs for both games: a position rebuilt from the wire is pgx's own state field
for field (observation and legal mask included), and every conformance-vector move, played as
pgx actions, reaches the server's successor. The one field it does not carry is Gess's `source`,
which pgx leaves at the last move's and nothing reads between moves; there is a test for that.

`test_agent.py` uses the newest run for each game under `~/calr3gh/pgx/checkpoints` (or
`TDG_JAX_CHECKPOINT`), and skips a game whose newest run was trained on other rules — an older
run with the right rules label is no substitute, since pgx changes an env's observation with its
rules and that checkpoint's network no longer fits the env installed now.

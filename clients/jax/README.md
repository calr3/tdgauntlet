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
`epaminondas/1`, `gess` becomes `gess/1`, and `g_hex` becomes `one-more-tri/1`. One process, one
checkpoint, one game.

The search is `ModelAgent` from `examples/alphazero/interactive_tournament.py`: the same network,
the same `recurrent_fn`, `gumbel_muzero_policy` with no Gumbel noise, one search per action —
three for an Epaminondas move, two for a Gess move (source, then destination). Concurrent
requests from different games are searched together in one batch.

- It refuses a checkpoint whose `env_version` is not the rules its game plays (`epaminondas/1`
  is pgx v6 through v11, which are one set of rules with a growing observation; `gess/1` is pgx
  v1; `one-more-tri/1` is pgx g_hex v1); `--allow-rules-mismatch` overrides that. Every g_hex
  checkpoint so far is v0, whose 10 ignored the tile on 4, so it needs the flag.
- One More Tri's move is a single pgx action (`tri * 10 + tile - 1`), so a move is one search.
  The g_hex checkpoints predate the current training script: they pickled their config as
  `__main__.Config`, which `read_checkpoint` maps to pgx's `config.Config`. Their network, the older
  `az_net` ResNet, has no `stem_conv` to measure, so the observation reaches it unchanged; g_hex
  has only ever had the one observation.
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

### `--value-only`

```bash
JAX_PLATFORMS=cpu python -m tdg_jax --port 9315 --value-only \
    --checkpoint ~/calr3gh/pgx/checkpoints/<run>/<step>.ckpt
```

The other half of the network, also without search: every legal move is played out in full (all
three actions of an Epaminondas move, both of a Gess move) and the value head scores the position
each one leaves, from the mover's side: the result if the game is over, otherwise the value
negated, since it is the opponent's turn. The best-scoring move is played, and its score is the
reported `eval`; `successors` says how many moves were scored. The policy head is never read.

It costs one network evaluation per legal move rather than one per action, so it is slower than
`--policy-only`: successors are stepped and scored in padded chunks of up to 1024 (all the batched
requests' together), and every chunk size is compiled before `/health` reports ready. Like
`--policy-only` it is deterministic, `--sims` is ignored and reported as 0, and the default client
name gains a `-value` suffix. The two flags cannot be combined.

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

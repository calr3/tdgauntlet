# CLAUDE.md

Working notes for this repo: how to build, test and measure things here, and the mistakes that
have already been made once.

## What this is

A tournament server for board-game agents. The Rust server owns the rules, clock, adjudication
and score; clients answer "what is your move in this position?" over HTTP and keep **no per-game
state**. Read `README.md` for the overview and `PROTOCOL.md` for the wire contract. The games are
Epaminondas, Gess and One More Tri (pgx's `g_hex`). Gess and One More Tri are played by the
game-agnostic clients (`random`, `keyboard`, `web`, the Python SDK) and by `clients/jax`, whose
checkpoint decides its game; `alphabeta` and `mcts` are Epaminondas only.

It sits next to `~/calr3gh/pgx` (JAX training, the reference rules, checkpoints) and replaces
pgx's `examples/alphazero/interactive_tournament.py`. Nothing here should ever edit pgx.

## Environment

- Rust is installed via rustup in `~/.cargo`, not on the default PATH. Start every shell command
  that needs it with `export PATH="$HOME/.cargo/bin:$PATH"`.
- Python work that touches JAX or pgx uses `~/.venv/bin/python3` (the pgx environment). **Never
  install anything into `~/.venv`**: JAX and its CUDA plugin must stay in lockstep, and an upgrade
  silently loses the GPU. Tools like ruff go in a throwaway venv in the scratchpad.
- No Docker, no `gh` CLI on this machine.
- The GPU is usually busy with a pgx training run. Run JAX here with `JAX_PLATFORMS=cpu` unless
  the user says the GPU is free. When it is, give each JAX process
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.3`, or the first grabs 75% and the second fails.
- Leave CPU for training: at most ~16-20 of the 24 cores for long jobs.

## Commands

```bash
cargo build --release
cargo test --release --workspace                    # includes the pgx conformance gate
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
./target/release/tdgauntlet run examples/<config>.toml
./target/release/tdgauntlet check http://127.0.0.1:<port>
python -m pytest                                    # the Python SDK; JAX tests skip without pgx
JAX_PLATFORMS=cpu ~/.venv/bin/python3 -m pytest -p no:xdist clients/jax   # in the pgx env
```

CI (`.github/workflows/ci.yml`) runs fmt, clippy with warnings as errors, all Rust tests, ruff,
ruff format, mypy and pytest. Run the same before committing.

## Rules and conformance

- `epaminondas/1` is pgx's Epaminondas at **env version v6** (100-move quiet clock, advancement
  tiebreak, no draws) and `gess/1` is pgx's Gess at **v1**. The Rust engines in `crates/games` are
  independent and are held to pgx by `crates/games/tests/conformance.rs` and
  `gess_conformance.rs`, which replay vectors dumped from pgx.
- **If pgx's rules change, the vectors must be regenerated and committed**, in the pgx env:
  `JAX_PLATFORMS=cpu ~/.venv/bin/python3 conformance/dump_pgx.py` (positions and perft; perft(3)
  takes ~15 minutes, run it detached) and `conformance/dump_eval.py` (alpha-beta evaluator
  reference). The test refuses vectors from uncommitted pgx rules; check `git status` in pgx.
- pgx has changed its rules mid-project before (v5 to v6 happened during the first session).
  Before porting or trusting a rule, check pgx's working copy against its HEAD.
- **pgx's Epaminondas is at v11, and v6 to v11 are all the same rules**: v7 onwards only grew
  the observation, from 7 planes to 19. Regenerating `conformance/epaminondas.jsonl.gz` from pgx
  v11 reproduces the committed v6 vectors byte for byte, which is the check that says so, and
  `clients/jax` accepts the whole family (`wire.rules_versions`) rather than the one label. It
  narrows the observation to the planes each checkpoint's stem was built for, so a v6 model and a
  v11 model play each other unchanged and no `--allow-rules-mismatch` is needed for either.
  Gess is unaffected: pgx is on v1 and so is `gess/1`.
- A coverage test fails if the vectors stop exercising captures, incursions and the clock. Plain
  random play almost never produces captures or incursions, which is why `dump_pgx.py` mixes in
  an aggressive policy. Keep it.
- The same goes for `dump_gess.py`, for the opposite reason: random play in Gess is mostly
  suicide (lifting a piece from around your own ring destroys it and loses), so those games are
  three moves long. Two of its three policies only play moves that keep their own ring, which is
  what makes a game long enough to reach a capture or the 20-move quiet clock.
- Gess regenerates in about a minute: `JAX_PLATFORMS=cpu ~/.venv/bin/python3
  conformance/dump_gess.py`. Its `moves` counter is tdgauntlet's own; pgx's state has none.
- `one-more-tri/1` is pgx's **`g_hex` at v1**. v0 listed 10 as a neighbour of 4 but not 4 of
  10; v1 fixed that in both codebases. Every g_hex checkpoint so far is v0, trained on the old
  rule, so the JAX client needs `--allow-rules-mismatch` to play one. `dump_one_more_tri.py`
  regenerates in ~20 s, and half its games keep 4 and 10 empty to the end so that edge stays
  covered. pgx randomises which seat plays
  colour 0; tdgauntlet does not need to, since minimatches swap colours.
- Anything that walks a random game to a fixed depth has to cope with a game that ends first.
  `check.rs` used to retry the same depth for ever and hung on Gess.
- pgx's `EpaminondasCli` labels board row 0 as rank 12, contradicting its own docstring. Here,
  square 0 is `a1` and white's home is rank 1. Don't copy pgx's labels.

## Writing code here

- Write tests with each change, not afterwards. Every crate and client has them; follow the
  existing style (hand-built positions written in notation, e.g. `pos(Side::First, "c1 c2", "c5")`).
- When a test fails, check the test before the engine. In this repo's history most first failures
  were wrong test setups: a black piece left on white's home rank, a square index miscounted, a
  hand-computed move count off at the board edge.
- **A timing test must measure the machine, not assume it.** CI builds without optimisation on a
  shared runner, where the searches are 10-30x slower than a local release build, so both
  deadline tests failed there for months while passing here (`cargo test --release` above is
  why nobody saw it). A fixed "budget + 100ms" slack encodes a speed. Calibrate instead: time
  the unit of work the deadline is only checked between - 1024 nodes for the alpha-beta, one
  simulation for the MCTS - and allow a small multiple of that plus a constant for scheduling.
- The stride between clock reads is what bounds the overshoot, so keep it small enough that the
  work between reads is short. The MCTS read the clock every 128 simulations; one simulation is
  a descent plus a full random playout, so a read per simulation costs nothing measurable and
  cut the worst overshoot from half a second to one playout.
- The tournament layer must stay game-agnostic: it sees only `DynGame`, opaque JSON, `Side` and
  `Outcome`. Game-specific code belongs in `crates/games/src/<game>/`.
- Protocol changes touch `PROTOCOL.md`, `schema/` and `crates/proto` together. The markdown
  examples and the schemas are tested against the Rust types, so they cannot drift silently.
- Config paths (`out`, openings files, and the working directory of `cmd`) are relative to the
  config file. A spawned `cmd` is run as `sh -c "exec <cmd>"`, so it must be a single command:
  use `env VAR=value prog`, never `cd x && prog`.
- The JAX client imports pgx's `config` and `network` modules from `examples/alphazero`. Anything
  that unpickles a checkpoint needs that directory on `sys.path` first (`_use_alphazero_modules`).
  Tests must not set it up on the agent's behalf; that hid a real bug once.
- Jit anything JAX that runs per request. Un-jitted `ENV.step`, `observe` or `legal_action_mask`
  cost seconds per move and made think times meaningless.

## Measuring engines

- `tdg-alphabeta` has two searches: `search.rs` over the LEONIDAS evaluation, and `solve.rs` over
  the rules alone (win, loss or don't know). Nothing game-specific may enter `solve.rs`, its move
  ordering included — that is the whole point of it. It is the floor the evaluation is measured
  against, so keep it honest.
- Read `clients/alphabeta/TUNING.md` before touching the alpha-beta. It records what has been
  measured, the open question (the faster search loses at 1 s; the odd/even-depth hypothesis is
  untested) and the current recommendation. When a result changes the recommendation, update that
  section, not just the log.
- Use the in-process tools in `clients/alphabeta/examples/` (`bench`, `reach`, `versus`, `spsa`,
  `solve`) for engine work; HTTP tournaments are for final validation.
- **Nothing below ~1 s a move can show a search improvement:** at 100 ms every variant stops at
  depth 2. Short matches measure noise.
- Use node or depth budgets, not milliseconds, whenever the CPU is shared (training, another
  match). Timed runs must not share the machine with other heavy jobs.
- Fixed-depth matches between engines that differ only in ordering should come out identical or
  tied. That's the check that a search change is only a speed change.
- Report sample sizes. With ~20-50 counted minimatches the standard error is 7-10 points; say so
  rather than calling a result. Epaminondas has no draws, so every minimatch is +1, -1 or excluded.
- **`--threads` on `tdg-alphabeta` and `tdg-mcts` is the client's `max_concurrency`, not search
  parallelism** (`clients/alphabeta/src/main.rs:83`). Each search is single-threaded, so
  `concurrency = 8` with `--threads 8` uses 8 of the 24 cores, not 64. This was once misread as
  oversubscription and "fixed" by dropping `concurrency` to 3 - which changed nothing for the
  engine and cut the *model's* think time from ~5000 ms to 2892 ms, because the JAX client
  batches whatever requests are in flight. Concurrency is a budget knob for the model client, so
  keep it fixed across runs you mean to compare.
- **A gauntlet result without both clocks attached is meaningless.** pgx's E17 scored 84.6%
  (+296) against the alpha-beta at 5.0 s a move against its 1.0 s. At a comparable budget, E18 -
  176 Elo stronger than E17 - scored 46.2%, a 7-6 dead heat. The 296 was the time, not the model.
- Think time is measured from when a client actually receives the request, not from when the
  game started waiting for a free slot on a busy client.
- Checkpoints record `env_version`; the JAX client refuses a mismatch with v6 unless told to.

## Running long jobs

- Start anything over ~10 minutes detached (`nohup setsid ... &`), log to the scratchpad, and
  wait with a background `until grep -q ... log; do sleep 10; done` rather than chained sleeps.
- **Never `pkill -f` / `pgrep -f` with a pattern that appears in the same command line** (a
  heredoc included). It matches the calling shell and kills it (exit 144). This happened three
  times. Write a kill script in one tool call and run it in the next, or kill by PID.
- Don't rebuild an example while a long run is executing that binary. Stop it first; `spsa`
  resumes from its output file and log.
- Python patch scripts that match exact source text break after `cargo fmt` rewraps it. Re-read
  the file after formatting, or use the Edit tool.

## Git

- `main` on `git@github.com:calr3/tdgauntlet.git`. Commit and push only when the user asks.
- `target/` and `results/` are ignored. Match results and the frozen baseline binary
  (`target/baseline/tdg-alphabeta-v0`) are local only; anything worth keeping goes in `TUNING.md`
  or `clients/alphabeta/weights/`.
- End commit messages with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

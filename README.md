# tdgauntlet

A tournament server for board-game agents, and a protocol small enough that an agent can be
written in any language that can answer an HTTP request.

The **server** owns the rules, the clock, adjudication and the score. A **client** answers one
question — "given this position, what is your move?" — and holds no per-game state, so it can be
asked about many games at once, in any order. That is what lets a client keep something expensive
in memory (a compiled JAX model, a transposition table) without the tournament having to care.

Positions and moves are opaque JSON tagged with a game id, so the tournament layer knows nothing
about any particular game. [Epaminondas](games/epaminondas.md) is the first one.

- [`PROTOCOL.md`](PROTOCOL.md) — the wire contract; start here to write a client.
- [`schema/`](schema/) — JSON Schema for every message.
- [`games/`](games/) — the state and move encoding for each game.

## Quick start

```bash
cargo build --release

# Two random clients, started by the server itself.
./target/release/tdgauntlet run examples/random_v_random.toml

# Alpha-beta against random.
./target/release/tdgauntlet run examples/alphabeta_v_random.toml

# Check a client you have written before entering it.
./target/release/tdgauntlet check http://127.0.0.1:9001
```

Paths in a config — `out`, an openings file, and the working directory of any `cmd` — are relative
to the config file, so a config means the same thing wherever it is run from.

## Layout

| path | what |
|---|---|
| `crates/proto` | wire types |
| `crates/games` | the `Game` trait, the registry, and the rules of each game |
| `crates/tourney` | the `tdgauntlet` binary: openings, scheduling, adjudication, scoring, reporting |
| `crates/client-kit` | serve a Rust engine as a client |
| `clients/random` | uniformly random legal moves; the reference client |
| `clients/keyboard` | a human at the terminal |
| `clients/alphabeta` | alpha-beta search over King and Peterson's LEONIDAS heuristics |
| `clients/jax` | a pgx AlphaZero checkpoint under Gumbel MCTS; also pgx's own `negamax.py` |
| `clients/pysdk` | write a client in Python with the standard library alone |
| `conformance/` | position vectors from pgx, and the scripts that dump them |
| `examples/` | tournament configs |

## Scoring

Players meet in **minimatches**: one opening, played twice, with colours swapped. From one
player's point of view:

| result | score | counted? |
|---|---|---|
| 2 wins | +1 | yes |
| 1 win 1 draw | +0.5 | yes |
| 2 draws | 0 | yes |
| 1 win 1 loss | — | **no** |
| 1 loss 1 draw | −0.5 | yes |
| 2 losses | −1 | yes |

A minimatch both players win from says nothing about either of them — the opening decided it — so
it is dropped from the denominator rather than recorded as a draw. The headline figure is
`100 × mean((score + 1) / 2)` over the minimatches that count, so 50% is par and the two players
of a pairing always sum to 100%.

Epaminondas has no draws, so every minimatch is +1, −1, or excluded, and the percentage is the
share of decisive minimatches won. The exclusion rate is reported alongside it: a high one means
the openings, not the players, are deciding games.

## Tournament config

```toml
game = "epaminondas/1"
seed = 77659          # openings, and the server's own moves under on_error = "random"
concurrency = 16      # games in flight at once
out = "../results/run.json"
max_plies = 12000     # safety net only: a game reaching it is a flagged draw

[openings]
kind = "random"       # "start" | "random" | "file"
plies = 2             # random moves from the start; 0 is the start position itself
count = 100
# path = "book.jsonl" # kind = "file": one {"state": ...} or {"moves": [...]} per line

[[player]]
name = "alphabeta"
cmd = "../target/release/tdg-alphabeta --port {port}"   # or url = "http://..."
budget_ms = 1000      # also budget_nodes, budget_sims: sent to the client, which honours what it can
on_error = "forfeit"  # "forfeit" | "random" | "retry:N"
# timeout_ms, max_concurrency, startup_timeout_s

# pairs = [["a", "b"]]   # top-level, before any table; default is a full round robin
```

The result file records every opening, every move with its think time and whatever the client
reported (`eval`, `depth`, `pv`, ...), every incident (illegal moves, timeouts, error statuses)
and what the server did about it, and the standings.

## The rules, and pgx

`epaminondas/1` is pgx's Epaminondas at env version **v6** (100-move quiet clock, advancement
tiebreak, no draws). The Rust engine in `crates/games` is an independent implementation, held to
pgx by `cargo test -p tdg-games --test conformance`: several thousand positions, every legal move
from each, every successor state and outcome, and perft counts, all dumped from pgx.

When pgx's rules change, regenerate the vectors in the pgx environment and commit them:

```bash
JAX_PLATFORMS=cpu python conformance/dump_pgx.py      # positions and perft (slow: perft(3))
JAX_PLATFORMS=cpu python conformance/dump_eval.py     # evaluator reference for clients/alphabeta
```

## Development

```bash
cargo test --workspace                                  # everything Rust, including conformance
python -m pytest                                        # the Python SDK (JAX tests skip without pgx)
JAX_PLATFORMS=cpu python -m pytest -p no:xdist clients/jax   # in the pgx environment
cargo run --release -p tdg-games --example perft -- 4
```

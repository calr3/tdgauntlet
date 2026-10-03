# Tuning the Gess negamax

A native port of the Gess web app's negamax (`~/calr3gh/gess/wasm-engine/src/lib.rs`), with its
evaluation's weights made tunable, so it can serve as a stronger yardstick for pgx's Gess networks.
The app's own engine is untouched; its weights are "v0" here. Like `clients/alphabeta/TUNING.md`:
when a result changes the recommendation, update the recommendation too.

## Current recommendation

- **Use `weights/v1.json`** - the app's five evaluation terms with weights refitted to game
  results: `stone 100, ring 195, ring_pair 2, deep 2, pressure 94` (v0: 100, 700, -210, 1, 10).
  It beats v0 by **0.67 per game at depth 3 and 0.82 at depth 4** (400 games each).
- Keep v0 (`--weights` omitted, or the app's `.wasm` through `clients/gess_negamax`) for
  comparisons with results measured against it (pgx E22-E25).
- Tune and compare at **fixed depth or nodes**, not milliseconds, whenever the CPU is shared.

## The port

The search, move generation, transposition table and evaluation are the app's line for line;
the browser parts (host imports, shared-memory buffers, Lazy-SMP helpers) are gone and the
statics live in a `Searcher`, so many engines can play in one process. Checked:

- `tests/identity.rs`: on 150 positions from strong games (`tests/data/identity.json.gz`, made by
  `tests/make_identity_fixture.mjs` from the app's own `.wasm`), the same root moves, outcomes and
  keys, the same per-move scores at depths 1-3 on the exact TT-off path, and the same best score
  at depths 1-4 on the production TT-on path.
- `tests/rules.rs`: move lists and outcomes equal `tdg_games::gess` on >1M moves from 40 random
  games, every outcome kind (win, loss, play on, a level quiet-rule draw) included.
- `examples/bench.rs` vs `examples/bench_wasm.mjs` (the app's `bench_fixed`): identical node
  counts on all 150 positions at depth 4 (6,223,827), **native 1.71x faster** (256 vs 149
  knodes/s, both while pgx trained on the GPU).

## Fitting the weights (3 October 2026)

`examples/dump.rs` wrote 493,501 positions with their terms and game results (side to move's
view): 388k from tdgauntlet match records (pgx models against each other and against negamax v0)
and 106k from 3,000 v0 self-play games at depth 3 from random 2-ply openings. `fit_eval.py`
fits P(win) = sigmoid(c + sum b_i f_i) (draws as half), holding out 20% of games; weights are
the slopes with stone at 100.

| weights | stone | ring | ring_pair | deep | pressure | held-out log-loss |
|---|---|---|---|---|---|---|
| v0 (best scale) | 100 | 700 | -210 | 1 | 10 | 0.5489 |
| fit, all positions (**v1**) | 100 | 195 | 2 | 2 | 94 | **0.5194** |
| fit, self-play only | 100 | 262 | -14 | 17 | 164 | (0.6058 on self-play) |
| fit, model games only | 100 | 283 | -51 | 3 | 122 | (0.4682 on model games) |

All three agree: a second ring is worth 2-3 stones, not 7; the ring-pair discount is ~0; ring
pressure (own stones round the opponent's rings minus theirs round one's own) is worth ~10x v0's.

## Matches (`examples/versus.rs`, 200 random 2-ply openings x both colours)

| a | b | depth | per game for a (+/- 95%) |
|---|---|---|---|
| fit, all (v1) | v0 | 3 | 0.667 +/- 0.046 (+121 Elo) |
| fit, models | v0 | 3 | 0.685 +/- 0.046 |
| **fit, all (v1)** | **v0** | **4** | **0.815 +/- 0.038 (+258 Elo)** |
| fit, models | v0 | 4 | 0.779 +/- 0.040 |
| fit, all | fit, models | 4 | 0.588 +/- 0.048 |

## Candidate terms (not adopted)

Five more terms (`threat`, `defenders`, `mobile`, `edge`, `quiet_stone`; see `engine::features`),
off by default and not computed then. Each improved the held-out fit (all five: 0.5005; the stone
lead scaled by the quiet clock alone: 0.5063), but none won in play against v1:

| a | b | depth | per game for a |
|---|---|---|---|
| v1 terms + quiet_stone | v1 | 3 | 0.432 +/- 0.049 |
| all ten terms | v1 | 3 | 0.485 +/- 0.049 |
| v1 terms + threat, defenders, mobile, edge | v1 | 3 | 0.525 +/- 0.049 |
| v1 terms + threat, defenders, mobile, edge | v1 | 4 | 0.425 +/- 0.048 |
| all ten terms | v1 | 4 | 0.440 +/- 0.049 |

`quiet_stone` most likely misleads the search near the 20-move rule, which the search already
scores exactly when it reaches it. A better fit to game results is not a better evaluation for
search; the match decides.

## SPSA (`examples/spsa.rs`)

150 iterations from v1 at depth 3, 32 games each (`weights/spsa_d3.json`, log alongside): drifted
to ring 244, ring_pair 17, deep 2, pressure 89, and **lost to v1 at depth 4, 0.434 +/- 0.048**.
Not adopted; tuning at depth 3 does not carry to depth 4.

## To do

- Timed confirmation with the CPU otherwise idle (after pgx E25): v1 vs v0 natively at 2 s, and
  native v0 vs the app's `.wasm` v0 at 2 s (the 1.7x speed).
- Measure pgx's checkpoints against v1 as well as v0.

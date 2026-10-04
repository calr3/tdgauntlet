# Tuning the Gess negamax

A native port of the Gess web app's negamax (`~/calr3gh/gess/wasm-engine/src/lib.rs`), with its
evaluation's weights made tunable, so it can serve as a stronger yardstick for pgx's Gess networks.
The app's own engine is untouched; its weights are "v0" here. Like `clients/alphabeta/TUNING.md`:
when a result changes the recommendation, update the recommendation too.

## Current recommendation

- **To measure pgx's networks, use v0** (the app's weights: `--weights` omitted, or the app's
  `.wasm` through `clients/gess_negamax`). v1 beats v0 head to head but is the weaker opponent
  for the networks (below), so it is not a better yardstick for them.
- **For negamax against negamax, v1 is stronger:** `weights/v1.json`, the app's five terms
  refitted to game results (`stone 100, ring 195, ring_pair 2, deep 2, pressure 94`; v0: 100,
  700, -210, 1, 10), beats v0 0.67 per game at depth 3, 0.82 at depth 4 and **0.670 +/- 0.065
  at 2 s a move** (native both, 100 openings x both colours, idle machine).
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

## Second round (3-4 October 2026): nothing beats v1

New data: 1,500 games of v1 self-play at depth 4 plus the recorded model games (474k positions,
`dump --weights v1.json`), each position flagged `captured` if the move played from it captured.
All matches are against v1 at depth 4, 200 openings x both colours, unless stated.

| idea | held-out log-loss (five-term fit: 0.5009) | per game vs v1 |
|---|---|---|
| refit of the five terms on the new data | 0.5009 | 0.449 +/- 0.048 |
| refit on quiet positions only (`--quiet-only`; their own baseline 0.4813) | 0.4813 | 0.468 +/- 0.048 |
| `spare_ring` (a ring one stone short) | 0.5009 (no gain) | not played |
| `material_ratio` (stone lead / stones on the board), `--anchor` | 0.4790 | 0.430 +/- 0.048 |
| the same on quiet positions | 0.4487 | 0.468 +/- 0.049 |
| 288 piece patterns (`--patterns --ridge 1e-4`) | 0.4822 | 0.485 +/- 0.049 |
| the same on quiet positions | 0.4606 | 0.434 +/- 0.048 |

The piece patterns (`engine::pattern_counts`): every legal piece of each side, keyed by its 3x3
own-stone pattern read from its own side (forward up; white's flipped), merged with its left-right
mirror only - the start is not symmetric under the other reflections - giving 288 classes, one
weight each, off unless a weights file gives a `patterns` list. They also cost 1.6x the time per
move at depth 4, so they would do worse still at equal time.

Quiescence search (`Switches::quiescence`, `versus ... v1.json@q<n>`: at the last ply, follow
captures for up to n more plies, with stand-pat), at **equal node budgets** against plain v1:

| plies | 50k nodes, 100 openings | 200k nodes, 100 openings | 100k nodes, 300 openings |
|---|---|---|---|
| 2 | 0.427 +/- 0.068 | 0.425 +/- 0.069 | |
| 4 | 0.555 +/- 0.069 | 0.550 +/- 0.069 | 0.482 +/- 0.040 |
| 6 | | | 0.468 +/- 0.040 |
| 8 | | | 0.377 +/- 0.039 |

The early 4-ply lead was noise. A capture in Gess is rarely forced and replies are many, so the
capture tree is wide, and the nodes it costs are taken from the main search.

Every evaluation idea fitted better than v1 and none played better. With five terms the
evaluation is not what limits this engine at depth 4; v1 stays the recommendation.

## Timed results (4 October 2026, after pgx E25; machine otherwise idle)

- Native v0 against the app's `.wasm` v0, 2 s a move, 100 openings x both colours
  (`examples/gess_negamax_native_v_wasm.toml`): **55.6%** over 54 counted minimatches (+/- ~7).
  The port's 1.7x speed is worth little at this time control: both search to depth ~5.
- pgx models at 128 simulations against v0 and v1 at 2 s (`examples/gess_e25_negamax.toml`):
  E25 it 183 / 120 / 60 and E24 scored 79.8 / 85.7 / 78.6 / 66.7% against v0 and 93.8 / 93.9 /
  91.2 / 90.0% against v1. v1 and v0 reached the same depths (mean 5.21 and 5.17), so the gap is
  the weights: v1's low ring weight (195 against 700) and high ring pressure presumably suit
  negamax's own play and not the networks'. Strength here is not transitive.

## To do

- If the yardstick is to get stronger against the networks, tune against them: fit to the
  model games only (`--source models`) or play SPSA against a network, rather than against v0.

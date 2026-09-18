# Tuning the alpha-beta client

A log of what has been tried on `tdg-alphabeta`, what it measured, and what it means. Like pgx's
`*_EXPERIMENTS.md`: when a result changes what you would recommend, update the recommendation here
too, not just the log.

## Current recommendation

- Run with the **plain search** (`--search plain`, the default) and the **recommended weights**
  (`--weights recommended`, the default): pgx's LEONIDAS weights plus `advancement` = 1. That one
  term scores 96-98% against pgx's weights (see "An advancement term", 19 September). The search
  refinements below have so far measured neutral or worse at real time controls, for reasons not
  yet pinned down.
- Use `--weights pgx` wherever the evaluation must match pgx's exactly, as in the calibration
  match against `negamax.py`.
- Measure strength at **~1 s per move or more**. At 100 ms every variant stops at depth 2, so
  changes to the search cannot show up at all (see "How deep each variant gets").

## Tools

All in `examples/`, all in process (no HTTP), all scored as tournaments are (minimatches, 1 win
1 loss excluded):

| example | what |
|---|---|
| `bench <depth>` | fixed-depth node counts over 27 positions, one row per search feature |
| `reach <ms>` | the depth each feature set reaches at a time control |
| `versus <budget> <openings> <threads> [ablate]` | engine against engine; budget `250` ms, `n40000` nodes, or `d3` depth |
| `versus <budget> <openings> <threads> weights <a.json>...` | each weights file against pgx's weights, plain search |
| `spsa <iterations> <openings> <threads> <out.json>` | SPSA self-play tuning of the weights; resumes from `<out.json>` |

Use nodes or depth, not milliseconds, whenever the machine is shared (a training run, another
match): a node budget gives the same games however loaded the CPU is.

## Findings, 18 September 2026

### The baseline was ordering captures by nothing

The original ordering sorted captures by `|lead - rear|` in *square indices*: 14 for a vertical
pair, 1 for a horizontal one. It now sorts by pieces captured (`Position::captured_count`).

### Quiescence is almost the whole search

At depth 3, 98% of nodes are quiescence nodes; at depth 4, 85-90%. Captures chain into long
exchanges in this game. Anything that makes quiescence cheaper or smaller matters far more than
main-search ordering.

### Search features at fixed depth (bench, depth 4, 27 positions)

| features | nodes | time | first-move cutoffs |
|---|---|---|---|
| none | 11.7M | 35.1 s | 92.7% |
| + capture order | 11.4M | 33.9 s | 92.9% |
| + history | 10.0M | 29.8 s | 94.1% |
| + PVS | 8.0M | 24.7 s | 94.6% |
| + aspiration | 7.2M | 22.4 s | 96.0% |
| + incursion-aware quiescence | 11.5M | 23.8 s | 95.9% |

Ordering is good from the start (93% of cutoffs on the first move); PVS and aspiration are where
the time goes. Incursion-aware quiescence searches more nodes but cheaper ones.

### How deep each variant gets

| features | mean depth at 100 ms | mean depth at 1 s | depth 4 reached at 1 s |
|---|---|---|---|
| none | 2.11 | 3.37 | 10/27 |
| + capture order, history | 2.11 | 3.56 | 15/27 |
| + PVS | 2.15 | 3.78 | 21/27 |
| + aspiration | 2.15 | 3.81 | 22/27 |
| all | 2.15 | 3.85 | 23/27 |

The branching factor is ~150-280, so a ply costs two orders of magnitude. At 100 ms nothing
reaches depth 3: matches there cannot measure search improvements.

### ...but the faster search does not win

| match | budget | score | counted | games |
|---|---|---|---|---|
| all features v none | 100 ms | 39.1% | 23 | 35-45 |
| ordering v none | 100 ms | 42.9% | 49 | 73-87 |
| incursions v ordering | 100 ms | 50.0% | 36 | 80-80 |
| delta pruning v ordering | 100 ms | **29.3%** | 41 | 63-97 |
| all features v none | 1 s | **33.3%** | 18 | 34-46 |
| ordering v none | depth 2 | even: every game identical | 0 | 60-60 |
| ordering v none | depth 3 | 50.0% | 4 | 60-60 |

At a fixed depth the refinements change nothing, as they should: identical games at depth 2,
an exact tie at depth 3. So they are not bugs in the search's result. Yet at 1 s, where they let
the engine reach depth 4 on most moves instead of 3, it scores 33%.

**Open question, the next thing to test:** is this engine simply worse at even depths than odd
ones with this evaluation? At depth 4 the opponent has the last move before quiescence; at depth
3 we do. Test: plain search at fixed depth 4 against plain search at fixed depth 3
(`versus d4` needs a variant with two depths). If depth 4 loses to depth 3, the fix belongs in
the evaluation or quiescence (or search to odd depths only), not the move ordering.

### Delta pruning is harmful here, and off

About -150 Elo. It bounds what a capture can gain by its material, but under the LEONIDAS weights
a piece on a home rank is worth far more than its material (crossing 250, material 30), so it
prunes precisely the captures that remove an incursion. It stays in `Features` as a switch,
default off.

### New evaluation terms, and a partial SPSA run

Three terms were added for what decides v6 games — the advancement tiebreak at the 100-move quiet
clock, which pgx's LEONIDAS weights do not measure at all: `advancement` (rows advanced),
`tiebreak` (a smooth stand-in for the rank-by-rank rule) and `tiebreak_clock` (the same, scaled by
how far the clock has run). Their default weight is zero, keeping `Weights::default()` identical
to pgx's evaluator (checked by `tests/eval.rs`).

SPSA ran 39 iterations (16 openings, 32 games each, 40k nodes a move, all search features on)
before being stopped to investigate the result above. Every iteration's perturbed pair differed
by 10-26 net games out of 32: the weights matter a great deal. Where it had got to is in
`weights/spsa_partial_iter38.json`:

| weight | pgx | after 39 iterations |
|---|---|---|
| material | 30 | 30 (fixed) |
| crossing | 250 | 240 |
| home_row_defense | 8 | 2.3 |
| territory | 0.5 | 1.0 |
| centrality | 4 | 4.5 |
| mobility | 2 | 2.5 |
| advancement | 0 | -0.5 |
| tiebreak | 0 | 4.8 |
| tiebreak_clock | 0 | 27 |

**Not validated**: no match has been played with these weights against the defaults, and they
were tuned with the search features that are under suspicion. The clearest signals — home-row
defence falling to a quarter, and the clock-scaled tiebreak rising steadily — are worth checking
first.

## Findings, 19 September 2026

### An advancement term: pgx's weights plus `advancement` = 1

Self-play at 1 s a move showed how the engine loses: after the middle-game exchanges both sides
retreat to their home ranks and shuffle until the 100-move quiet clock, and a single stray
piece decides the advancement tiebreak. Nothing in pgx's weights pulls a piece forward.

Each weights file below against pgx's weights, plain search, 40k nodes a move, 60 openings:

| weights | score | counted | games |
|---|---|---|---|
| advancement 0.5 | 82.9% | 41 | 87-33 |
| advancement 1 | **97.9%** | 47 | 105-15 |
| advancement 1.5 | 97.7% | 44 | 102-18 |
| advancement 2 | 93.0% | 43 | 97-23 |
| advancement 3 | **25.8%** | 31 | 45-75 |
| tiebreak_clock 27 | 73.7% | 38 | 78-42 |
| tiebreak 5, tiebreak_clock 27 | 68.3% | 41 | 75-45 |
| advancement 1, tiebreak_clock 27 | 97.9% | 47 | 105-15 |

At 150k nodes a move, 30 openings: advancement 1 scores 96.0% (25 counted, 53-7) and 1.5 scores
100% (18 counted, 48-12). The gain survives a deeper search.

The peak is broad (1-1.5) but there is a cliff between 2 and 3; 1 was chosen for its distance
from it. The clock-scaled tiebreak helps on its own but adds nothing to advancement. The largest
signals from the partial SPSA run below (home-row defence falling, the clock tiebreak rising)
were probably the same pressure found indirectly; that run started from zero advancement.

Not yet measured: at real time controls, and with the search features on (the open question
below was measured under pgx's weights and may look different now).

## Next steps

1. Confirm the recommended weights at 1 s a move in an HTTP tournament against `--weights pgx`,
   on an otherwise idle machine.
2. Depth 3 against depth 4, plain search, fixed depth: settle the odd/even question, now under
   the recommended weights.
3. Restart SPSA from the recommended weights rather than resuming the partial run, which began
   from zero advancement; `weights/spsa_partial_iter38.json` is superseded as a starting point.

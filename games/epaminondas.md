# `epaminondas/1`

Epaminondas (Robert Abbott, 1975) on a 14 wide by 12 high board. The rules are those implemented
by `pgx/_src/games/epaminondas.py` at pgx env version **v6**; its header comment is the
specification.

## Squares

Row-major, `idx = row * 14 + col`, `0 <= idx < 168`. Row 0 is **white's home rank**; row 11 is
black's. In human notation columns are `a`..`n` left to right and rows are `1`..`12`, so index 0
is `a1` and index 167 is `n12`.

## State

```json
{
  "board": "WWWWWWWWWWWWWWWWWWWWWWWWWWWW............ (168 chars) ...BBBBBBBBBBBBBB",
  "to_move": 0,
  "moves": 13,
  "quiet_moves": 5,
  "last_capturer": -1
}
```

| field | meaning |
|---|---|
| `board` | exactly 168 characters of `.` (empty), `W` (white) or `B` (black), row-major from `a1` |
| `to_move` | `0` white, `1` black |
| `moves` | completed moves, for reporting only |
| `quiet_moves` | moves since the last capture; the game is decided at 100 |
| `last_capturer` | `0`, `1`, or `-1` if nobody has captured — input to the mirror tiebreak |

## Move

```json
{"lead": 17, "rear": 3, "dest": 45, "san": "d1d2-d4"}
```

- `lead` — the moving piece that ends up furthest along the direction of travel.
- `rear` — the rear piece of the moving phalanx. **`rear == lead` means a single piece.**
  Together `lead` and `rear` fix the direction and the number of pieces moving.
- `dest` — where the lead piece lands.
- `san` — optional, ignored on input, filled in by the server for logs. A single piece is
  written `c2-c3`; a phalanx names its rear, then its lead, then the lead's destination:
  `d1d2-d4`. A capture uses `x` in place of `-`.

This is exactly pgx's three-action encoding (`step(lead); step(rear); step(dest)`), so a pgx-based
client converts in both directions with no work.

## Rules summary

- A **phalanx** is two or more friendly pieces adjacent in a straight line, orthogonal or diagonal.
- A single piece moves one square in any of the eight directions, onto an empty square.
- A phalanx moves along its own line, every piece the same number of squares, at most as many
  squares as there are pieces moving. Any contiguous sub-run that includes the lead piece may move
  on its own (splitting). Nothing moves onto or over a friendly piece, or over an enemy piece.
- **Capture**: the lead may land on an enemy piece if the enemy line starting there and continuing
  away in the direction of movement is *strictly shorter* than the moving phalanx. That whole
  enemy line is removed and the movement stops there.
- **Win**: at the start of your turn, if you have more pieces on your opponent's back rank than
  they have on yours, you win — so an incursion gives the opponent one turn to answer it.
- A piece may not move onto the opponent's back rank if that would make the whole position
  left-to-right symmetric.
- A side with no legal move loses.
- After **100 consecutive moves without a capture** the game is decided on **advancement**: piece
  counts compared rank by rank starting from the opponent's home rank, first difference wins. An
  exact mirror falls back to the last capturer, and failing that to black. **There are no draws.**
- Every capture resets that count, so there is no absolute bound on game length. Each side has
  28 pieces, so at most 55 captures can happen: the worst case is about 5,600 moves.

# `gess/1`

Gess (Cambridge University Games Society, 1993) on a 20 x 20 grid. The rules are those
implemented by `pgx/_src/games/gess.py` at pgx env version **v1**; its header comment is the
specification.

## Cells

Row-major, `idx = row * 20 + col`, `0 <= idx < 400`. Row 0 is the **top** of the board, on white's
side; row 19 is the bottom, on black's. In human notation columns are `a`..`t` left to right and
ranks count up from the bottom, `rank = 20 - row`, so index 0 is `a20` and index 399 is `t1`.

The outer ring — row or column 0 or 19 — is a **border**: it never holds a stone, and a stone
pushed onto it is removed from the game. The playing area is the inner 18 x 18, `b2`..`s19`. A
piece may still be *centred* on the border, with part of its footprint hanging over the edge.

**Black moves first**, so black is side `0`.

## State

```json
{
  "board": "....................  (400 chars) ....................",
  "to_move": 0,
  "moves": 13,
  "quiet_moves": 5
}
```

| field | meaning |
|---|---|
| `board` | exactly 400 characters of `.` (empty), `B` (black) or `W` (white), row-major from `a20` |
| `to_move` | `0` black, `1` white |
| `moves` | completed moves, for reporting only |
| `quiet_moves` | moves since a stone was last taken off the board; the game is decided at 20 |

pgx's own state has no move counter; `moves` is tdgauntlet's, and nothing in the rules reads it.

## Move

```json
{"from": 126, "to": 166, "san": "g14-g12"}
```

- `from` — the centre of the 3 x 3 piece being picked up.
- `to` — the centre it lands on.
- `san` — optional, ignored on input, filled in by the server for logs. `g14-g12`, or `g14xg12`
  when the move takes at least one stone off the board.

This is exactly pgx's two-action encoding (`step(from); step(to)`), so a pgx-based client converts
in both directions with no work.

## Rules summary

- A **piece** is any 3 x 3 footprint that holds at least one of your stones among the eight cells
  around its centre and **none** of your opponent's anywhere in it. Its centre can be any cell of
  the grid, the border included. A piece is not a permanent object: every move chooses one afresh.
- A piece moves in a direction only if it has a stone on that side of its centre — a stone in the
  north cell opens north, the north-east cell opens north-east, and so on. With a stone in the
  **centre** as well it may travel any distance; without one, at most three cells.
- Every stone of the footprint moves together, the same number of cells in the same direction.
- The piece stops on the first cell where its footprint would meet a stone — its own or the
  opponent's. It may land there, which is how a **capture** is made: everything standing in the
  footprint it lands on is removed, including its own stones. It may not travel past that cell.
- Stones carried onto the border ring are removed. A piece that runs off the edge loses those
  stones for good.
- A **ring** is an empty cell of the playing area with eight of your own stones around it. The
  moment you have none you lose. A move that destroys your own last ring loses even if the same
  move destroys your opponent's — the mover is checked first.
- After **20 consecutive moves in which no stone is removed** — captures, self-captures, and
  stones pushed over the border all count — the game is decided on stones left on the playing
  area. An exact tie is the only **draw** in the game.
- A side that still has a ring always has a move: the ring's centre is a piece with all eight
  directions open. So there is no stalemate rule, and a position has no legal moves only when the
  game is already over.

# `one-more-tri/1`

One More Tri: two players take turns placing numbered tiles on a board of 21 triangles. When
every tile is down, one triangle is left empty, and the tiles around it decide the game. The rules
are those implemented by `pgx/_src/games/g_hex.py`, pgx's `g_hex` env at version **v0**.

## Triangles

Numbered `0`..`20`, as in pgx's diagram, which is also what the server renders. An empty triangle
shows its own number there; a tile shows as `B7` or `W3`, with `T` for a 10.

```
      ____________
     /\    /\    /\
    / 0\ 1/ 2\ 3/ 4\
   /____\/____\/____\
  /\    /\    /\    /\
 / 5\ 6/ 7\ 8/ 9\10/11\
/____\/____\/____\/____\
\    /\    /\
 \12/13\14/15\
  \/____\/____\______
   \    /\    /\    /
    \16/17\18/19\20/
     \/____\/____\/
```

A triangle's **neighbours** are the ones it shares an edge with:

| triangle | neighbours | triangle | neighbours | triangle | neighbours |
|---|---|---|---|---|---|
| 0 | 1, 6 | 7 | 6, 8, 14 | 14 | 7, 13, 15 |
| 1 | 0, 2 | 8 | 2, 7, 9 | 15 | 14, 18 |
| 2 | 1, 3, 8 | 9 | 8, 10 | 16 | 13, 17 |
| 3 | 2, 4 | 10 | 9, 11 | 17 | 16, 18 |
| 4 | 3, 10 | 11 | 10 | 18 | 15, 17, 19 |
| 5 | 6, 12 | 12 | 5, 13 | 19 | 18, 20 |
| 6 | 0, 5, 7 | 13 | 12, 14, 16 | 20 | 19 |

**One pair is one-sided, as in pgx:** 4 lists 10, but 10 does not list 4, although the two
triangles share an edge. So with 4 left empty, the tile on 10 counts; with 10 left empty, the
tile on 4 does not. This is pgx's `_ADJ` as written, and the rule its checkpoints were trained
under. Correcting it would be a new game version.

**Black moves first**, so black is side `0`. pgx stores black's tiles as positive values and
white's as negative ones, and so does the wire.

## State

```json
{
  "board": [-10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 1],
  "to_move": 1
}
```

| field | meaning |
|---|---|
| `board` | exactly 21 integers, one per triangle in the order above: a tile value, positive for black and negative for white, or `0` for empty |
| `to_move` | `0` black, `1` white |

Each side's tiles `1`..`10` may each appear at most once. Which tiles a side still holds, and how
many moves have been played, both follow from the board. `to_move` does too: black moves first
and nobody passes. A state where it disagrees with the tile counts is rejected.

## Move

```json
{"tri": 13, "tile": 7, "san": "7 on 13"}
```

- `tri`: the empty triangle to play on, `0`..`20`.
- `tile`: the value of a tile the mover still holds, `1`..`10`.
- `san`: optional, ignored on input, filled in by the server for logs. It is the tile, then the
  triangle, `7 on 13`, the form pgx's interactive tournament reads.

pgx takes this as a single action, `tri * 10 + tile - 1`.

## Rules summary

- Each side has ten tiles, valued 1 to 10. On your turn you put any tile you still hold on any
  empty triangle. There is no passing, and a tile once placed stays.
- The game ends when both hands are empty: 20 moves, 10 each, leaving exactly one of the 21
  triangles empty.
- Add up the tiles on the empty triangle's neighbours, black's counting positive and white's
  negative. A positive total wins for black and a negative one for white. A total of exactly 0 is
  a **draw**.

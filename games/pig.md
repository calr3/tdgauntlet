# `pig/1`

Pig: on your turn you roll a die as often as you like, adding each roll to the turn's total,
then stop and bank it. Roll a 1 and the turn's total is lost. The first to bank 100 wins. The
rules are those implemented by `pgx/_src/games/pig.py`, pgx's `pig` env at version **v0**.

This is the one game here with chance in it, and **the server rolls the die**. A client only
ever decides whether to roll again, and learns the roll from the next position it is sent. As
in pgx, the roll is part of the move that leads to it:

- the game starts with the first player's first roll already made;
- `roll` rolls again for the same player;
- `stop` banks and rolls the other player's first roll.

So every position shows the roll its player is deciding on. A 1 has already cost them the turn:
the only move left is `stop`, which banks nothing. The client is still asked for it.

The server draws the dice from the tournament seed and the minimatch, so both games of a
minimatch see the same sequence of rolls. Two players that behave alike then get alike luck,
until their decisions put them out of step.

## State

```json
{"totals": [42, 17], "turn_total": 9, "last_roll": 5, "to_move": 1}
```

| field | meaning |
|---|---|
| `totals` | the points each player has banked, indexed by side |
| `turn_total` | the rolls of this turn so far, `last_roll` included. After a 1, what is about to be lost |
| `last_roll` | the roll `to_move` is deciding on, `1`..`6` |
| `to_move` | `0` or `1`. Side `0` starts |

The rolls before `last_roll` in a turn are each 2 or more, so `turn_total` is `last_roll`, or
`last_roll` plus at least 2. A state where it is not is rejected, as is one where the player to
move has already banked 100.

## Move

```json
{"action": "roll", "san": "roll"}
```

- `action`: `"roll"` to roll again, or `"stop"` to bank and pass the die. `roll` is illegal after
  a 1.
- `san`: optional, ignored on input, filled in by the server for logs. It is the action again:
  the words pgx's interactive tournament reads, `r`/`s` there.

pgx's actions are `0` for `stop` and `1` for `roll`.

## Rules summary

- A turn starts with a roll. After each roll that is not a 1, the player either rolls again,
  adding the roll to the turn total, or stops, adding the turn total to their banked total.
- A 1 ends the turn with nothing banked.
- Whoever banks 100 or more wins at once. A turn total past 100 wins nothing until it is banked,
  and a 1 still loses it.
- No game is drawn, and nothing limits a game's length but the server's `max_plies`.

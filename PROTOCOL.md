# tdgauntlet protocol, version 1

A **client** is any process that can answer "given this position, what is your move?" over HTTP.
The **server** owns the rules, the clock, adjudication and the score; it is the only authority.

The protocol is deliberately small. A conformant client is three endpoints and no state machine.

## The contract

### 1. A client holds no per-game state

Every request carries the complete position. The server may:

- interleave positions from many different games, in any order;
- ask about the same position twice;
- ask about a position that never occurs in any game it is running;
- never follow up on a position it asked about.

A client that caches **by position** — a transposition table, an evaluation cache, a compiled
model, an opening book — is conformant and encouraged. That is position state, not game state.

A client that assumes two consecutive requests belong to the same game, or that maintains a board
it updates as requests arrive, is **not** conformant and will produce nonsense under concurrency.

### 2. The server is authoritative

The server generates the legal moves itself and verifies everything a client returns. A client
does not need the server's agreement to think, but it cannot play a move the server rejects.

### 3. Positions and moves are opaque

The envelope is fixed; `state` and `move` are arbitrary JSON defined by the game, identified by
the `game` field (`"<name>/<version>"`). The tournament layer never interprets them. See
[`games/`](games/) for each game's state and move encoding: `epaminondas/1`, `gess/1` and `one-more-tri/1` so far.

## Endpoints

### `GET /identify`

```json
{
  "protocol": 1,
  "name": "jax-20260918",
  "version": "0.3.1",
  "games": ["epaminondas/1"],
  "max_concurrency": 8
}
```

| field | meaning |
|---|---|
| `protocol` | must be `1` |
| `name` | how the client appears in results; the server may override it from its config |
| `version` | free-form, recorded in the result file |
| `games` | game ids this client can play |
| `max_concurrency` | most simultaneous `/move` requests the server may have in flight |

### `GET /health`

`200` once the client is ready to answer moves (model loaded, tables built). Anything else, or a
connection failure, means not ready. The server polls this after spawning a client.

### `POST /move`

Request:

```json
{
  "protocol": 1,
  "request_id": "0f9c1f7a-6c1e-4f6a-9c6e-9b1d0f2a3c4d",
  "game": "epaminondas/1",
  "state": {"board": "...", "to_move": 0, "moves": 13, "quiet_moves": 5, "last_capturer": -1},
  "budget": {"ms": 1000, "nodes": null, "sims": null},
  "context": {"match_id": "m17", "game_id": "m17-a", "ply": 26, "side": 0}
}
```

Response:

```json
{
  "request_id": "0f9c1f7a-6c1e-4f6a-9c6e-9b1d0f2a3c4d",
  "move": {"lead": 17, "rear": 3, "dest": 45},
  "info": {"eval": 0.31, "depth": 9, "nodes": 184203, "pv": [], "elapsed_ms": 987}
}
```

- `request_id` must be echoed back unchanged.
- `budget` fields are all nullable; honour whichever you understand and ignore the rest. Reporting
  `info.elapsed_ms` lets the server show who is actually respecting the clock — the tournament
  does not enforce equal work, only equal declared budget.
- `context` is informational: useful for logging, never required to choose a move. `side` is which
  side the client is playing, which is always the side to move in `state`.
- `info` is entirely optional and free-form beyond the keys above.

### Errors

A client that cannot move returns a non-2xx status with:

```json
{"error": "unsupported game: chess/1"}
```

## What the server does when a client misbehaves

Per-player `on_error` policy in `tournament.toml`, applied to illegal moves, malformed responses,
non-2xx statuses and timeouts alike:

| policy | effect |
|---|---|
| `forfeit` | the client loses that game immediately (the default) |
| `random` | the server plays a uniformly random legal move and the game continues |
| `retry:N` | re-ask up to N times, then forfeit |

Every occurrence is recorded in the result file. A client that forfeits on move 3 of every game is
visible as an incident count, not as a silent 0%.

## Writing a client

The shortest conformant client is `clients/random` (Rust) and `clients/pysdk` (Python). In Python:

```python
from tdgauntlet_client import serve


def choose_move(game, state, budget, context):
    ...
    return {"lead": 17, "rear": 3, "dest": 45}


serve(name="my-agent", games=["epaminondas/1"], choose_move=choose_move, port=9001)
```

Check a running client against the contract above before entering it in a tournament:

```bash
tdgauntlet check http://127.0.0.1:9001
```

Besides the endpoints, this sends many unrelated positions under a single game id, shuffled and
concurrently. A client that remembers "the current game" answers some of them for the wrong board,
and fails.

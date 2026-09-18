"""Converting between the `epaminondas/1` wire encoding and pgx.

The wire state is exactly the stage-0 part of pgx's `GameState`: board, side to move, the two
move counters and the last capturer. `stage`, `lead` and `rear` are intra-move scratch and are
always zero between moves, and `winner` is -1 in any position the server asks about (it never
asks about a finished game). So the conversion is exact in both directions, and the observation
the network sees is bit-identical to the one it would have seen in self-play.

A wire move {lead, rear, dest} is pgx's three consecutive actions, one per field.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from pgx._src.games.epaminondas import Game, GameState, N
from pgx.epaminondas import Epaminondas, State

ENV = Epaminondas()
GAME = Game()
# Jitted once: called un-jitted, these cost far more than the rest of a request put together.
_legal_action_mask = jax.jit(GAME.legal_action_mask)
_observe = jax.jit(ENV.observe)
RULES_VERSION = "v6"  # the pgx env version `epaminondas/1` corresponds to
GAME_ID = "epaminondas/1"

_GLYPHS = ".WB"


def to_pgx(wire: dict[str, Any]) -> State:
    """A pgx state for a wire state, ready for `ENV.step` and the network."""
    board = wire["board"]
    if len(board) != N or any(c not in _GLYPHS for c in board):
        raise ValueError(f"board must be {N} characters of '.', 'W', 'B'")
    to_move = int(wire["to_move"])
    if to_move not in (0, 1):
        raise ValueError(f"to_move must be 0 or 1, not {to_move}")
    x = GameState(
        color=jnp.int32(to_move),
        board=jnp.asarray(np.array([_GLYPHS.index(c) for c in board], dtype=np.int8)),
        stage=jnp.int32(0),
        lead=jnp.int32(0),
        rear=jnp.int32(0),
        winner=jnp.int32(-1),
        moves=jnp.int32(int(wire["moves"])),
        quiet_moves=jnp.int32(int(wire["quiet_moves"])),
        last_capturer=jnp.int32(int(wire["last_capturer"])),
    )
    state = State(
        current_player=jnp.int32(to_move),
        _x=x,
        legal_action_mask=_legal_action_mask(x),
        # Three actions per move; nothing reads it, but keep it what self-play would have had.
        _step_count=jnp.int32(3 * int(wire["moves"])),
    )
    return state.replace(observation=_observe(state))  # type: ignore[attr-defined]


def from_pgx(state: State) -> dict[str, Any]:
    """The wire state for a pgx state between moves (stage 0)."""
    x = state._x
    if int(x.stage) != 0:
        raise ValueError("only a state between moves (stage 0) has a wire encoding")
    return {
        "board": "".join(_GLYPHS[int(c)] for c in np.asarray(x.board)),
        "to_move": int(x.color),
        "moves": int(x.moves),
        "quiet_moves": int(x.quiet_moves),
        "last_capturer": int(x.last_capturer),
    }


def move_to_actions(move: dict[str, Any]) -> list[int]:
    return [int(move["lead"]), int(move["rear"]), int(move["dest"])]


def actions_to_move(actions: list[int]) -> dict[str, int]:
    lead, rear, dest = (int(a) for a in actions)
    return {"lead": lead, "rear": rear, "dest": dest}

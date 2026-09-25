"""Converting between a game's wire encoding and pgx.

Each game this client can play has a [`Wire`]: the pgx env behind it, the pgx rules version its
wire id stands for, and the conversions both ways. A checkpoint records the env it was trained on
and that is what picks the wire a process serves — one checkpoint, one game.

In both games the wire state is exactly the between-moves part of pgx's `GameState`. `stage` is
zero between moves and `winner` is -1 in any position the server asks about, since it never asks
about a finished game. What the stages have chosen so far is scratch: Epaminondas clears it, and
Gess leaves its `source` at the last move's, which nothing then reads — at stage 0 it enters
neither the observation (the source footprint plane is zeroed unless `stage` is 1) nor the legal
mask, and the next action overwrites it. So the wire carries no `source`, and the observation the
network sees is bit-identical to the one it would have seen in self-play.

A wire move is pgx's consecutive actions, one per field: `{lead, rear, dest}` in Epaminondas,
`{from, to}` in Gess. pgx has no move counter in Gess, so `moves` there is the server's alone;
nothing in the rules reads it, and it only feeds `_step_count`.

One More Tri (pgx's `g_hex`) is the exception on both counts. Its board is a list of signed tile
values rather than a string, and its move `{tri, tile}` is a single pgx action, `tri * 10 + tile -
1`; it has no stages, so every state is between moves. pgx also draws which seat plays colour 0
at random; here the seat is always the colour, as it is in the other two games.

Pig is the one game of chance. Its state is a handful of counters, its move `{action}` is one
pgx action (0 `stop`, 1 `roll`), and every step rolls a die - drawn from the key `env.step` is
given. [`Wire.chance_keys`] is how the agent reaches every roll rather than a sample of one.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pgx
import pgx.epaminondas
import pgx.g_hex
import pgx.gess
import pgx.pig
from pgx._src.games import epaminondas as epaminondas_game
from pgx._src.games import g_hex as g_hex_game
from pgx._src.games import gess as gess_game
from pgx._src.games import pig as pig_game


class Wire:
    """One game's translation between the wire and pgx.

    Subclasses supply the state conversion, which is the only part that knows what a `GameState`
    holds; everything else here is shared.
    """

    game_id: str
    env_id: str
    #: Every pgx env version whose *rules* are this game id's, newest last. pgx bumps its env
    #: version for observation changes too, and those leave the rules alone, so one wire id
    #: stands for a family: `epaminondas/1` is pgx v6, and v7 to v11 only grew the planes the
    #: network sees. The first is the version the conformance vectors were dumped from, and the
    #: family is checked by regenerating them - v11 reproduces the v6 vectors byte for byte.
    rules_versions: tuple[str, ...]
    #: Board characters, indexed by the cell value pgx stores.
    glyphs: str
    #: The move's fields, in the order pgx takes them as actions.
    move_fields: tuple[str, ...]

    def __init__(self, env: Any, game: Any, state_cls: Any, cells: int) -> None:
        self.env = env
        self.game = game
        self.state_cls = state_cls
        self.cells = cells
        # Jitted once: called un-jitted, these cost far more than the rest of a request together.
        self._legal_action_mask = jax.jit(game.legal_action_mask)
        self._observe = jax.jit(env.observe)

    @property
    def actions_per_move(self) -> int:
        return len(self.move_fields)

    @property
    def chance_keys(self) -> jnp.ndarray:
        """Keys for `env.step`, one per equally likely outcome of the chance in a step. A game
        without chance ignores the key, so one does."""
        return jax.random.PRNGKey(0)[None]

    def to_pgx(self, wire: dict[str, Any]) -> Any:
        """A pgx state for a wire state, ready for `env.step` and the network."""
        raise NotImplementedError

    def from_pgx(self, state: Any) -> dict[str, Any]:
        """The wire state for a pgx state between moves (stage 0)."""
        raise NotImplementedError

    def move_to_actions(self, move: dict[str, Any]) -> list[int]:
        return [int(move[f]) for f in self.move_fields]

    def actions_to_move(self, actions: list[int]) -> dict[str, Any]:
        if len(actions) != self.actions_per_move:
            raise ValueError(f"{self.game_id} takes {self.actions_per_move} actions a move")
        return {f: int(a) for f, a in zip(self.move_fields, actions, strict=True)}

    # ─── shared pieces of the state conversion ───────────────────────────────

    def _board(self, wire: dict[str, Any]) -> jnp.ndarray:
        board = wire["board"]
        if len(board) != self.cells or any(c not in self.glyphs for c in board):
            glyphs = "', '".join(self.glyphs)
            raise ValueError(f"board must be {self.cells} characters of '{glyphs}'")
        return jnp.asarray(np.array([self.glyphs.index(c) for c in board], dtype=np.int8))

    def _board_text(self, board: Any) -> str:
        return "".join(self.glyphs[int(c)] for c in np.asarray(board))

    def _to_move(self, wire: dict[str, Any]) -> int:
        to_move = int(wire["to_move"])
        if to_move not in (0, 1):
            raise ValueError(f"to_move must be 0 or 1, not {to_move}")
        return to_move

    def _state(self, x: Any, to_move: int, moves: int) -> Any:
        state = self.state_cls(
            current_player=jnp.int32(to_move),
            _x=x,
            legal_action_mask=self._legal_action_mask(x),
            # Nothing reads it, but keep it what self-play would have had.
            _step_count=jnp.int32(self.actions_per_move * moves),
        )
        return state.replace(observation=self._observe(state))

    def between_moves(self, state: Any) -> bool:
        """Whether a pgx state is at a move boundary, where the wire can describe it."""
        return int(state._x.stage) == 0

    def _between_moves(self, state: Any) -> None:
        if not self.between_moves(state):
            raise ValueError("only a state between moves (stage 0) has a wire encoding")


class _Epaminondas(Wire):
    game_id = "epaminondas/1"
    env_id = "epaminondas"
    rules_versions = ("v6", "v7", "v8", "v9", "v10", "v11")
    glyphs = ".WB"
    move_fields = ("lead", "rear", "dest")

    def __init__(self) -> None:
        super().__init__(
            pgx.epaminondas.Epaminondas(),
            epaminondas_game.Game(),
            pgx.epaminondas.State,
            epaminondas_game.N,
        )

    def to_pgx(self, wire: dict[str, Any]) -> Any:
        moves = int(wire["moves"])
        x = epaminondas_game.GameState(
            color=jnp.int32(self._to_move(wire)),
            board=self._board(wire),
            stage=jnp.int32(0),
            lead=jnp.int32(0),
            rear=jnp.int32(0),
            winner=jnp.int32(-1),
            moves=jnp.int32(moves),
            quiet_moves=jnp.int32(int(wire["quiet_moves"])),
            last_capturer=jnp.int32(int(wire["last_capturer"])),
        )
        return self._state(x, self._to_move(wire), moves)

    def from_pgx(self, state: Any) -> dict[str, Any]:
        self._between_moves(state)
        x = state._x
        return {
            "board": self._board_text(x.board),
            "to_move": int(x.color),
            "moves": int(x.moves),
            "quiet_moves": int(x.quiet_moves),
            "last_capturer": int(x.last_capturer),
        }


class _Gess(Wire):
    game_id = "gess/1"
    env_id = "gess"
    rules_versions = ("v1",)
    # pgx stores 1 for black, who moves first, and 2 for white: the other way round to Epaminondas.
    glyphs = ".BW"
    move_fields = ("from", "to")

    def __init__(self) -> None:
        super().__init__(pgx.gess.Gess(), gess_game.Game(), pgx.gess.State, gess_game.N)

    def to_pgx(self, wire: dict[str, Any]) -> Any:
        moves = int(wire["moves"])
        x = gess_game.GameState(
            color=jnp.int32(self._to_move(wire)),
            board=self._board(wire),
            stage=jnp.int32(0),
            # pgx would have the previous move's source here; at stage 0 nothing reads it.
            source=jnp.int32(0),
            winner=jnp.int32(-1),
            no_capture_turns=jnp.int32(int(wire["quiet_moves"])),
        )
        return self._state(x, self._to_move(wire), moves)

    def from_pgx(self, state: Any) -> dict[str, Any]:
        self._between_moves(state)
        x = state._x
        return {
            "board": self._board_text(x.board),
            "to_move": int(x.color),
            # pgx counts actions, not moves, and not at all in Gess: recover the server's counter
            # from the step count, which `_state` set from it.
            "moves": int(state._step_count) // self.actions_per_move,
            "quiet_moves": int(x.no_capture_turns),
        }


class _OneMoreTri(Wire):
    game_id = "one-more-tri/1"
    env_id = "g_hex"
    rules_versions = ("v1",)
    glyphs = ""
    move_fields = ("tri", "tile")
    triangles = 21
    tiles = 10

    def __init__(self) -> None:
        super().__init__(pgx.g_hex.GHex(), g_hex_game.Game(), pgx.g_hex.State, self.triangles)

    @property
    def actions_per_move(self) -> int:
        return 1  # both fields make one action

    def between_moves(self, state: Any) -> bool:
        return True

    def move_to_actions(self, move: dict[str, Any]) -> list[int]:
        tri, tile = int(move["tri"]), int(move["tile"])
        if not (0 <= tri < self.triangles and 1 <= tile <= self.tiles):
            raise ValueError(f"no triangle {tri} or no tile {tile}")
        return [tri * self.tiles + tile - 1]

    def actions_to_move(self, actions: list[int]) -> dict[str, Any]:
        if len(actions) != 1:
            raise ValueError(f"{self.game_id} takes 1 action a move")
        tri, tile = divmod(int(actions[0]), self.tiles)
        return {"tri": tri, "tile": tile + 1}

    def to_pgx(self, wire: dict[str, Any]) -> Any:
        values = [int(v) for v in wire["board"]]
        if len(values) != self.triangles or any(abs(v) > self.tiles for v in values):
            raise ValueError(f"board must be {self.triangles} tile values in -10..10")
        tiles = np.ones((2, self.tiles), dtype=bool)
        for v in values:
            if v:
                colour, index = int(v < 0), abs(v) - 1
                if not tiles[colour, index]:
                    raise ValueError(f"tile {v} is on the board twice")
                tiles[colour, index] = False
        to_move = self._to_move(wire)
        placed = sum(1 for v in values if v)
        if to_move != placed % 2:
            raise ValueError(f"to_move {to_move} with {placed} tiles placed; black moves first")
        # pgx pads the board with a 22nd, always-empty triangle for "no neighbour".
        board = np.zeros(self.triangles + 1, dtype=np.int32)
        board[: self.triangles] = values
        x = g_hex_game.GameState(
            board=jnp.asarray(board),
            tiles=jnp.asarray(tiles),
            color=jnp.int32(to_move),
            winner=jnp.int32(-1),
        )
        return self._state(x, to_move, placed)

    def from_pgx(self, state: Any) -> dict[str, Any]:
        x = state._x
        return {
            "board": [int(v) for v in np.asarray(x.board)[: self.triangles]],
            "to_move": int(x.color),
        }


class _Pig(Wire):
    game_id = "pig/1"
    env_id = "pig"
    # v1 only added a feature to the observation, the banked total if the mover held now.
    rules_versions = ("v0", "v1")
    glyphs = ""
    move_fields = ("action",)
    actions = ("stop", "roll")
    faces = 6

    def __init__(self) -> None:
        super().__init__(pgx.pig.Pig(), pig_game.Game(), pgx.pig.State, 0)
        # pgx rolls straight from the key it is given, so a key it turns into a face forces it.
        keys: dict[int, jnp.ndarray] = {}
        i = 0
        while len(keys) < self.faces:
            key = jax.random.PRNGKey(i)
            keys.setdefault(int(pig_game._roll(key)), key)
            i += 1
        self._face_keys = jnp.stack([keys[face] for face in range(1, self.faces + 1)])

    @property
    def chance_keys(self) -> jnp.ndarray:
        """`chance_keys[r - 1]` makes pgx roll `r`."""
        return self._face_keys

    def between_moves(self, state: Any) -> bool:
        return True

    def move_to_actions(self, move: dict[str, Any]) -> list[int]:
        action = move["action"]
        if action not in self.actions:
            raise ValueError(f"action must be 'stop' or 'roll', not {action!r}")
        return [self.actions.index(action)]

    def actions_to_move(self, actions: list[int]) -> dict[str, Any]:
        if len(actions) != 1:
            raise ValueError(f"{self.game_id} takes 1 action a move")
        return {"action": self.actions[int(actions[0])]}

    def to_pgx(self, wire: dict[str, Any]) -> Any:
        totals = [int(t) for t in wire["totals"]]
        turn_total, last_roll = int(wire["turn_total"]), int(wire["last_roll"])
        if len(totals) != 2 or min(totals) < 0:
            raise ValueError("totals must be two counts")
        if not 1 <= last_roll <= self.faces or turn_total < last_roll:
            raise ValueError(f"no turn total {turn_total} with a last roll of {last_roll}")
        to_move = self._to_move(wire)
        x = pig_game.GameState(
            color=jnp.int32(to_move),
            totals=jnp.asarray(totals, dtype=jnp.int32),
            turn_total=jnp.int32(turn_total),
            last_roll=jnp.int32(last_roll),
            winner=jnp.int32(-1),
        )
        return self._state(x, to_move, 0)

    def from_pgx(self, state: Any) -> dict[str, Any]:
        x = state._x
        return {
            "totals": [int(t) for t in np.asarray(x.totals)],
            "turn_total": int(x.turn_total),
            "last_roll": int(x.last_roll),
            "to_move": int(x.color),
        }


EPAMINONDAS = _Epaminondas()
GESS = _Gess()
ONE_MORE_TRI = _OneMoreTri()
PIG = _Pig()
WIRES: tuple[Wire, ...] = (EPAMINONDAS, GESS, ONE_MORE_TRI, PIG)


def for_env_id(env_id: str) -> Wire:
    for wire in WIRES:
        if wire.env_id == env_id:
            return wire
    known = ", ".join(w.env_id for w in WIRES)
    raise ValueError(f"no wire for pgx env {env_id!r}; this client plays {known}")


def for_game_id(game_id: str) -> Wire:
    for wire in WIRES:
        if wire.game_id == game_id:
            return wire
    known = ", ".join(w.game_id for w in WIRES)
    raise ValueError(f"no wire for {game_id!r}; this client plays {known}")

"""engine_worker.mjs plays a legal move (needs Node 22 and the Gess app's built engine)."""

import json
import os
import subprocess

import pytest

NODE = os.path.expanduser("~/node22/bin/node")
WASM = os.path.expanduser(
    "~/calr3gh/gess/wasm-engine/target/wasm32-unknown-unknown/release/gess_engine_wasm.wasm"
)
WORKER = os.path.join(os.path.dirname(__file__), "..", "engine_worker.mjs")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(NODE) and os.path.exists(WASM)), reason="needs Node 22 and the built engine"
)


def _ask(state: dict) -> dict:
    p = subprocess.run(
        [NODE, WORKER, WASM],
        input=json.dumps({**state, "ms": 200}) + "\n",
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(p.stdout.splitlines()[0])


def _start_board() -> str:
    gess = pytest.importorskip("pgx._src.games.gess")
    board = gess.Game().init().board
    return "".join(".BW"[int(v)] for v in board)


@pytest.mark.parametrize("to_move", [0, 1])
def test_plays_a_legal_move_from_the_start(to_move: int) -> None:
    gess = pytest.importorskip("pgx._src.games.gess")
    jnp = pytest.importorskip("jax.numpy")
    board = _start_board()
    r = _ask({"board": board, "to_move": to_move, "quiet_moves": 0})
    assert "error" not in r
    game = gess.Game()
    x = game.init()._replace(color=jnp.int32(to_move))
    assert bool(game.legal_action_mask(x)[r["from"]])
    assert bool(game.legal_action_mask(game.step(x, r["from"]))[r["to"]])

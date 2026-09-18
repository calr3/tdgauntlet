"""Serve pgx's own alpha-beta engine (`examples/alphazero/negamax.py`) as a tdgauntlet client.

    JAX_PLATFORMS=cpu python -m tdg_jax.negamax --port 9314

This is the fixed reference EPAMINONDAS_EXPERIMENTS.md measures models against, now reachable
over the protocol, so the Rust `tdg-alphabeta` (the same evaluation, searched natively) can be
calibrated against it at an equal clock. pgx's engine decides one action at a time, so a move's
millisecond budget is split evenly across its three actions.

The engine keeps per-search state on itself, so this client searches one position at a time and
says so in /identify.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pysdk"))
from tdgauntlet_client import ClientError, serve  # noqa: E402

from .agent import _use_alphazero_modules  # noqa: E402
from .wire import ENV, GAME_ID, actions_to_move, to_pgx  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=9314)
    ap.add_argument("--name", default="negamax.py")
    ap.add_argument("--default-ms", type=int, default=1500, help="per move, when none is given")
    ap.add_argument("--max-depth", type=int, default=64)
    args = ap.parse_args()

    _use_alphazero_modules()
    import jax
    import jax.numpy as jnp
    from negamax import NegamaxEngine, make_evaluator  # type: ignore[import-not-found]

    engine = NegamaxEngine(
        ENV, make_evaluator("epaminondas"), max_depth=args.max_depth, time_limit_s=1.0
    )
    step = jax.jit(ENV.step)
    lock = threading.Lock()

    def choose_move(game: str, state: Any, budget: dict[str, Any], context: dict[str, Any]) -> Any:
        try:
            s = to_pgx(state)
        except (ValueError, KeyError, TypeError) as e:
            raise ClientError(f"bad state: {e}") from e
        per_action = (budget.get("ms") or args.default_ms) / 3 / 1000
        actions, depths, nodes = [], [], 0
        with lock:
            engine.time_limit_s = per_action
            for _ in range(3):
                action, stats = engine.select_action(s)
                actions.append(action)
                depths.append(stats.depth_reached)
                nodes += stats.nodes
                s = step(s, jnp.int32(action))
        return actions_to_move(actions), {"depth": min(depths), "nodes": nodes}

    serve(
        name=args.name,
        version="pgx",
        games=[GAME_ID],
        choose_move=choose_move,
        port=args.port,
        max_concurrency=1,
    )


if __name__ == "__main__":
    main()

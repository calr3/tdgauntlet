"""Serve a pgx Epaminondas checkpoint as a tdgauntlet client.

    JAX_PLATFORMS=cpu python -m tdg_jax --checkpoint ~/calr3gh/pgx/checkpoints/<run>/000040.ckpt

Run it in the pgx environment. It needs only the standard library beyond what pgx training
already uses: the HTTP side is `clients/pysdk`.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pysdk"))
from tdgauntlet_client import ClientError, serve  # noqa: E402

from .wire import GAME_ID, actions_to_move, to_pgx  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--port", type=int, default=9001)
    ap.add_argument("--name", default=None, help="defaults to the checkpoint's run and step")
    ap.add_argument("--sims", type=int, default=256, help="MCTS simulations per decision")
    ap.add_argument(
        "--policy-only",
        action="store_true",
        help="no search: play the legal action with the largest policy logit (ignores --sims)",
    )
    ap.add_argument("--considered", type=int, default=16, help="max_num_considered_actions")
    ap.add_argument("--max-batch", type=int, default=16, help="requests searched together")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-rules-mismatch", action="store_true")
    args = ap.parse_args()

    name = args.name or (
        f"jax-{args.checkpoint.parent.name}-{args.checkpoint.stem}"
        + ("-policy" if args.policy_only else "")
    )
    loaded = threading.Event()
    agent_box: dict[str, Any] = {}

    def choose_move(game: str, state: Any, budget: dict[str, Any], context: dict[str, Any]) -> Any:
        agent = agent_box["agent"]
        try:
            pgx_state = to_pgx(state)
        except (ValueError, KeyError, TypeError) as e:
            raise ClientError(f"bad state: {e}") from e
        # The simulation count is fixed at startup; a request's `sims` budget is honoured only
        # when it matches, since each distinct count is a separate compilation.
        actions, info = agent.choose(pgx_state)
        return actions_to_move(actions), info

    def load() -> None:
        try:
            from .agent import Agent

            agent = Agent(
                args.checkpoint,
                num_simulations=args.sims,
                max_considered=args.considered,
                max_batch=args.max_batch,
                seed=args.seed,
                allow_rules_mismatch=args.allow_rules_mismatch,
                policy_only=args.policy_only,
            )
            agent.warm_up(lambda size: print(f"{name}: compiled batch {size}", file=sys.stderr))
        except BaseException as e:  # noqa: BLE001
            # Exit rather than sit unhealthy: a server that spawned us then reports it at once.
            print(f"{name}: failed to load: {e}", file=sys.stderr, flush=True)
            os._exit(1)
        agent_box["agent"] = agent
        loaded.set()
        print(f"{name}: ready", file=sys.stderr, flush=True)

    # Open the port straight away and report healthy once the model is in memory, so a server
    # that spawned this process can tell "loading" from "dead".
    threading.Thread(target=load, daemon=True).start()
    serve(
        name=name,
        version="0.1.0",
        games=[GAME_ID],
        choose_move=choose_move,
        port=args.port,
        max_concurrency=args.max_batch,
        ready=loaded.is_set,
    )


if __name__ == "__main__":
    main()

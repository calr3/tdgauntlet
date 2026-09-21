"""Serve a pgx AlphaZero checkpoint as a tdgauntlet client.

    JAX_PLATFORMS=cpu python -m tdg_jax --checkpoint ~/calr3gh/pgx/checkpoints/<run>/000040.ckpt

The checkpoint decides the game: it records the pgx env it was trained on, and the client
advertises the one tdgauntlet game id that goes with it (`epaminondas/1` or `gess/1`).

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

from .agent import load_checkpoint  # noqa: E402


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
    # Which game this is cannot wait for the model: the client has to name it in `identify`.
    # Reading the checkpoint is the quick part of loading it; building the network is not.
    try:
        checkpoint = load_checkpoint(args.checkpoint, args.allow_rules_mismatch)
    except BaseException as e:  # noqa: BLE001
        print(f"{name}: failed to load: {e}", file=sys.stderr, flush=True)
        raise SystemExit(1) from e
    wire = checkpoint.wire
    loaded = threading.Event()
    agent_box: dict[str, Any] = {}

    def choose_move(game: str, state: Any, budget: dict[str, Any], context: dict[str, Any]) -> Any:
        agent = agent_box["agent"]
        try:
            pgx_state = wire.to_pgx(state)
        except (ValueError, KeyError, TypeError) as e:
            raise ClientError(f"bad state: {e}") from e
        # The simulation count is fixed at startup; a request's `sims` budget is honoured only
        # when it matches, since each distinct count is a separate compilation.
        actions, info = agent.choose(pgx_state)
        return wire.actions_to_move(actions), info

    def load() -> None:
        try:
            from .agent import Agent

            agent = Agent(
                checkpoint,
                num_simulations=args.sims,
                max_considered=args.considered,
                max_batch=args.max_batch,
                seed=args.seed,
                policy_only=args.policy_only,
            )
            agent.warm_up(lambda size: print(f"{name}: compiled batch {size}", file=sys.stderr))
        except BaseException as e:  # noqa: BLE001
            # Exit rather than sit unhealthy: a server that spawned us then reports it at once.
            print(f"{name}: failed to load: {e}", file=sys.stderr, flush=True)
            os._exit(1)
        agent_box["agent"] = agent
        loaded.set()
        print(f"{name}: ready to play {wire.game_id}", file=sys.stderr, flush=True)

    # Open the port straight away and report healthy once the model is in memory, so a server
    # that spawned this process can tell "loading" from "dead".
    threading.Thread(target=load, daemon=True).start()
    serve(
        name=name,
        version="0.1.0",
        games=[wire.game_id],
        choose_move=choose_move,
        port=args.port,
        max_concurrency=args.max_batch,
        ready=loaded.is_set,
    )


if __name__ == "__main__":
    main()

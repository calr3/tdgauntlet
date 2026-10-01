"""tdgauntlet client for Gess: the web app's negamax engine (~/calr3gh/gess/wasm-engine).

Each worker is a Node process running one single-threaded instance of the engine
(engine_worker.mjs); requests go to whichever worker is free, and each search gets
the request's `budget.ms`. The app itself searches with several threads, so at the
same time budget this client is somewhat weaker than the app's "Easy (0.5s)" /
"Normal (2s)" levels.

  python clients/gess_negamax/gess_negamax.py --port 9441 --workers 12 \\
      --wasm ~/calr3gh/gess/wasm-engine/target/wasm32-unknown-unknown/release/gess_engine_wasm.wasm
"""

import argparse
import json
import os
import queue
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysdk"))
from tdgauntlet_client import ClientError, serve  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=9441)
p.add_argument("--workers", type=int, default=8)
p.add_argument("--name", default="negamax")
p.add_argument("--node", default=os.path.expanduser("~/node22/bin/node"))
p.add_argument(
    "--wasm",
    default=os.path.expanduser(
        "~/calr3gh/gess/wasm-engine/target/wasm32-unknown-unknown/release/gess_engine_wasm.wasm"
    ),
)
p.add_argument("--default-ms", type=int, default=1000)
args = p.parse_args()

worker_js = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine_worker.mjs")
free: "queue.Queue[subprocess.Popen]" = queue.Queue()
for _ in range(args.workers):
    free.put(
        subprocess.Popen(
            [args.node, worker_js, args.wasm],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    )


def choose_move(game, state, budget, context):
    ms = int(budget.get("ms") or args.default_ms)
    w = free.get()
    try:
        w.stdin.write(
            json.dumps(
                {
                    "board": state["board"],
                    "to_move": state["to_move"],
                    "quiet_moves": state["quiet_moves"],
                    "ms": ms,
                }
            )
            + "\n"
        )
        w.stdin.flush()
        r = json.loads(w.stdout.readline())
    finally:
        free.put(w)
    if "error" in r:
        raise ClientError(r["error"])
    return {"from": r["from"], "to": r["to"]}, {
        "depth": r["depth"],
        "score": r["score"],
        "elapsed_ms": r["ms"],
    }


serve(
    name=args.name,
    games=["gess/1"],
    choose_move=choose_move,
    port=args.port,
    max_concurrency=args.workers,
)

"""Find a player's blunders in Gess games, with the web app's negamax engine as referee.

For every move `player` made in the games it lost, the negamax engine (engine_worker.mjs)
finds its own best move in that position, then scores the position after the move played and
after its best move, both with the opponent to move and the same search time (comparing
scores with different sides to move is biased: the side to move scores ~3 stones better at
these depths). A move is a blunder when it is worse than negamax's best by at least --drop
(engine units: 100 per stone, 70 per ring, ~1e6 for a forced win) or walks into a forced
loss. The player's own evaluation of the move (`info.eval` from the game record) shows
whether it saw the problem.

With --recheck URL (repeatable), the first blunder of each game is also sent to those
tdgauntlet clients (e.g. the same network at more simulations) and their moves are refereed
the same way: does more search avoid the blunder?

  python clients/gess_negamax/blunders.py results/gess_negamax_pilot.json --player e22 \\
      --opponent negamax2s --out results/gess_negamax_pilot.blunders.json
"""

import argparse
import json
import os
import queue
import subprocess
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("results")
p.add_argument("--player", required=True)
p.add_argument("--opponent", required=True)
p.add_argument("--ms", type=int, default=1000, help="referee search time per position")
p.add_argument("--drop", type=int, default=200, help="blunder threshold, engine units")
p.add_argument("--workers", type=int, default=12)
p.add_argument("--recheck", action="append", default=[], metavar="URL")
p.add_argument("--out")
p.add_argument("--node", default=os.path.expanduser("~/node22/bin/node"))
p.add_argument(
    "--wasm",
    default=os.path.expanduser(
        "~/calr3gh/gess/wasm-engine/target/wasm32-unknown-unknown/release/gess_engine_wasm.wasm"
    ),
)
args = p.parse_args()

# Rules: replay with pgx's Gess (CPU is plenty).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from pgx._src.games import gess as G  # noqa: E402

game = G.Game()
gstep = jax.jit(game.step)
WIN = 500_000

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


def search(state):
    """Negamax's best move and score for the side to move in `state` (a gess/1 state dict)."""
    w = free.get()
    try:
        w.stdin.write(json.dumps({**state, "ms": args.ms}) + "\n")
        w.stdin.flush()
        r = json.loads(w.stdout.readline())
    finally:
        free.put(w)
    return (None, None) if "error" in r else ({"from": r["from"], "to": r["to"]}, r["score"])


def wire(x, moves):
    b = np.asarray(x.board).reshape(-1)
    return {
        "board": "".join(".BW"[v] for v in b),
        "to_move": int(x.color),
        "moves": moves,
        "quiet_moves": int(x.no_capture_turns),
    }


def after(state, mv):
    x = from_wire(state)
    x = gstep(x, mv["from"])
    x = gstep(x, mv["to"])
    return x


def from_wire(st):
    x = game.init()
    b = np.array([".BW".index(c) for c in st["board"]], dtype=np.asarray(x.board).dtype)
    return x._replace(
        board=jnp.asarray(b.reshape(np.asarray(x.board).shape)),
        color=jnp.int32(st["to_move"]),
        no_capture_turns=jnp.int32(st["quiet_moves"]),
    )


def value_after(state, mv):
    """Negamax's score, for the mover, of the position after `mv` (opponent to move)."""
    x = after(state, mv)
    if bool(game.is_terminal(x)):
        r = float(np.asarray(game.rewards(x))[state["to_move"]])
        return WIN if r > 0 else -WIN if r < 0 else 0
    _, a = search(wire(x, state["moves"] + 1))
    return None if a is None else -a


def same(a, b):
    return a["from"] == b["from"] and a["to"] == b["to"]


def judge(state, mv, best=None):
    """(value of negamax's best move, value of `mv`, shortfall, negamax's best move)."""
    if best is None:
        best, _ = search(state)
    if best is None:
        return None, None, None, None
    vb = value_after(state, best)
    vm = vb if same(best, mv) else value_after(state, mv)
    if vb is None or vm is None:
        return vb, vm, None, best
    return vb, vm, max(vb - vm, 0), best


def is_blunder(best, mine, drop):
    if mine is None or best is None:
        return False
    return drop >= args.drop or (mine <= -WIN and best > -WIN)


D = json.load(open(args.results))
OPEN = {o["id"]: o["state"] for o in D["openings"]}
lost = [
    g
    for g in D["games"]
    if args.player in g["players"]
    and args.opponent in g["players"]
    and g["outcome"]["result"] == "win"
    and g["players"][g["outcome"]["side"]] == args.opponent
]
print(f"{len(lost)} games {args.player} lost to {args.opponent}", flush=True)
print = __import__("functools").partial(print, flush=True)

jobs = []  # (game id, ply, state, move, own eval)
for g in lost:
    x = from_wire(OPEN[g["opening"]])
    n = OPEN[g["opening"]]["moves"]
    for ply, m in enumerate(g["moves"]):
        st = wire(x, n + ply)
        if g["players"][m["side"]] == args.player:
            jobs.append((g["id"], ply, st, m["move"], (m.get("info") or {}).get("eval")))
        x = gstep(x, m["move"]["from"])
        x = gstep(x, m["move"]["to"])

with ThreadPoolExecutor(args.workers) as ex:
    judged = list(ex.map(lambda j: judge(j[2], j[3]), jobs))

rows = []
for (gid, ply, st, mv, ev), (vbest, mine, drop, best) in zip(jobs, judged, strict=True):
    rows.append(
        dict(
            game=gid,
            ply=ply,
            state=st,
            move={"from": mv["from"], "to": mv["to"]},
            own_eval=ev,
            best=best,
            best_value=vbest,
            value=mine,
            drop=drop,
            agrees=best is not None and same(best, mv),
            blunder=is_blunder(vbest, mine, drop),
        )
    )
bl = [r for r in rows if r["blunder"]]
per_game: dict[str, list[dict]] = {}
for r in rows:
    per_game.setdefault(r["game"], []).append(r)
first = [next((r for r in rs if r["blunder"]), None) for rs in per_game.values()]
print(f"{len(rows)} moves judged; {len(bl)} blunders (drop >= {args.drop} or into a forced loss)")
print(f"games with a blunder: {sum(f is not None for f in first)} of {len(per_game)}")
print(
    f"{args.player} played negamax's own choice in "
    f"{np.mean([r['agrees'] for r in rows]):.0%} of moves; "
    f"median shortfall {np.median([r['drop'] for r in rows if r['drop'] is not None]):.0f}"
)
into_loss = [r for r in bl if r["value"] is not None and r["value"] <= -WIN]
print(f"  of the blunders, into a forced loss (negamax sees the win): {len(into_loss)}")
evs = [r["own_eval"] for r in bl if r["own_eval"] is not None]
if evs:
    print(
        f"  {args.player}'s own eval of its blunders: median {np.median(evs):+.2f}; "
        f"still >= 0 in {np.mean(np.array(evs) >= 0):.0%}"
    )
fp = [f["ply"] for f in first if f is not None]
if fp:
    print(
        f"  first blunder at ply: median {np.median(fp):.0f} (games' median length "
        f"{np.median([g['final_state']['moves'] for g in lost]):.0f} moves)"
    )


def ask(url, st):
    req = {
        "protocol": 1,
        "request_id": str(uuid.uuid4()),
        "game": "gess/1",
        "state": st,
        "budget": {"ms": None, "nodes": None, "sims": None},
        "context": {},
    }
    data = urllib.request.urlopen(
        urllib.request.Request(
            url + "/move", json.dumps(req).encode(), {"Content-Type": "application/json"}
        ),
        timeout=600,
    ).read()
    return json.loads(data)["move"]


firsts = [f for f in first if f is not None]
for url in args.recheck:
    with ThreadPoolExecutor(16) as ex:
        mvs = list(ex.map(lambda r, url=url: ask(url, r["state"]), firsts))
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(
            ex.map(
                lambda a: judge(a[0]["state"], a[1], a[0]["best"]), zip(firsts, mvs, strict=True)
            )
        )
    avoided = [not is_blunder(b, m, d) for b, m, d, _ in res]
    kept = [same(mv, r["move"]) for mv, r in zip(mvs, firsts, strict=True)]
    for r, mv, ok in zip(firsts, mvs, avoided, strict=True):
        r.setdefault("recheck", {})[url] = dict(move=mv, avoided=ok)
    print(
        f"recheck {url}: on each game's first blunder ({len(firsts)}), avoids it in "
        f"{np.mean(avoided):.0%} (plays the same move {np.mean(kept):.0%})"
    )

if args.out:
    json.dump(dict(rows=rows), open(args.out, "w"))
    print(f"wrote {args.out}")

"""After a player's moves into a forced loss (from blunders.py), how does its own network rate
the opponent's winning reply?

For each such position (opponent to move), negamax's best reply is the "killer". We record the
killer's slide distance, stones captured and whether it wins at once, the network's prior on it
(piece rank at stage 0, destination rank at stage 1 given that piece), and whether the network's
own 128-simulation Gumbel search, playing the opponent's side, finds it.

  JAX_PLATFORMS=cpu python clients/gess_negamax/killer_priors.py \\
      results/gess_negamax_pilot.blunders.json ~/calr3gh/pgx/checkpoints/<run>/<step>.ckpt
"""

import json
import os
import pickle
import subprocess
import sys

import numpy as np

blunders_path, ckpt_path = sys.argv[1:3]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
PGX = os.path.expanduser("~/calr3gh/pgx")
sys.path.insert(0, os.path.join(PGX, "examples/alphazero"))
os.chdir(PGX)
from config import Config  # noqa: E402

import __main__  # noqa: E402

__main__.Config = Config
cfg = pickle.load(open(ckpt_path, "rb"))["config"]
sys.argv = (
    ["x"]
    + [
        f"{k}={int(v) if k == 'gf_joint_head' else v}"
        for k, v in cfg.model_dump().items()
        if k.startswith(("env_id", "architecture", "gf_", "obs_planes"))
    ]
    + ["selfplay_bf16=false"]
)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mctx  # noqa: E402
import train  # noqa: E402
from pgx._src.games import gess as G  # noqa: E402

env, game = train.env, G.Game()
model = pickle.load(open(ckpt_path, "rb"))["model"]
WIN = 500_000

rows = json.load(open(blunders_path))["rows"]
cases = [r for r in rows if r["blunder"] and r["value"] is not None and r["value"] <= -WIN]
print(f"{len(cases)} moves into a forced loss", flush=True)

node = os.path.expanduser("~/node22/bin/node")
wasm = os.path.expanduser(
    "~/calr3gh/gess/wasm-engine/target/wasm32-unknown-unknown/release/gess_engine_wasm.wasm"
)
here = os.path.dirname(os.path.abspath(__file__))
eng = subprocess.Popen(
    [node, os.path.join(here, "engine_worker.mjs"), wasm],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    bufsize=1,
)


def from_wire(st):
    x = game.init()
    b = np.array([".BW".index(c) for c in st["board"]], dtype=np.asarray(x.board).dtype)
    return x._replace(
        board=jnp.asarray(b.reshape(np.asarray(x.board).shape)),
        color=jnp.int32(st["to_move"]),
        no_capture_turns=jnp.int32(st["quiet_moves"]),
    )


def wire(x):
    return {
        "board": "".join(".BW"[v] for v in np.asarray(x.board).reshape(-1)),
        "to_move": int(x.color),
        "quiet_moves": int(x.no_capture_turns),
    }


@jax.jit
def to_state(x):
    s = env.init(jax.random.PRNGKey(0))
    s = s.replace(_x=x, current_player=x.color, legal_action_mask=game.legal_action_mask(x))
    return s.replace(observation=env.observe(s, s.current_player))


@jax.jit
def priors(obs):
    (logits, _), _ = train.forward.apply(model[0], model[1], obs, is_eval=True)
    return logits


@jax.jit
def search(states, key):
    (logits, v), _ = train.forward.apply(model[0], model[1], states.observation, is_eval=True)
    root = mctx.RootFnOutput(prior_logits=logits, value=v, embedding=states)
    return mctx.gumbel_muzero_policy(
        params=model,
        rng_key=key,
        root=root,
        recurrent_fn=train.recurrent_fn,
        num_simulations=64,
        invalid_actions=~states.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=1.0,
    )


def rank_prob(logits, mask, a):
    lm = np.where(mask, logits, -np.inf)
    p = np.exp(lm - lm.max())
    p /= p.sum()
    return int((lm > lm[a]).sum()), float(p[a])


gstep = jax.jit(game.step)
out = []
for r in cases:
    x = from_wire(r["state"])
    x = gstep(x, r["move"]["from"])
    x = gstep(x, r["move"]["to"])  # opponent to move
    assert eng.stdin is not None and eng.stdout is not None
    eng.stdin.write(json.dumps({**wire(x), "ms": 1500}) + "\n")
    eng.stdin.flush()
    k = json.loads(eng.stdout.readline())
    if "error" in k:
        continue
    s, d = k["from"], k["to"]
    mover, opp = 1 + int(x.color), 2 - int(x.color)
    y = gstep(gstep(x, s), d)
    caps = int((np.asarray(x.board) == opp).sum() - (np.asarray(y.board) == opp).sum())
    wins_now = bool(game.is_terminal(y)) and int(y.winner) == int(x.color)
    dist = max(abs(s // 20 - d // 20), abs(s % 20 - d % 20))
    st0 = to_state(x)
    l0 = np.asarray(priors(st0.observation[None])[0])
    r0, p0 = rank_prob(l0, np.asarray(st0.legal_action_mask), s)
    st1 = env.step(st0, s)
    l1 = np.asarray(priors(st1.observation[None])[0])
    r1, p1 = rank_prob(l1, np.asarray(st1.legal_action_mask), d)
    out.append(
        dict(
            dist=dist,
            caps=caps,
            wins_now=wins_now,
            src_rank=r0,
            p_src=p0,
            dst_rank=r1,
            p_dst=p1,
            n_dst=int(np.asarray(st1.legal_action_mask).sum()),
            state=wire(x),
            killer=(s, d),
        )
    )

# the network's own search, as the opponent, at these positions (two stages)
B = len(out)
sts = jax.tree_util.tree_map(
    lambda *a: jnp.stack(a), *[to_state(from_wire(o["state"])) for o in out]
)
a0 = np.asarray(search(sts, jax.random.PRNGKey(0)).action)
sts1 = jax.vmap(env.step)(sts, jnp.asarray(a0))
a1 = np.asarray(search(sts1, jax.random.PRNGKey(1)).action)
for o, s_, d_ in zip(out, a0, a1, strict=True):
    o["search_finds"] = (int(s_), int(d_)) == tuple(o["killer"])

print(f"{len(out)} killer replies (negamax's best reply after the blunder)")
w = np.array([o["wins_now"] for o in out])
dist = np.array([o["dist"] for o in out])


def med(key):
    return np.median([o[key] for o in out])


def frac(pred, rows):
    return np.mean([pred(o) for o in rows])


caps = med("caps")
print(
    f"  wins on the spot: {w.mean():.0%}; slide distance median {np.median(dist):.0f}, "
    f">= 4 squares {np.mean(dist >= 4):.0%}; stones captured median {caps:.0f}"
)
top16 = frac(lambda o: o["src_rank"] < 16, out)
print(
    f"  network prior, piece step: rank median {med('src_rank'):.0f}, "
    f"in top 16 {top16:.0%}, prob median {med('p_src'):.3f}"
)
first = frac(lambda o: o["dst_rank"] == 0, out)
print(
    f"  network prior, destination step (given the piece): ranked first {first:.0%}, "
    f"prob median {med('p_dst'):.3f}"
)
finds = frac(lambda o: o["search_finds"], out)
print(f"  the network's own 64-sim search, as the opponent, plays the killer: {finds:.0%}")
for name, sel in (("immediate wins", w), ("not immediate", ~w)):
    if sel.any():
        sub = [o for o, m in zip(out, sel, strict=True) if m]
        print(
            f"  {name} ({len(sub)}): "
            f"piece in top 16 {frac(lambda o: o['src_rank'] < 16, sub):.0%}, "
            f"dest first {frac(lambda o: o['dst_rank'] == 0, sub):.0%}, "
            f"search finds {frac(lambda o: o['search_finds'], sub):.0%}"
        )
json.dump(out, open(blunders_path.replace(".json", ".killers.json"), "w"))
eng.kill()

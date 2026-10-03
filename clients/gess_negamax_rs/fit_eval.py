"""Fit the Gess negamax evaluation's weights to game results (Texel-style).

    python clients/gess_negamax_rs/fit_eval.py results/gess_eval_positions.jsonl
        [--out weights.json] [--terms stone,ring,...] [--min-quiet N]
        [--source selfplay|models|all]

Input: `examples/dump.rs`'s JSON lines (features and result from the side to move's view).
Model: P(side to move wins) = sigmoid(c + sum_i b_i * f_i), fitted by maximum likelihood with
draws as half a win; the intercept c absorbs any side-to-move advantage and is not part of the
engine. Engine weights are the slopes rescaled so the stone weight is 100 (the engine's unit,
a hundredth of a stone): w_i = 100 * b_i / b_stone. 20% of games are held out (by game, not
position); the default weights are scored with their own best-fitting scale for a fair
comparison.
"""

from __future__ import annotations

import argparse
import json
import zlib

import numpy as np
from scipy.optimize import minimize

NAMES = [
    "stone",
    "ring",
    "ring_pair",
    "deep",
    "pressure",
    "threat",
    "defenders",
    "mobile",
    "edge",
    "quiet_stone",
]
DEFAULT = np.array([100, 700, -210, 1, 10, 0, 0, 0, 0, 0], dtype=float)


def load(path: str, min_quiet: int, source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats, results, held = [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["quiet"] < min_quiet:
                continue
            if source == "selfplay" and r["source"] != "selfplay":
                continue
            if source == "models" and r["source"] == "selfplay":
                continue
            feats.append(r["features"])
            results.append(r["result"])
            held.append(zlib.crc32(r["game"].encode()) % 5 == 0)
    return np.array(feats, dtype=float), np.array(results), np.array(held)


def nll(z: np.ndarray, y: np.ndarray) -> float:
    """Mean cross-entropy of results y (0, 0.5, 1) under logits z."""
    return float(np.mean(np.logaddexp(0, z) - y * z))


def fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Intercept and slopes maximising the likelihood."""
    xa = np.column_stack([np.ones(len(x)), x])

    def f(p: np.ndarray) -> tuple[float, np.ndarray]:
        z = xa @ p
        g = xa.T @ (1 / (1 + np.exp(-z)) - y) / len(y)
        return nll(z, y), g

    return minimize(f, np.zeros(xa.shape[1]), jac=True, method="L-BFGS-B").x


def fit_scale(e: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Intercept and scale for a fixed evaluation e: P = sigmoid(c + k * e)."""
    p = fit(e[:, None], y)
    return float(p[0]), float(p[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("positions")
    ap.add_argument("--out")
    ap.add_argument("--min-quiet", type=int, default=0)
    ap.add_argument("--source", default="all", choices=["all", "selfplay", "models"])
    ap.add_argument(
        "--terms",
        default=",".join(NAMES[:5]),
        help="comma-separated terms to fit (stone first); the others stay at 0",
    )
    a = ap.parse_args()
    x, y, held = load(a.positions, a.min_quiet, a.source)
    use = [NAMES.index(t) for t in a.terms.split(",")]
    assert use[0] == 0, "stone must come first: it sets the scale"
    tr, te = ~held, held
    print(f"{len(y)} positions ({tr.sum()} train, {te.sum()} held out); mean result {y.mean():.3f}")

    # The default weights, with their best scale.
    e = x @ DEFAULT
    c0, k0 = fit_scale(e[tr], y[tr])
    base = nll(c0 + k0 * e[te], y[te])

    # Free slopes for the chosen terms.
    p = fit(x[tr][:, use], y[tr])
    c = p[0]
    b = np.zeros(len(NAMES))
    b[use] = p[1:]
    fitted = nll(c + x[te] @ b, y[te])
    w = 100 * b / b[0]
    const = nll(np.full(te.sum(), np.log(y[tr].mean() / (1 - y[tr].mean()))), y[te])

    print(
        f"held-out log-loss: constant {const:.5f}, default weights {base:.5f}, fitted {fitted:.5f}"
    )
    print(f"  default: scale {1 / k0:.0f} eval units per logit, intercept {c0:+.3f}")
    print(
        f"  fitted:  scale {100 / b[0]:.0f} per logit (a stone is {b[0]:.3f} logits), "
        f"intercept {c:+.3f}"
    )
    print("weight      default   fitted")
    for n, d, f in zip(NAMES, DEFAULT, w, strict=True):
        print(f"{n:<10} {d:>8.0f} {f:>8.1f}")
    if a.out:
        out = {n: int(round(v)) for n, v in zip(NAMES, w, strict=True)}
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {a.out}: {out}")


if __name__ == "__main__":
    main()

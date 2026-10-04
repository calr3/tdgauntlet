"""Fit the Gess negamax evaluation's weights to game results (Texel-style).

    python clients/gess_negamax_rs/fit_eval.py results/gess_eval_positions.jsonl
        [--out weights.json] [--terms stone,ring,...] [--min-quiet N]
        [--source selfplay|models|all] [--quiet-only] [--anchor] [--patterns --ridge L]

Input: `examples/dump.rs`'s JSON lines (features and result from the side to move's view).
Model: P(side to move wins) = sigmoid(c + sum_i b_i * f_i), fitted by maximum likelihood with
draws as half a win; the intercept c absorbs any side-to-move advantage and is not part of the
engine. Engine weights are the slopes rescaled so the stone weight is 100 (the engine's unit,
a hundredth of a stone): w_i = 100 * b_i / b_stone. 20% of games are held out (by game, not
position); the default weights are scored with their own best-fitting scale for a fair
comparison.

With --patterns, the 288 piece pattern classes (`pattern_counts` in the engine; the ring class
left out, as the ring term covers it) join the chosen terms, their slopes under an L2 penalty of
`--ridge` per logit squared, and are written as the weights file's `patterns` list.
"""

from __future__ import annotations

import argparse
import json
import zlib

import numpy as np
import scipy.sparse as sp
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
    "spare_ring",
    "material_ratio",
]
DEFAULT = np.array([100, 700, -210, 1, 10, 0, 0, 0, 0, 0, 0, 0], dtype=float)


def _mirror(p: int) -> int:
    return sum(int(f"{(p >> (3 * r)) & 7:03b}"[::-1], 2) << (3 * r) for r in range(3))


# The engine's numbering (`pattern_class`): each pattern and its left-right mirror share the
# rank of the smaller among all such pairs.
_CANON = sorted({min(p, _mirror(p)) for p in range(512)})
NUM_PATTERNS = len(_CANON)
RING_CLASS = _CANON.index(0o757)  # covered by the ring term


def load(
    paths: list[str], min_quiet: int, source: str, quiet_only: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, sp.csr_matrix]:
    feats: list[list[int]] = []
    results: list[float] = []
    held: list[bool] = []
    rows: list[int] = []
    cols: list[int] = []
    vals: list[int] = []
    for path in paths:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r["quiet"] < min_quiet:
                    continue
                if quiet_only and r.get("captured", True):
                    continue
                if source == "selfplay" and r["source"] != "selfplay":
                    continue
                if source == "models" and r["source"] == "selfplay":
                    continue
                for k, n in r.get("patterns", []):
                    rows.append(len(results))
                    cols.append(k)
                    vals.append(n)
                feats.append(r["features"])
                results.append(r["result"])
                held.append(zlib.crc32(r["game"].encode()) % 5 == 0)
    pats = sp.csr_matrix(
        (np.array(vals, dtype=float), (rows, cols)), shape=(len(results), NUM_PATTERNS)
    )
    return np.array(feats, dtype=float), np.array(results), np.array(held), pats


def nll(z: np.ndarray, y: np.ndarray) -> float:
    """Mean cross-entropy of results y (0, 0.5, 1) under logits z."""
    return float(np.mean(np.logaddexp(0, z) - y * z))


def fit(x: np.ndarray | sp.spmatrix, y: np.ndarray, ridge: np.ndarray | None = None) -> np.ndarray:
    """Intercept and slopes maximising the likelihood, less `ridge[i] / 2 * slope_i^2`."""
    xa = (
        sp.hstack([np.ones((x.shape[0], 1)), x]).tocsr()
        if sp.issparse(x)
        else (np.column_stack([np.ones(len(x)), x]))
    )
    lam = np.zeros(xa.shape[1]) if ridge is None else np.concatenate([[0.0], ridge])

    def f(p: np.ndarray) -> tuple[float, np.ndarray]:
        z = xa @ p
        g = xa.T @ (1 / (1 + np.exp(-z)) - y) / len(y) + lam * p
        return nll(z, y) + float(lam @ (p * p)) / 2, np.asarray(g).ravel()

    return minimize(f, np.zeros(xa.shape[1]), jac=True, method="L-BFGS-B").x


def fit_scale(e: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Intercept and scale for a fixed evaluation e: P = sigmoid(c + k * e)."""
    p = fit(e[:, None], y)
    return float(p[0]), float(p[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("positions", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--min-quiet", type=int, default=0)
    ap.add_argument("--source", default="all", choices=["all", "selfplay", "models"])
    ap.add_argument(
        "--anchor",
        action="store_true",
        help="weights on the five-term fit's scale (100 = its stone), not this one's",
    )
    ap.add_argument(
        "--quiet-only",
        action="store_true",
        help="only positions whose next move captured nothing (static eval is meaningful there)",
    )
    ap.add_argument("--patterns", action="store_true", help="fit piece pattern weights too")
    ap.add_argument("--ridge", type=float, default=1e-4, help="L2 penalty on pattern slopes")
    ap.add_argument(
        "--terms",
        default=",".join(NAMES[:5]),
        help="comma-separated terms to fit (stone first); the others stay at 0",
    )
    a = ap.parse_args()
    x, y, held, pats = load(a.positions, a.min_quiet, a.source, a.quiet_only)
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
    pw = np.zeros(NUM_PATTERNS)
    if a.patterns:
        cols = [k for k in range(NUM_PATTERNS) if k != RING_CLASS]
        xs = sp.hstack([sp.csr_matrix(x[:, use]), pats[:, cols]]).tocsr()
        ridge = np.concatenate([np.zeros(len(use)), np.full(len(cols), a.ridge)])
        p = fit(xs[tr], y[tr], ridge)
        c = p[0]
        b = np.zeros(len(NAMES))
        b[use] = p[1 : 1 + len(use)]
        pw[cols] = p[1 + len(use) :]
        fitted = nll(c + xs[te] @ p[1:], y[te])
        print(f"pattern slopes: {np.count_nonzero(np.abs(pw) > 1e-3)} non-negligible")
    if a.anchor:
        # Scale from the app's five terms fitted to the same data, so a term that takes over
        # the stone term's role (e.g. material_ratio) cannot blow the ratios up.
        base_b = fit(x[tr][:, :5], y[tr])[1:]
        w = 100 * b / base_b[0]
    else:
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
    if a.patterns:
        pwe = 100 * pw / b[0] if not a.anchor else 100 * pw / base_b[0]
        order = np.argsort(pwe)
        print("patterns, most negative:", [(int(k), round(pwe[k])) for k in order[:6]])
        print("patterns, most positive:", [(int(k), round(pwe[k])) for k in order[-6:]])
    if a.out:
        out: dict[str, object] = {n: int(round(v)) for n, v in zip(NAMES, w, strict=True)}
        if a.patterns:
            out["patterns"] = [int(round(v)) for v in pwe]
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {a.out}: {out}")


if __name__ == "__main__":
    main()

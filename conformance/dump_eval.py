"""Score conformance positions with pgx's alpha-beta evaluator, for the Rust port to match.

Run in the pgx environment after dump_pgx.py:

    JAX_PLATFORMS=cpu python conformance/dump_eval.py

Reads epaminondas.jsonl.gz and writes epaminondas_eval.json: a spread of positions, each with
the six LEONIDAS heuristic terms and the weighted total from
`pgx/examples/alphazero/negamax.py::EpaminondasEvaluator`, from the side to move's point of view.
clients/alphabeta checks its evaluation against these to within floating-point noise.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402
import pgx  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(pgx.__file__).resolve().parent.parent / "examples" / "alphazero"))
from negamax import EpaminondasEvaluator, _runs_all_dirs  # noqa: E402


def main() -> None:
    with gzip.open(HERE / "epaminondas.jsonl.gz", "rt", encoding="utf-8") as f:
        records = [json.loads(line) for line in f][1:]
    # Every 10th position: a spread of openings, middlegames and clock endings.
    chosen = records[::10]
    boards = np.array([[".WB".index(c) for c in r["state"]["board"]] for r in chosen], np.int8)
    colors = np.array([r["state"]["to_move"] for r in chosen])

    ev = EpaminondasEvaluator()
    own = boards == (colors + 1)[:, None]
    opp = boards == (2 - colors)[:, None]
    empty = boards == 0
    terms = {
        "material": ev._material(own, opp),
        "crossing": ev._crossing(own, opp, colors),
        "home_row_defense": ev._home_row_defense(own, opp, colors),
        "territory": ev._territory(own, opp, empty),
        "centrality": ev._centrality(own, opp),
        "mobility": ev._mobility_from_runs(
            _runs_all_dirs(own), _runs_all_dirs(opp), _runs_all_dirs(empty)
        ),
    }
    total = ev.score(boards, colors)
    w = ev.weights
    cases = [
        {
            "state": r["state"],
            "terms": {k: float(v[i]) for k, v in terms.items()},
            "total": float(total[i]),
        }
        for i, r in enumerate(chosen)
    ]
    out = {
        "weights": {k: getattr(w, k) for k in terms},
        "cases": cases,
    }
    (HERE / "epaminondas_eval.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {len(cases)} evaluations", file=sys.stderr)


if __name__ == "__main__":
    main()

r"""
Run a few questions through the full audit (exactly what the website's "Intercept & audit" button does) so that they
are cached and appear in the website's History panel.

  python scripts/showcase_queries.py                 # 5 questions from data/benchmark_100_queries.txt
  python scripts/showcase_queries.py Q001 Q025 Q085  # any IDs from that file

Cost: two cheap Claude Haiku calls per question (about $0.002 each); repeated questions are served from cache.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

DEFAULT = ["Q001", "Q025", "Q012", "Q047", "Q085"]


def main(ids):
    from kg_audit.pipelines.evaluation_pipeline import EvaluationPipeline
    from kg_audit.utils import workspace as ws
    qs = {}
    with open(os.path.join("data", "benchmark_100_queries.txt"), encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5:
                qs[parts[0]] = (parts[1], parts[-1])
    pipe = EvaluationPipeline(use_mlflow=False, workspace_id=ws.active_workspace_id())
    for qid in ids:
        cat, q = qs[qid]
        _, _, ev = pipe.run_pipeline(q, mode="closed_book")
        d = ev.diagnostics
        print(f"{qid} [{cat}] -> {ev.decision_status:12s} P(hallucination)={d.get('hallucination_probability')}  "
              f"unsupported share={d.get('hallucination_extent')}  claims={d.get('response_claims')}\n   {q[:110]}")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)

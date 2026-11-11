"""
Audit a batch of user queries (no labels needed): every line of a .txt file is one question.

  Stage 1  INTERCEPT  answers + claim graphs for all queries      (I/O thread pool, cached)
  Stage 2  EVALUATE   CASSE features, hallucination probability & extent, verdict (sequential)

Results are written to <workspace>/batches/<batch_id>/ (results.jsonl, results.csv, summary.json)
and the website re-opens the latest batch automatically.

  python scripts/run_batch.py --file my_queries.txt
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
import run_benchmark as rb  # noqa: E402  (sets cwd to the project root)

from kg_audit.utils.api_errors import friendly  # noqa: E402
from kg_audit.pipelines.evaluation_pipeline import EvaluationPipeline  # noqa: E402
from kg_audit.utils import workspace as ws  # noqa: E402
from kg_audit.utils.main_utils import limit_cpu_threads, write_json_atomic  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--file", required=True, help=".txt (one question per line) or .jsonl")
    ap.add_argument("--batch-id", default="")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cpu-threads", type=int, default=2)
    ap.add_argument("--mode", choices=["closed_book", "grounded"], default="closed_book")
    args = ap.parse_args()

    rb.be_nice()
    os.environ["EPISTEME_CPU_THREADS"] = str(args.cpu_threads)
    limit_cpu_threads(args.cpu_threads)
    ws_id = args.workspace or ws.active_workspace_id()
    batch_id = args.batch_id or f"batch_{datetime.now():%Y%m%d_%H%M%S}"
    batch_dir = os.path.join(ws.workspace_dir(ws_id), "batches", batch_id)
    os.makedirs(batch_dir, exist_ok=True)
    src = os.path.join(batch_dir, "queries" + os.path.splitext(args.file)[1])
    if os.path.abspath(args.file) != os.path.abspath(src):
        shutil.copy2(args.file, src)

    prog = rb.Progress(batch_dir)
    items = rb.load_items(src)
    if not items:
        prog.finish("failed", error="the file contains no questions")
        sys.exit("No questions found.")
    write_json_atomic(os.path.join(batch_dir, "config.json"), {
        "batch_id": batch_id, "workspace": ws_id, "n": len(items), "mode": args.mode, "source_file": os.path.basename(args.file),
        "started_at": datetime.now(timezone.utc).isoformat()})
    pipe = EvaluationPipeline(use_mlflow=False, workspace_id=ws_id)
    try:
        rb.stage_intercept(items, batch_dir, pipe, args.workers, prog, args.mode, label="intercept")
        df = rb.stage_features(items, batch_dir, pipe, prog, args.mode, require_judgement=False, label="evaluate")
        df.to_csv(os.path.join(batch_dir, "results.csv"), index=False)
        dec = df["decision"].value_counts().to_dict() if len(df) else {}
        summary = {
            "batch_id": batch_id, "n": int(len(df)), "decisions": {k: int(v) for k, v in dec.items()},
            "mean_hallucination_extent": float(df["hallucination_extent"].mean()) if len(df) else None,
            "mean_hallucination_probability": (float(df["hallucination_probability"].dropna().mean())
                                               if len(df) and df["hallucination_probability"].notna().any() else None),
            "mean_llm_confidence": float(df["c_llm"].mean()) if len(df) else None,
            "brain_transferred": bool(df["brain_transferred"].any()) if len(df) else False,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json_atomic(os.path.join(batch_dir, "summary.json"), summary)
        write_json_atomic(os.path.join(ws.workspace_dir(ws_id), "batches", "latest.json"), {"batch_id": batch_id})
        prog.finish("completed", result=summary)
    except Exception as e:
        prog.finish("failed", error=friendly(e))
        raise


if __name__ == "__main__":
    main()

r"""
Remove files that are no longer needed (old experiment logs, superseded brains, per-item debug output).

  python scripts/cleanup.py            # DRY RUN: lists what would be deleted and how much space it frees
  python scripts/cleanup.py --apply    # actually deletes

Never touched (needed to reproduce every number at $0 or to run the website):
  Artifacts/query_cache, llm_cache, script_cache, cache       every cached LLM call (reruns cost nothing)
  workspace: kg_store.json, latest_chunks.json, uploads/ (the PDFs), benchmark/benchmark.jsonl,
             benchmark/runs/latest/{intercepts, judgements, judgements_archive, crosscheck, features.jsonl, usage.json},
             model.pkl, production_model_metrics.json, labelled_*, cv_/test_predictions.csv, *_analysis.json,
             brain_v3_backup/ and brain_v4_haikulabels/ (used for the before/after comparison in the paper)
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WS = os.path.join(ROOT, "final_model", "workspaces", "kg_legacy")

CANDIDATES = [
    ("mlruns", "MLflow experiment logs of the v1/v2 prototypes (no longer used)"),
    ("mlflow.db", "MLflow database of the v1/v2 prototypes"),
    ("logs", "old pipeline log files"),
    ("Artifacts/[0-9]*_[0-9]*", "per-run artifact folders of the v1/v2 prototypes (October)"),
    ("final_model/legacy_v1", "the v1 brain trained on synthetic data (its numbers are invalid)"),
    ("data/benchmark_100_results.csv", "v1 results: labels were category priors, not judgements (invalid)"),
    ("data/live_telemetry_claims.csv", "v1 telemetry"),
    ("data/uploads", "old upload folder; the PDFs live in the workspace's uploads/"),
    ("final_model/workspaces/kg_legacy/artifacts", "per-retrain copies of the trainer output (production files are kept)"),
    ("final_model/workspaces/kg_legacy/benchmark/runs/latest/artifacts", "per-question debug output of the feature stage"),
    ("final_model/workspaces/kg_legacy/benchmark/runs/latest/payloads", "raw interception payload captures"),
    ("final_model/workspaces/kg_legacy/labelled_features_backup_*.csv", "automatic backups of the training table"),
    ("final_model/workspaces/kg_legacy/brain_v4a_backup", "an intermediate brain superseded by the final one"),
    ("final_model/workspaces/kg_legacy/jobs/*", "logs of finished background jobs"),
    ("Artifacts/script_cache/batches/*.jsonl", "copies of submitted OpenAI batch files"),
    ("**/__pycache__", "Python bytecode caches"),
]
OPTIONAL = [
    ("final_model/workspaces/kg_legacy/benchmark/runs/latest__gpt-5.4-mini",
     "the earlier partial GPT-5.4-mini run (239 answers); keep it only for a future cross-model study"),
]


def size(p: str) -> int:
    if os.path.isfile(p):
        return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="delete (default: dry run)")
    ap.add_argument("--include-optional", action="store_true", help="also delete the optional items")
    a = ap.parse_args()
    os.chdir(ROOT)
    items = CANDIDATES + (OPTIONAL if a.include_optional else [])
    total = 0
    for pattern, why in items:
        for p in sorted(glob.glob(pattern, recursive=True)):
            if "venv" in p.split(os.sep):
                continue
            n = size(p)
            total += n
            print(f"{'DELETE' if a.apply else 'would delete'}  {n / 1e6:7.2f} MB  {p}   ({why})")
            if a.apply:
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
    print(f"\n{'Freed' if a.apply else 'Would free'} {total / 1e6:.1f} MB."
          + ("" if a.apply else "  Run with --apply to delete."))
    if not a.include_optional:
        for pattern, why in OPTIONAL:
            for p in glob.glob(pattern):
                print(f"(optional, kept) {size(p) / 1e6:.1f} MB  {p}  - {why}; add --include-optional to remove")


if __name__ == "__main__":
    main()

"""Re-run the Phase 4 tournament on <workspace>/labelled_features.csv (champion/challenger promotion)."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(ROOT)

from kg_audit.utils.api_errors import friendly  # noqa: E402
from kg_audit.pipelines.evaluation_pipeline import EvaluationPipeline  # noqa: E402
from kg_audit.utils import workspace as ws  # noqa: E402
from kg_audit.utils.jobs import JobStatus  # noqa: E402
from kg_audit.utils.main_utils import limit_cpu_threads  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--cpu-threads", type=int, default=2)
    args = ap.parse_args()
    os.environ["EPISTEME_CPU_THREADS"] = str(args.cpu_threads)
    limit_cpu_threads(args.cpu_threads)
    ws_id = args.workspace or ws.active_workspace_id()
    job = JobStatus(os.getenv("EPISTEME_JOB_DIR"))
    job.update("train", 0, 1, note="10-model nested-CV tournament")
    try:
        pipe = EvaluationPipeline(use_mlflow=False, workspace_id=ws_id)
        pipe.trainer_config.n_jobs = args.cpu_threads
        art = pipe.retrain()
        try:
            import export_paper_assets
            export_paper_assets.main(["--workspace", ws_id])
        except Exception as e:
            print(f"paper export skipped: {e}")
        job.finish("completed", result={"champion": art.best_model_name, "promoted": art.is_model_accepted,
                                        "test_auc": art.roc_auc_score, "test_log_loss": art.log_loss})
    except Exception as e:
        job.finish("failed", error=friendly(e))
        raise


if __name__ == "__main__":
    main()

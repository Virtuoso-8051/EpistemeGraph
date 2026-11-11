"""
Build (or extend) the knowledge graph of one corpus workspace.

Every file in <workspace>/uploads is ingested; documents already in the workspace registry are
skipped (content hash), so extending a corpus only processes the new PDFs. LLM extraction is cached
by chunk text, so even a fresh workspace over previously seen PDFs costs $0 in API calls.

  python scripts/build_kg.py                       # active workspace
  python scripts/build_kg.py --workspace <id> --rebuild   # scoped rebuild of THIS corpus only
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from kg_audit.utils.api_errors import friendly  # noqa: E402
from kg_audit.pipelines.evaluation_pipeline import EvaluationPipeline  # noqa: E402
from kg_audit.utils import workspace as ws  # noqa: E402
from kg_audit.utils.jobs import JobStatus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--rebuild", action="store_true", help="discard this corpus' graph and rebuild it from scratch")
    ap.add_argument("--sync-neo4j", action="store_true",
                    help="push the complete local graph of this corpus to Neo4j (e.g. after creating a new AuraDB instance); no LLM calls")
    args = ap.parse_args()

    ws_id = args.workspace or ws.active_workspace_id()
    ws_dir = ws.workspace_dir(ws_id)
    job = JobStatus(os.getenv("EPISTEME_JOB_DIR"))

    if args.sync_neo4j:
        from kg_audit.components.kg_builder import KGBuilder
        from kg_audit.entity.config_entity import KGAuditPipelineConfig, KGBuilderConfig
        from kg_audit.utils.main_utils import read_json
        store = read_json(os.path.join(ws_dir, "kg_store.json"), default=None)
        state = read_json(os.path.join(ws_dir, "kg_fidelity_state.json"), default={}) or {}
        if not store:
            job.finish("failed", error="no local graph to sync")
            sys.exit("No local graph in this workspace.")
        job.update("sync", 0, 1, note=f"{len(store['entities'])} entities, {len(store['claims'])} triples -> Neo4j")
        ok = KGBuilder(KGBuilderConfig(KGAuditPipelineConfig(workspace_id=ws_id))).push_to_neo4j_cloud(
            store["entities"], store["claims"], float(state.get("overall_kg_accuracy_score", 0.0)), clear_existing=True,
            fidelity_state=state)
        ws.update_meta(ws_id, neo4j_pushed=ok)
        (job.finish("completed", result={"neo4j_pushed": True}) if ok else job.finish("failed", error="Neo4j unreachable - check NEO4J_URI"))
        print("synced" if ok else "Neo4j sync FAILED")
        sys.exit(0 if ok else 1)
    files = sorted(glob.glob(os.path.join(ws_dir, "uploads", "*.pdf")) + glob.glob(os.path.join(ws_dir, "uploads", "*.txt")))
    if not files:
        job.finish("failed", error="no PDF/TXT files in this workspace")
        sys.exit("No files to ingest.")

    ws.update_meta(ws_id, status="building")
    job.update("prepare", 0, len(files), note=f"chunking {len(files)} document(s)")

    def on_progress(i: int, n: int, chunk_id: str) -> None:
        note = "verifying evidence quotes, resolving entities, MERGE into Neo4j" if i == n else chunk_id
        job.update("extract", i, n, note=note)

    try:
        pipe = EvaluationPipeline(use_mlflow=False, workspace_id=ws_id)
        art = pipe.run_phase1_kg_construction(files, reset=args.rebuild, progress_callback=on_progress)
        ws.update_meta(ws_id, status="ready", documents=sorted(os.path.basename(f) for f in files),
                       s_kg=art.overall_kg_accuracy_score, neo4j_pushed=art.neo4j_cloud_pushed)
        job.finish("completed", result={
            "S_KG": art.overall_kg_accuracy_score, "triples": art.verified_triplets_pushed,
            "entities": art.resolved_entity_count, "new_documents": art.new_documents,
            "skipped_documents": art.skipped_documents, "neo4j_pushed": art.neo4j_cloud_pushed})
    except Exception as e:
        ws.update_meta(ws_id, status="failed")
        job.finish("failed", error=friendly(e))
        raise


if __name__ == "__main__":
    main()

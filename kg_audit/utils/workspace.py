"""
Corpus workspaces.

Every uploaded set of PDFs becomes a *workspace*: an isolated, versioned corpus with its own
knowledge graph (namespaced in Neo4j by `kg_id`), fidelity state, benchmark, calibrated brain,
batch runs and audit history. Uploading a new set starts a FRESH graph from scratch; earlier
graphs are never overwritten, and the website re-opens on the last active workspace with all
its cached results.

  final_model/
    active_workspace.json            {"id": "<workspace id>"}
    workspaces/<id>/
      meta.json                      name, kg_id, created_at, documents, status
      uploads/                       the PDFs of this corpus
      kg_store.json  corpus_registry.json  latest_chunks.json  kg_fidelity_state.json
      rejected_claims.json  kg_precision_audit.json
      model.pkl  production_model_metrics.json  test_predictions.csv
      labelled_features.csv  live_telemetry.csv  query_history.jsonl
      benchmark/benchmark.jsonl  benchmark/runs/latest/...
      batches/<batch id>/...   jobs/<job id>/...   artifacts/   cache/
"""
from __future__ import annotations

import os
import random
import shutil
import string
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from kg_audit.utils.main_utils import read_json, write_json_atomic

MODEL_ROOT = "final_model"
WS_ROOT = os.path.join(MODEL_ROOT, "workspaces")
POINTER = os.path.join(MODEL_ROOT, "active_workspace.json")
LEGACY_ID = "kg_legacy"
LEGACY_KG_ID = "legacy"           # Neo4j nodes created before namespacing get kg_id = 'legacy'
_STATE_FILES = ("kg_store.json", "corpus_registry.json", "latest_chunks.json",
                "kg_fidelity_state.json", "rejected_claims.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def workspace_dir(ws_id: Optional[str] = None) -> str:
    return os.path.join(WS_ROOT, ws_id or active_workspace_id())


def meta(ws_id: Optional[str] = None) -> Dict[str, Any]:
    return read_json(os.path.join(workspace_dir(ws_id), "meta.json"), default={}) or {}


def update_meta(ws_id: str, **fields) -> Dict[str, Any]:
    m = meta(ws_id)
    m.update(fields)
    m["updated_at"] = _now()
    write_json_atomic(os.path.join(workspace_dir(ws_id), "meta.json"), m)
    return m


def list_workspaces() -> List[Dict[str, Any]]:
    ensure_workspace()
    out = []
    for d in sorted(os.listdir(WS_ROOT)) if os.path.isdir(WS_ROOT) else []:
        m = read_json(os.path.join(WS_ROOT, d, "meta.json"), default=None)
        if m:
            st = read_json(os.path.join(WS_ROOT, d, "kg_fidelity_state.json"), default={}) or {}
            m["s_kg"] = st.get("overall_kg_accuracy_score")
            m["triples"] = st.get("verified_triplets_pushed")
            m["has_brain"] = os.path.exists(os.path.join(WS_ROOT, d, "model.pkl"))
            out.append(m)
    return sorted(out, key=lambda m: m.get("created_at", ""), reverse=True)


def create_workspace(name: str = "", kg_id: Optional[str] = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ws_id = f"kg_{stamp}_{''.join(random.choices(string.ascii_lowercase + string.digits, k=4))}"
    d = os.path.join(WS_ROOT, ws_id)
    os.makedirs(os.path.join(d, "uploads"), exist_ok=True)
    write_json_atomic(os.path.join(d, "meta.json"), {
        "id": ws_id, "kg_id": kg_id or ws_id, "name": name or f"Corpus {stamp}",
        "created_at": _now(), "updated_at": _now(), "status": "empty", "documents": [],
    })
    return ws_id


def set_active(ws_id: str) -> None:
    if not os.path.exists(os.path.join(WS_ROOT, ws_id, "meta.json")):
        raise ValueError(f"Unknown workspace {ws_id}")
    write_json_atomic(POINTER, {"id": ws_id, "activated_at": _now()})


def active_workspace_id() -> str:
    env = os.getenv("EPISTEME_WORKSPACE")
    if env:
        return env
    ensure_workspace()
    p = read_json(POINTER, default={}) or {}
    if p.get("id") and os.path.exists(os.path.join(WS_ROOT, p["id"], "meta.json")):
        return p["id"]
    ws = sorted(os.listdir(WS_ROOT)) if os.path.isdir(WS_ROOT) else []
    if ws:
        set_active(ws[-1])
        return ws[-1]
    ws_id = create_workspace("Empty corpus")
    set_active(ws_id)
    return ws_id


def kg_id(ws_id: Optional[str] = None) -> str:
    return meta(ws_id).get("kg_id") or (ws_id or active_workspace_id())


def ensure_workspace() -> None:
    """One-off migration of the pre-workspace layout (final_model/*.json) into workspaces/kg_legacy."""
    if os.path.isdir(WS_ROOT) and any(os.path.exists(os.path.join(WS_ROOT, d, "meta.json")) for d in os.listdir(WS_ROOT)):
        return
    os.makedirs(WS_ROOT, exist_ok=True)
    if not os.path.exists(os.path.join(MODEL_ROOT, "kg_store.json")):
        return
    d = os.path.join(WS_ROOT, LEGACY_ID)
    os.makedirs(os.path.join(d, "uploads"), exist_ok=True)
    for f in _STATE_FILES:
        src = os.path.join(MODEL_ROOT, f)
        if os.path.exists(src):
            shutil.move(src, os.path.join(d, f))
    registry = read_json(os.path.join(d, "corpus_registry.json"), default={}) or {}
    docs = []
    for v in registry.values():
        src = os.path.join("data", "uploads", v["source_name"])
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(d, "uploads", v["source_name"]))
        docs.append(v["source_name"])
    # the v1 brain was trained on synthetic data with another feature schema: archive, never load
    legacy_dir = os.path.join(MODEL_ROOT, "legacy_v1")
    for f in ("model.pkl", "production_model_metrics.json"):
        src = os.path.join(MODEL_ROOT, f)
        if os.path.exists(src):
            os.makedirs(legacy_dir, exist_ok=True)
            shutil.move(src, os.path.join(legacy_dir, f))
    write_json_atomic(os.path.join(d, "meta.json"), {
        "id": LEGACY_ID, "kg_id": LEGACY_KG_ID, "name": "LINE · node2vec · DeepWalk",
        "created_at": "2026-10-25T05:12:08+00:00", "updated_at": _now(), "status": "ready",
        "documents": sorted(docs), "migrated_from": "final_model/ (pre-workspace layout)",
    })
    write_json_atomic(POINTER, {"id": LEGACY_ID, "activated_at": _now()})

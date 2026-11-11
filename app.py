"""
EpistemeGraph web server: JSON API + the interactive website (web/).

    python app.py                 ->  http://localhost:8000
    uvicorn app:app --port 8000

Long-running work (KG builds, query batches, benchmark generation / evaluation, retraining, KG
precision audits) runs as separate low-priority processes with a status heartbeat, so the site stays
responsive, survives reloads, and always re-opens the last cached results of the active workspace.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import threading
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from kg_audit.components.decision import FEATURE_SCHEMA_VERSION, FEATURES  # noqa: E402
from kg_audit.utils import jobs as J  # noqa: E402
from kg_audit.utils import llm as _llm  # noqa: E402
from kg_audit.utils import workspace as ws  # noqa: E402
from kg_audit.utils.main_utils import limit_cpu_threads, read_json, read_jsonl  # noqa: E402


def _json_safe(o: Any) -> Any:
    """NaN / inf (e.g. a coverage on an empty class, an AUROC on a one-class split) -> null: valid JSON."""
    import math
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


class SafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        import json
        return json.dumps(_json_safe(content), ensure_ascii=False, allow_nan=False, separators=(",", ":"), default=str).encode("utf-8")


limit_cpu_threads()
app = FastAPI(title="EpistemeGraph", version="0.3.0", default_response_class=SafeJSONResponse,
              description="KG-grounded hallucination auditing with topological evidence and conformal guarantees")

_PIPES: Dict[str, Any] = {}
_PIPE_LOCK = threading.Lock()
_AUDIT_LOCK = threading.Lock()

# ------------------------------------------------------------------------------------------ public-mode guard
# EPISTEME_PUBLIC_MODE=1 (set it when the site is reachable from the internet):
#   * write / expensive endpoints need header  X-Admin-Token: <EPISTEME_ADMIN_TOKEN>
#   * single audits are open but rate-limited per IP (EPISTEME_RATE_LIMIT per hour), unless the visitor
#     brings their own OpenAI key in header X-OpenAI-Key (used for that request only, never stored)
PUBLIC = os.getenv("EPISTEME_PUBLIC_MODE", "0") == "1"
ADMIN_TOKEN = os.getenv("EPISTEME_ADMIN_TOKEN", "")
RATE = int(os.getenv("EPISTEME_RATE_LIMIT", "20"))
_HITS: Dict[str, List[float]] = {}


def is_admin(request: Request) -> bool:
    return (not PUBLIC) or (bool(ADMIN_TOKEN) and request.headers.get("x-admin-token", "") == ADMIN_TOKEN)


def require_admin(request: Request) -> None:
    if not is_admin(request):
        raise HTTPException(403, "This action needs the admin token (public mode).")


def _byok(request: Request) -> str:
    """Visitor's own key (Claude 'sk-ant-...' or OpenAI), sent per request; never stored server-side."""
    return (request.headers.get("x-llm-key") or request.headers.get("x-openai-key") or "").strip()


def rate_limit(request: Request) -> None:
    if is_admin(request) or _byok(request):
        return
    import time as _t
    ip = request.client.host if request.client else "?"
    now = _t.time()
    hits = [h for h in _HITS.get(ip, []) if now - h < 3600]
    if len(hits) >= RATE:
        raise HTTPException(429, f"Public demo limit reached ({RATE} audits/hour). Add your own OpenAI key in Settings.")
    _HITS[ip] = hits + [now]


# ------------------------------------------------------------------------------------------ helpers
def W(ws_id: Optional[str] = None) -> str:
    return ws.workspace_dir(ws_id or ws.active_workspace_id())


def pipeline(ws_id: Optional[str] = None):
    from kg_audit.pipelines.evaluation_pipeline import EvaluationPipeline
    ws_id = ws_id or ws.active_workspace_id()
    with _PIPE_LOCK:
        if ws_id not in _PIPES:
            _PIPES[ws_id] = EvaluationPipeline(use_mlflow=False, workspace_id=ws_id)
        return _PIPES[ws_id]


@app.on_event("startup")
def _default_corpus() -> None:
    """Every server start opens the reference corpus (the 3-paper study), whatever was uploaded last session."""
    d = os.getenv("EPISTEME_DEFAULT_WORKSPACE", "kg_legacy")
    try:
        if d and os.path.exists(os.path.join(ws.WS_ROOT, d, "meta.json")):
            ws.set_active(d)
    except Exception as e:
        print(f"[startup] default corpus not set: {e}", flush=True)


@app.on_event("startup")
def _warm_up() -> None:
    """Load the heavy ML stack (embedder, NLI, pipeline) in the background AFTER the page is served, so the
    website opens instantly and the first audit does not pay ~30 s of model loading."""
    def work():
        import time as _t
        _t.sleep(4)
        try:
            kg_graph()
            pipeline()
            from kg_audit.utils.main_utils import embed
            embed("warm up")
        except Exception as e:  # warm-up is best effort; real requests report real errors
            print(f"[warm-up] skipped: {e}", flush=True)
    threading.Thread(target=work, daemon=True, name="warm-up").start()


def safe_name(name: str) -> str:
    base = os.path.basename(name or "file")
    return re.sub(r"[^\w.\- ]", "_", base)[:120] or "file"


def running(ws_dir: str, job_type: str) -> Optional[Dict[str, Any]]:
    for j in J.list_jobs(ws_dir, job_type):
        if j.get("state") in ("queued", "running"):
            return j
    return None


def start_job(job_type: str, argv: List[str], params: Dict[str, Any], ws_id: Optional[str] = None) -> Dict[str, Any]:
    ws_id = ws_id or ws.active_workspace_id()
    d = W(ws_id)
    busy = running(d, job_type)
    if busy:
        raise HTTPException(409, f"A {job_type} job is already running ({busy['id']}).")
    job_id = J.new_job(d, job_type, params)
    J.launch(d, job_id, argv + ["--workspace", ws_id], ROOT)
    return J.get_job(d, job_id)


def brain_summary(ws_id: str) -> Dict[str, Any]:
    d = W(ws_id)
    prod = read_json(os.path.join(d, "production_model_metrics.json"), default={}) or {}
    if prod.get("schema_version") == FEATURE_SCHEMA_VERSION and os.path.exists(os.path.join(d, "model.pkl")):
        src, transferred = ws_id, False
    else:
        src, transferred, prod = None, False, {}
        for w in ws.list_workspaces():
            if w["id"] != ws_id and w.get("has_brain"):
                p = read_json(os.path.join(W(w["id"]), "production_model_metrics.json"), default={}) or {}
                if p.get("schema_version") == FEATURE_SCHEMA_VERSION:
                    src, transferred, prod = w["id"], True, p
                    break
    if not prod:
        return {"status": "uncalibrated"}
    t = prod.get("test_metrics", {})
    cb = (prod.get("decision_bundle") or {}).get("conformal", {})
    return {"status": "transferred" if transferred else "calibrated", "source_workspace": src,
            "champion": prod.get("winning_model"), "n_labelled": prod.get("n_labelled"),
            "test_auc": (t.get("S_final") or {}).get("roc_auc"), "test_ece": (t.get("S_final") or {}).get("ece"),
            "llm_auc": ((t.get("baselines") or {}).get("llm_self_confidence") or {}).get("roc_auc"),
            "coverage": (t.get("conformal") or {}).get("coverage"), "alpha": cb.get("alpha"),
            "updated_at": prod.get("updated_at")}


def history(ws_id: str) -> List[Dict[str, Any]]:
    return read_jsonl(os.path.join(W(ws_id), "query_history.jsonl"))


def slim_history(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {k: rec.get(k) for k in ("id", "timestamp", "query", "decision", "hallucination_probability",
                                    "hallucination_extent", "llm_confidence", "S_final", "mode")}


def latest_batch(ws_id: str) -> Optional[str]:
    p = read_json(os.path.join(W(ws_id), "batches", "latest.json"), default={}) or {}
    if p.get("batch_id"):
        return p["batch_id"]
    jobs = J.list_jobs(W(ws_id), "batch")
    return (jobs[0].get("params") or {}).get("batch_id") if jobs else None


# ------------------------------------------------------------------------------------------ state
@app.get("/api/state")
def state(request: Request):
    ws_id = ws.active_workspace_id()
    d = W(ws_id)
    kg = read_json(os.path.join(d, "kg_fidelity_state.json"), default={}) or {}
    reg = read_json(os.path.join(d, "corpus_registry.json"), default={}) or {}
    kga = read_json(os.path.join(d, "kg_precision_audit.json"), default={}) or {}
    hist = history(ws_id)
    bench_summary = read_json(os.path.join(d, "benchmark", "summary.json"), default=None)
    bench_items = read_jsonl(os.path.join(d, "benchmark", "benchmark.jsonl"))
    b_id = latest_batch(ws_id)
    active_jobs = [j for j in J.list_jobs(d) if j.get("state") in ("queued", "running")]
    return {
        "workspace": ws.meta(ws_id), "workspaces": ws.list_workspaces(),
        "kg": {k: kg.get(k) for k in ("overall_kg_accuracy_score", "provenance_grounding_score", "semantic_coverage_score",
                                       "topological_coherence_score", "internal_consistency_score", "documents_in_corpus",
                                       "total_chunks_processed", "raw_triplets_extracted", "verified_triplets_pushed",
                                       "resolved_entity_count", "cross_pdf_shared_entities_count", "cross_pdf_shared_entities",
                                       "formula", "timestamp")} if kg else None,
        "kg_diagnostics": kg.get("diagnostics"),
        "documents": [{"name": v["source_name"], "chunks": v.get("chunks"), "ingested_at": v.get("ingested_at"), "hash": k[:12]}
                      for k, v in reg.items()],
        "kg_precision": {"auditor": kga.get("auditor_model"), "human": bool(kga.get("human")),
                         "overall": ((kga.get("human") or kga.get("llm") or {}).get("overall")),
                         "model_assisted": ((kga.get("human") or kga.get("llm") or {}).get("model_assisted"))} if kga else None,
        "sensitivity": read_json(os.path.join(d, "sensitivity_phase1.json"), default=None),
        "brain": brain_summary(ws_id),
        "history": [slim_history(h) for h in reversed(hist[-60:])],
        "history_count": len(hist),
        "latest_audit": hist[-1] if hist else None,
        "batch": {"id": b_id, "summary": read_json(os.path.join(d, "batches", b_id or "_", "summary.json"), default=None)} if b_id else None,
        "benchmark": {"n_items": len(bench_items), "by_category": dict(Counter(i.get("category") for i in bench_items)),
                      "summary": bench_summary},
        "active_jobs": active_jobs,
        "is_admin": is_admin(request),
        "features": FEATURES,
        "access": {"public_mode": PUBLIC, "rate_limit_per_hour": RATE},
        "system": {"llm_model": _llm.model_for("answer"), "judge_model": _llm.model_for("judge"),
                   "provider": _llm.provider_of(_llm.model_for("answer")),
                   "api_key": _llm.key_status()[_llm.provider_of(_llm.model_for("answer"))],
                   "openai_key": _llm.key_status()[_llm.provider_of(_llm.model_for("answer"))],
                   "spend_usd": round(_llm.ledger_total(), 3), "budget_usd": _llm.budget_usd(),
                   "neo4j": bool(os.getenv("NEO4J_URI")), "alpha": float(os.getenv("EPISTEME_ALPHA", "0.10")),
                   "server_time": datetime.now().isoformat()},
    }


@app.get("/api/workspaces")
def workspaces():
    return {"active": ws.active_workspace_id(), "workspaces": ws.list_workspaces()}


PROTECTED = {w.strip() for w in os.getenv("EPISTEME_PROTECTED_WORKSPACES", "kg_legacy").split(",") if w.strip()}


def guard_protected(ws_id: Optional[str] = None, action: str = "change") -> None:
    """The reference corpus (the 3-paper study) cannot be modified from the website: its graph, benchmark and
    calibrated brain are the published results. Upload a NEW corpus instead (terminal scripts are not restricted)."""
    ws_id = ws_id or ws.active_workspace_id()
    if ws_id in PROTECTED:
        raise HTTPException(423, f"'{ws_id}' is the protected reference corpus of the study, so the website cannot {action} it. "
                                 "Upload your PDFs as a NEW corpus (it gets its own graph and reuses the trained brain), "
                                 "or switch to another corpus first.")


@app.post("/api/workspaces/{ws_id}/activate", dependencies=[Depends(require_admin)])
def activate(ws_id: str):
    try:
        ws.set_active(ws_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"active": ws_id}


# ------------------------------------------------------------------------------------------ knowledge graph
@app.post("/api/kg/upload", dependencies=[Depends(require_admin)])
async def kg_upload(files: List[UploadFile] = File(...), mode: str = Form("new"), name: str = Form("")):
    """mode=new: a FRESH corpus/graph from these files (previous graphs untouched); mode=append: extend the active one."""
    files = [f for f in files if (f.filename or "").lower().endswith((".pdf", ".txt"))]
    if not files:
        raise HTTPException(400, "Upload at least one .pdf or .txt file.")
    if mode == "new":
        ws_id = ws.create_workspace(name.strip() or ", ".join(safe_name(f.filename).rsplit(".", 1)[0] for f in files)[:80])
    else:
        ws_id = ws.active_workspace_id()
        guard_protected(ws_id, "add documents to")
        if running(W(ws_id), "build_kg"):
            raise HTTPException(409, "A build is already running for this corpus.")
    up = os.path.join(W(ws_id), "uploads")
    os.makedirs(up, exist_ok=True)
    saved = []
    for f in files:
        dest = os.path.join(up, safe_name(f.filename))
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(os.path.basename(dest))
    ws.update_meta(ws_id, status="queued", pending_documents=saved)
    ws.set_active(ws_id)
    job = start_job("build_kg", ["scripts/build_kg.py"], {"mode": mode, "files": saved}, ws_id)
    return {"workspace": ws_id, "job": job}


@app.post("/api/kg/rebuild", dependencies=[Depends(require_admin)])
def kg_rebuild():
    guard_protected(action="rebuild")
    return {"job": start_job("build_kg", ["scripts/build_kg.py", "--rebuild"], {"mode": "rebuild"})}


@app.post("/api/kg/sync-neo4j", dependencies=[Depends(require_admin)])
def kg_sync():
    """Push the whole local graph of the active corpus to Neo4j (e.g. after creating a new AuraDB instance)."""
    return {"job": start_job("build_kg", ["scripts/build_kg.py", "--sync-neo4j"], {"mode": "sync"})}


_GRAPH_CACHE: Dict[Any, Any] = {}


@app.get("/api/kg/graph")
def kg_graph(limit: int = 1500):
    path = os.path.join(W(), "kg_store.json")
    try:
        key = (path, os.path.getmtime(path), limit)
    except OSError:
        return {"nodes": [], "links": []}
    if key in _GRAPH_CACHE:
        return _GRAPH_CACHE[key]
    from kg_audit.components.kg_builder import KGBuilder
    store = read_json(path, default=None)
    if not store:
        return {"nodes": [], "links": []}
    ents = store.get("entities", {})
    rows = KGBuilder._aggregate_edges(store.get("claims", []))
    deg = Counter()
    for r in rows:
        deg[r["subj"]] += 1
        deg[r["obj"]] += 1
    keep = {n for n, _ in deg.most_common(limit)}
    nodes = [{"id": n, "type": ents.get(n, {}).get("type", "CONCEPT"), "degree": deg[n],
              "aliases": [a for a in ents.get(n, {}).get("aliases", []) if a != n][:10],
              "pdfs": ents.get(n, {}).get("source_pdfs", [])} for n in keep]
    links = [{"source": r["subj"], "target": r["obj"], "rel": r["rel"], "grounding": round(r["grounding"], 3),
              "quote": r["best_quote"][:400], "pdfs": r["sources"], "support": len(r["chunk_ids"])}
             for r in rows if r["subj"] in keep and r["obj"] in keep]
    _GRAPH_CACHE.clear()
    _GRAPH_CACHE[key] = {"nodes": nodes, "links": links}
    return _GRAPH_CACHE[key]


@app.get("/api/kg/precision")
def kg_precision():
    return read_json(os.path.join(W(), "kg_precision_audit.json"), default={}) or {}


class PrecisionReq(BaseModel):
    n: int = 100


@app.post("/api/kg/precision-audit", dependencies=[Depends(require_admin)])
def kg_precision_audit(req: PrecisionReq):
    guard_protected(action="re-audit")
    return {"job": start_job("kg_precision", ["scripts/audit_kg_precision.py", "--n", str(max(10, min(1000, req.n)))], req.model_dump())}


# ------------------------------------------------------------------------------------------ single audit
class AuditReq(BaseModel):
    query: str
    mode: str = "closed_book"


@app.post("/api/audit", dependencies=[Depends(rate_limit)])
def audit(req: AuditReq, request: Request):
    q = (req.query or "").strip()[:2000]
    if not q:
        raise HTTPException(400, "Empty query.")
    byok = _byok(request)
    need = _llm.provider_of(_llm.model_for("answer"))
    if not byok and not _llm.key_status()[need]:
        raise HTTPException(402, ("ANTHROPIC_API_KEY is empty: paste your Claude API key into the .env file "
                                  "(line ANTHROPIC_API_KEY=) and restart the website.") if need == "anthropic"
                            else "OPENAI_API_KEY is not set in .env")
    mode = "grounded" if req.mode == "grounded" else "closed_book"
    try:
        with _AUDIT_LOCK:
            if byok:  # visitor's own key: a throw-away interceptor, the key is never written anywhere
                from kg_audit.components.llm_interceptor import LLMInterceptor
                from kg_audit.pipelines.evaluation_pipeline import EvaluationPipeline
                pipe = EvaluationPipeline(use_mlflow=False, workspace_id=ws.active_workspace_id())
                pipe._interceptor = LLMInterceptor(pipe.interceptor_config, client=_llm.LLMClient(api_key=byok, timeout=90))
                pipe.run_pipeline(q, mode=mode)
            else:
                pipeline().run_pipeline(q, mode=mode)
    except Exception as e:
        from kg_audit.utils.api_errors import classify as _classify, friendly as _friendly
        code = 402 if _classify(e) == "fatal" else 500     # 402: billing / key problem the user must fix
        msg = _friendly(e)
        raise HTTPException(code, msg.replace(byok, "***") if byok else msg)
    hist = history(ws.active_workspace_id())
    return hist[-1]


@app.get("/api/history")
def get_history(limit: int = 100):
    hist = history(ws.active_workspace_id())
    return [slim_history(h) for h in reversed(hist[-limit:])]


@app.get("/api/history/{rec_id}")
def get_history_item(rec_id: str):
    for h in reversed(history(ws.active_workspace_id())):
        if h.get("id") == rec_id:
            return h
    raise HTTPException(404, "not found")


# ------------------------------------------------------------------------------------------ batches (user .txt)
@app.post("/api/batch", dependencies=[Depends(require_admin)])
async def batch(file: Optional[UploadFile] = File(None), queries: str = Form(""), mode: str = Form("closed_book"),
                workers: int = Form(6)):
    ws_id = ws.active_workspace_id()
    batch_id = f"batch_{datetime.now():%Y%m%d_%H%M%S}"
    bdir = os.path.join(W(ws_id), "batches", batch_id)
    os.makedirs(bdir, exist_ok=True)
    src = os.path.join(bdir, "upload.txt")
    if file is not None and file.filename:
        ext = ".jsonl" if file.filename.lower().endswith(".jsonl") else ".txt"
        src = os.path.join(bdir, "upload" + ext)
        with open(src, "wb") as out:
            shutil.copyfileobj(file.file, out)
    elif queries.strip():
        with open(src, "w", encoding="utf-8") as out:
            out.write(queries.strip() + "\n")
    else:
        raise HTTPException(400, "Provide a .txt file or paste queries (one per line).")
    job = start_job("batch", ["scripts/run_batch.py", "--file", src, "--batch-id", batch_id, "--workers", str(workers),
                              "--mode", "grounded" if mode == "grounded" else "closed_book"], {"batch_id": batch_id}, ws_id)
    return {"batch_id": batch_id, "job": job}


def _batch_payload(ws_id: str, batch_id: str) -> Dict[str, Any]:
    bdir = os.path.join(W(ws_id), "batches", batch_id)
    rows = read_jsonl(os.path.join(bdir, "features.jsonl"))
    jobs = [j for j in J.list_jobs(W(ws_id), "batch") if (j.get("params") or {}).get("batch_id") == batch_id]
    return {"batch_id": batch_id, "config": read_json(os.path.join(bdir, "config.json"), default={}),
            "summary": read_json(os.path.join(bdir, "summary.json"), default=None), "rows": rows,
            "job": J.get_job(W(ws_id), jobs[0]["id"]) if jobs else None}


@app.get("/api/batch/latest")
def batch_latest():
    ws_id = ws.active_workspace_id()
    b = latest_batch(ws_id)
    return _batch_payload(ws_id, b) if b else {"batch_id": None, "rows": []}


@app.get("/api/batch/{batch_id}")
def batch_get(batch_id: str):
    return _batch_payload(ws.active_workspace_id(), safe_name(batch_id))


@app.get("/api/batch/{batch_id}/csv")
def batch_csv(batch_id: str):
    bdir = os.path.join(W(), "batches", safe_name(batch_id))
    rows = read_jsonl(os.path.join(bdir, "features.jsonl"))
    if not rows:
        raise HTTPException(404, "no results yet")
    p = os.path.join(bdir, "results_export.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    return FileResponse(p, filename=f"{batch_id}_results.csv", media_type="text/csv")


@app.get("/api/batches")
def batches():
    d = os.path.join(W(), "batches")
    out = []
    for b in sorted(os.listdir(d), reverse=True) if os.path.isdir(d) else []:
        if b.startswith("batch_"):
            out.append({"batch_id": b, "summary": read_json(os.path.join(d, b, "summary.json"), default=None),
                        "config": read_json(os.path.join(d, b, "config.json"), default=None)})
    return out


# ------------------------------------------------------------------------------------------ research benchmark
class GenReq(BaseModel):
    n: int = 500
    workers: int = 6
    batch_api: bool = False


@app.post("/api/benchmark/generate", dependencies=[Depends(require_admin)])
def bench_generate(req: GenReq):
    guard_protected(action="replace the benchmark of")
    argv = ["scripts/generate_benchmark.py", "--n", str(max(20, min(5000, req.n))), "--workers", str(req.workers)]
    return {"job": start_job("bench_gen", argv + (["--use-batch-api"] if req.batch_api else []), req.model_dump())}


@app.post("/api/benchmark/upload", dependencies=[Depends(require_admin)])
async def bench_upload(file: UploadFile = File(...)):
    guard_protected(action="replace the benchmark of")
    """Use your own curated question set (.jsonl from the generator, or .txt one question per line)."""
    bdir = os.path.join(W(), "benchmark")
    os.makedirs(bdir, exist_ok=True)
    ext = ".jsonl" if (file.filename or "").lower().endswith(".jsonl") else ".txt"
    dest = os.path.join(bdir, "benchmark" + ext)
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    if ext == ".txt":  # normalise to jsonl so the rest of the pipeline sees one format
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from run_benchmark import load_items
        import json
        items = load_items(dest)
        with open(os.path.join(bdir, "benchmark.jsonl"), "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return {"n_items": len(read_jsonl(os.path.join(bdir, "benchmark.jsonl")))}


class BenchReq(BaseModel):
    limit: int = 0
    workers: int = 6
    cpu_threads: int = 2
    mode: str = "closed_book"
    judge_model: str = ""
    batch_api: bool = True
    crosscheck_n: int = 100
    crosscheck_model: str = "claude-sonnet-5-5"


@app.post("/api/benchmark/run", dependencies=[Depends(require_admin)])
def bench_run(req: BenchReq):
    guard_protected(action="re-run the benchmark of")
    if not read_jsonl(os.path.join(W(), "benchmark", "benchmark.jsonl")):
        raise HTTPException(400, "No benchmark in this workspace: generate or upload one first.")
    argv = ["scripts/run_benchmark.py", "--workers", str(req.workers), "--cpu-threads", str(req.cpu_threads),
            "--mode", "grounded" if req.mode == "grounded" else "closed_book"]
    if req.limit:
        argv += ["--limit", str(req.limit)]
    if req.judge_model.strip():
        argv += ["--judge-model", req.judge_model.strip()]
    argv += ["--use-batch-api"] if req.batch_api else ["--no-batch"]
    argv += ["--crosscheck-n", str(max(0, min(1000, req.crosscheck_n))), "--crosscheck-model", req.crosscheck_model.strip() or "claude-sonnet-5-5"]
    return {"job": start_job("benchmark", argv, req.model_dump())}


@app.get("/api/benchmark")
def bench_get():
    d = os.path.join(W(), "benchmark")
    items = read_jsonl(os.path.join(d, "benchmark.jsonl"))
    jobs = J.list_jobs(W(), "benchmark")
    run_progress = read_json(os.path.join(d, "runs", "latest", "progress.json"), default=None)
    # A corpus without its own study (e.g. a freshly uploaded one) shows the reference study's results, labelled as such.
    res_ws, reference = W(), None
    if not os.path.exists(os.path.join(d, "results.csv")):
        ref_id = os.getenv("EPISTEME_DEFAULT_WORKSPACE", "kg_legacy")
        if ref_id != ws.active_workspace_id() and os.path.exists(os.path.join(W(ref_id), "benchmark", "results.csv")):
            res_ws = W(ref_id)
            meta = read_json(os.path.join(res_ws, "meta.json"), default={}) or {}
            reference = {"id": ref_id, "name": meta.get("name", ref_id)}
    d = os.path.join(res_ws, "benchmark")
    res = pd.read_csv(os.path.join(d, "results.csv")) if os.path.exists(os.path.join(d, "results.csv")) else None
    rows = []
    if res is not None:
        keep = [c for c in ["id", "category", "judge_verdict", "label", "decision", "S_final", "hallucination_probability",
                            "hallucination_extent", "c_llm", "split", "question", "oof_S", "oof_decision", "match"] + [f["key"] for f in FEATURES] if c in res]
        keep = list(dict.fromkeys(keep))
        rows = res[keep].where(pd.notna(res[keep]), None).to_dict("records")
    tp = os.path.join(res_ws, "test_predictions.csv")
    preds = pd.read_csv(tp).to_dict("records") if os.path.exists(tp) else []
    return {"items": [{k: it.get(k) for k in ("id", "category", "difficulty", "question")} for it in items],
            "reference": reference,
            "summary": read_json(os.path.join(d, "summary.json"), default=None), "rows": rows, "test_predictions": preds,
            "job": J.get_job(W(), jobs[0]["id"]) if jobs else None, "progress": run_progress,
            "crosscheck": read_json(os.path.join(d, "judge_crosscheck.json"), default=None),
            "matching": (read_json(os.path.join(d, "results_analysis.json"), default={}) or {}).get("matching")}


# ------------------------------------------------------------------------------------------ brain
@app.get("/api/brain")
def brain():
    ws_id = ws.active_workspace_id()
    s = brain_summary(ws_id)
    src = s.get("source_workspace")
    full = read_json(os.path.join(W(src), "production_model_metrics.json"), default={}) if src else {}
    full = {k: v for k, v in (full or {}).items() if k != "feature_schema"}
    return {"summary": s, "metrics": full}


@app.post("/api/brain/retrain", dependencies=[Depends(require_admin)])
def brain_retrain():
    guard_protected(action="retrain the brain of")
    return {"job": start_job("retrain", ["scripts/retrain.py"], {})}


# ------------------------------------------------------------------------------------------ jobs
@app.get("/api/jobs")
def jobs_list(job_type: Optional[str] = None):
    return J.list_jobs(W(), job_type)


@app.get("/api/jobs/{job_id}")
def job_get(job_id: str):
    j = J.get_job(W(), safe_name(job_id))
    if not j:
        raise HTTPException(404, "job not found")
    return j


@app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(require_admin)])
def job_cancel(job_id: str):
    return {"cancelled": J.cancel(W(), safe_name(job_id))}


# ------------------------------------------------------------------------------------------ paper
@app.post("/api/paper/export", dependencies=[Depends(require_admin)])
def paper_export():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import export_paper_assets
    export_paper_assets.main(["--workspace", ws.active_workspace_id()])
    return {"zip": "/api/paper/overleaf.zip"}


@app.get("/api/paper/overleaf.zip")
def paper_zip():
    p = os.path.join(ROOT, "paper", "EpistemeGraph_overleaf.zip")
    if not os.path.exists(p):
        paper_export()
    return FileResponse(p, filename="EpistemeGraph_overleaf.zip", media_type="application/zip")


# ------------------------------------------------------------------------------------------ website
app.mount("/", StaticFiles(directory=os.path.join(ROOT, "web"), html=True), name="web")

def _serve_locally() -> None:
    """Double-click friendly start: reuse a server that is already running, else find a free port, and open
    the browser only once the server is actually listening (no more endless 'loading')."""
    import socket
    import urllib.request
    import webbrowser
    port = int(os.getenv("PORT", "8000"))

    def busy(p: int) -> bool:
        with socket.socket() as sk:
            sk.settimeout(0.5)
            return sk.connect_ex(("127.0.0.1", p)) == 0

    if busy(port):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=4)
            print(f"EpistemeGraph is already running on http://localhost:{port} - opening it.")
            webbrowser.open(f"http://localhost:{port}")
            return
        except Exception:
            port = next((p for p in range(port + 1, port + 20) if not busy(p)), port)
            print(f"Port {os.getenv('PORT', '8000')} is used by another program - using {port} instead.")
    url = f"http://localhost:{port}"
    if os.getenv("EPISTEME_OPEN_BROWSER", "1") == "1":
        def opener():
            import time as _t
            for _ in range(120):
                if busy(port):
                    webbrowser.open(url)
                    return
                _t.sleep(0.5)
        threading.Thread(target=opener, daemon=True).start()
    print(f"EpistemeGraph -> {url}  (close this window to stop the server)")
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    _serve_locally()

"""
Phase 1 sensitivity analysis - evidence that the design constants do not drive the conclusions.

Every Phase 1 constant that is not a mathematical definition is perturbed one at a time around its
default, and the full post-extraction pipeline (claim filtering -> evidence-quote verification ->
entity resolution -> S_KG) is re-run on the CACHED LLM extractions. No API call is made, so the
analysis is free and exactly reproducible.

Reported per setting: S_KG, its four pillars, #verified triples, #entities, acceptance rate.
Output: <workspace>/sensitivity_phase1.json (+ printed table); exported to the paper as a table.

    python scripts/sensitivity_phase1.py
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import kg_audit.components.kg_builder as kb  # noqa: E402
from kg_audit.entity.config_entity import KGAuditPipelineConfig, KGBuilderConfig  # noqa: E402
from kg_audit.utils import workspace as ws  # noqa: E402
from kg_audit.utils.main_utils import limit_cpu_threads, read_json, write_json_atomic  # noqa: E402

DEFAULTS = {"QUOTE_ACCEPT": kb.QUOTE_ACCEPT, "ENTITY_SUPPORT_MIN": kb.ENTITY_SUPPORT_MIN, "COV_TAU": kb.COV_TAU,
            "MIN_EXTRACTION_CONFIDENCE": kb.MIN_EXTRACTION_CONFIDENCE, "entity_sim_threshold": 0.85}
GRID = {"QUOTE_ACCEPT": [0.75, 0.90], "ENTITY_SUPPORT_MIN": [0.75, 0.95], "COV_TAU": [0.50, 0.70],
        "MIN_EXTRACTION_CONFIDENCE": [0.30, 0.70], "entity_sim_threshold": [0.80, 0.90]}
MEANING = {"QUOTE_ACCEPT": "verbatim quote-match ratio", "ENTITY_SUPPORT_MIN": "entity support in the quote",
           "COV_TAU": "coverage normaliser tau", "MIN_EXTRACTION_CONFIDENCE": "min. extractor confidence",
           "entity_sim_threshold": "entity-merge cosine"}


def run_once(builder: kb.KGBuilder, chunks, settings):
    for k, v in settings.items():
        if k == "entity_sim_threshold":
            builder.config.entity_sim_threshold = v
        else:
            setattr(kb, k, v)
    raw = []
    for ch in chunks:
        key = hashlib.sha256(f"{kb.PROMPT_VERSION}_{builder.config.llm_model_name}_{ch['text']}".encode("utf-8")).hexdigest()
        parsed = read_json(os.path.join(builder.config.llm_cache_dir, f"{key}.json"))
        if parsed is None:
            continue  # never call the API here
        for c in builder._filter_claims(parsed.get("claims", []), parsed.get("entities", [])):
            c.update({"document_id": ch["document_id"], "chunk_id": ch["chunk_id"], "source_name": ch["source_name"]})
            raw.append(c)
    chunks_by_id = {c["chunk_id"]: c["text"] for c in chunks}
    verified, canonical, grounding = builder.validate_and_resolve_entities(raw, chunks_by_id)
    store = {"claims": verified, "grounding_scores": grounding, "chunk_coverage": builder.chunk_coverage(chunks, verified)}
    prov, cov, topo, cons, s_kg = builder.aggregate_fidelity(store)
    return {"S_KG": s_kg, "phi_prov": prov, "phi_cov": cov, "phi_topo": topo, "phi_cons": cons,
            "verified": len(verified), "raw": len(grounding), "entities": len(canonical),
            "acceptance": len(verified) / max(1, len(grounding))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=None)
    args = ap.parse_args()
    limit_cpu_threads(2)
    ws_id = args.workspace or ws.active_workspace_id()
    d = ws.workspace_dir(ws_id)
    chunks = read_json(os.path.join(d, "latest_chunks.json"), default=[]) or []
    builder = kb.KGBuilder(KGBuilderConfig(KGAuditPipelineConfig(workspace_id=ws_id)))
    import io
    import contextlib
    rows = []
    settings_list = [("default", None, None)] + [(p, p, v) for p, vs in GRID.items() for v in vs]
    for label, param, value in settings_list:
        s = dict(DEFAULTS)
        if param:
            s[param] = value
        t = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            r = run_once(builder, chunks, s)
        r.update({"parameter": param or "default", "value": value if param else None,
                  "default_value": DEFAULTS.get(param) if param else None, "meaning": MEANING.get(param, "all defaults")})
        rows.append(r)
        print(f"{label:28s} {'' if value is None else value!s:>6}  S_KG={r['S_KG']:.4f}  prov={r['phi_prov']:.3f} cov={r['phi_cov']:.3f} "
              f"topo={r['phi_topo']:.3f} cons={r['phi_cons']:.3f}  triples={r['verified']:4d} entities={r['entities']:4d}  ({time.time() - t:.0f}s)", flush=True)
    for k, v in DEFAULTS.items():  # restore
        if k != "entity_sim_threshold":
            setattr(kb, k, v)
    base = rows[0]["S_KG"]
    span = max(r["S_KG"] for r in rows) - min(r["S_KG"] for r in rows)
    out = {"workspace": ws_id, "defaults": DEFAULTS, "rows": rows, "S_KG_default": base, "S_KG_range": span,
           "max_abs_deviation": max(abs(r["S_KG"] - base) for r in rows)}
    write_json_atomic(os.path.join(d, "sensitivity_phase1.json"), out, indent=2)
    print(f"\nS_KG at defaults {base:.4f}; max |deviation| over all perturbations {out['max_abs_deviation']:.4f}")


if __name__ == "__main__":
    main()

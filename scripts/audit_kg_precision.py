"""
Extrinsic accuracy of the knowledge graph (complements the intrinsic S_KG).

S_KG is an unsupervised quality index. To report the KG's ACCURACY as a measured quantity, we draw a
uniform random sample of n accepted triples and have an auditor check each triple against its source
passage. Precision is reported with a 95% Wilson score interval (Wilson, 1927):

    p_hat = k / n,
    CI    = [ (p_hat + z^2/2n  -/+  z * sqrt(p_hat(1-p_hat)/n + z^2/4n^2)) / (1 + z^2/n) ],   z = 1.96

  strict  precision : CORRECT
  lenient precision : CORRECT or PARTIALLY_CORRECT
Also broken down by source PDF and by evidence-match type (verbatim vs paraphrase-anchored).

The auditor is an LLM (use a model different from the extractor via --model). For the paper, export the
same sample for human checking with --export-human and re-score with --human <csv>.

  python scripts/audit_kg_precision.py --n 100
  python scripts/audit_kg_precision.py --n 100 --export-human        # -> CSV to annotate
  python scripts/audit_kg_precision.py --human kg_audit_sample.csv   # human-verified precision
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(__file__))
from _llm import ROOT, StageReport, be_nice, chat_json, run_parallel  # noqa: E402
from kg_audit.utils.api_errors import FatalAPIError, friendly  # noqa: E402
from kg_audit.utils.llm import model_for  # noqa: E402

import pandas as pd  # noqa: E402

from kg_audit.utils import workspace as ws  # noqa: E402
from kg_audit.utils.jobs import JobStatus  # noqa: E402
from kg_audit.utils.main_utils import read_json, write_json_atomic  # noqa: E402

SYSTEM = "You audit knowledge-graph triples extracted from scientific papers. Reply with one JSON object."
USER = """PASSAGE (source of the triple):
\"\"\"{chunk}\"\"\"

EVIDENCE SENTENCE cited by the extractor:
\"\"\"{quote}\"\"\"

TRIPLE:  ({s})  --[{r}]-->  ({o})

Is this triple a correct and faithful statement of what the PASSAGE says? Check (1) both entities are named
correctly and are the right things, (2) the relation and its direction are right, (3) nothing is added that
the passage does not say.

Return JSON: {{"verdict": "CORRECT|PARTIALLY_CORRECT|INCORRECT", "error_type": "none|entity|relation|direction|unsupported",
"reason": "<one sentence>"}}"""


def wilson(k: int, n: int, z: float = 1.96) -> Dict[str, float]:
    if n == 0:
        return {"p": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"p": p, "lo": max(0.0, centre - half), "hi": min(1.0, centre + half), "n": n, "k": k}


def summarise(rows: List[Dict[str, Any]], label_key: str) -> Dict[str, Any]:
    def prec(sub):
        n = len(sub)
        strict = sum(r[label_key] == "CORRECT" for r in sub)
        lenient = sum(r[label_key] in ("CORRECT", "PARTIALLY_CORRECT") for r in sub)
        return {"strict": wilson(strict, n), "lenient": wilson(lenient, n)}

    by_pdf, by_match, errors = defaultdict(list), defaultdict(list), defaultdict(int)
    for r in rows:
        by_pdf[r["source_name"]].append(r)
        by_match[r.get("evidence_match", "verbatim")].append(r)
        errors[r.get("error_type", "none")] += 1
    return {"overall": prec(rows), "by_pdf": {k: prec(v) for k, v in by_pdf.items()},
            "by_evidence_match": {k: prec(v) for k, v in by_match.items()}, "error_types": dict(errors)}


TRIPLE_FEATURES = ["grounding_score", "verbatim", "extraction_confidence", "log_support", "log_deg_subject",
                   "log_deg_object", "log_relation_freq", "cross_pdf", "log_quote_len"]


def triple_feature_matrix(claims: List[Dict[str, Any]], store: Dict[str, Any]):
    """Per-triple evidence signals available for EVERY triple of the KG (no labels needed)."""
    import numpy as np
    from collections import Counter
    deg, rel, sup = Counter(), Counter(), Counter()
    for c in store.get("claims", []):
        deg[c["subject"]] += 1
        deg[c["object"]] += 1
        rel[c["relation"]] += 1
        sup[(c["subject"], c["relation"], c["object"])] += 1
    ents = store.get("entities", {})
    rows = []
    for c in claims:
        rows.append([
            float(c.get("grounding_score", 0.0) or 0.0),
            float(c.get("evidence_match", "verbatim") == "verbatim"),
            float(c.get("extraction_confidence", 0.9) or 0.9),
            np.log1p(sup[(c["subject"], c["relation"], c["object"])]),
            np.log1p(deg[c["subject"]]), np.log1p(deg[c["object"]]), np.log1p(rel[c["relation"]]),
            float(len(ents.get(c["subject"], {}).get("source_pdfs", [])) > 1 or len(ents.get(c["object"], {}).get("source_pdfs", [])) > 1),
            np.log1p(len(c.get("evidence_quote", "") or "")),
        ])
    return np.asarray(rows, dtype=float)


def model_assisted_precision(population: List[Dict[str, Any]], sample_idx: List[int], y: List[int],
                             store: Dict[str, Any], folds: int = 5, seed: int = 0) -> Dict[str, Any]:
    """
    LEARNED KG accuracy: a model-assisted (difference / GREG) estimator from survey sampling
    (Sarndal, Swensson & Wretman, 1992).

      1. learn  P(triple correct | evidence signals)  with a regularised logistic model on the audited sample
      2. predict it for ALL N triples of the KG                                   -> mean(y_hat_pop)
      3. correct the model's bias with cross-fitted residuals on the sample       -> mean(y_s - y_hat_s)
         p_DE = mean(y_hat_pop) + mean(y_s - y_hat_s^cf)
         SE   = sqrt( (1 - n/N) * Var(y_s - y_hat_s^cf) / n )
    The estimator is unbiased under simple random sampling whatever the model, and its variance shrinks
    as the model explains more of the label variance - so the learned weights of the evidence signals
    buy a tighter interval than the plain proportion (Wilson), never a biased one.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y, dtype=int)
    N, n = len(population), len(y)
    if n < 20 or len(np.unique(y)) < 2:
        return {"available": False, "reason": "need >= 20 audited triples with both outcomes"}
    Xp = triple_feature_matrix(population, store)
    Xs = Xp[sample_idx]
    make = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))  # noqa: E731
    cf = np.zeros(n)
    k = min(folds, int(min(np.bincount(y))))
    if k >= 2:
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(Xs, y):
            cf[te] = make().fit(Xs[tr], y[tr]).predict_proba(Xs[te])[:, 1]
    else:
        cf[:] = y.mean()
    model = make().fit(Xs, y)
    y_pop = model.predict_proba(Xp)[:, 1]
    resid = y - cf
    est = float(y_pop.mean() + resid.mean())
    se = float(np.sqrt(max(0.0, 1 - n / N) * resid.var(ddof=1) / n))
    w = model.named_steps["logisticregression"].coef_[0]
    return {"available": True, "estimate": float(np.clip(est, 0, 1)), "se": se,
            "lo": float(np.clip(est - 1.96 * se, 0, 1)), "hi": float(np.clip(est + 1.96 * se, 0, 1)),
            "n": int(n), "N": int(N), "plain_proportion": float(y.mean()),
            "learned_weights": dict(zip(TRIPLE_FEATURES, map(float, w))),
            "intercept": float(model.named_steps["logisticregression"].intercept_[0]),
            "mean_predicted_correctness": float(y_pop.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default=model_for("judge"))
    ap.add_argument("--export-human", action="store_true")
    ap.add_argument("--human", default="")
    args = ap.parse_args()
    be_nice()
    os.chdir(ROOT)

    ws_id = args.workspace or ws.active_workspace_id()
    ws_dir = ws.workspace_dir(ws_id)
    out_path = os.path.join(ws_dir, "kg_precision_audit.json")
    job = JobStatus(os.getenv("EPISTEME_JOB_DIR"))

    store = read_json(os.path.join(ws_dir, "kg_store.json"), default={}) or {}
    chunks = {c["chunk_id"]: c["text"] for c in read_json(os.path.join(ws_dir, "latest_chunks.json"), default=[]) or []}
    claims = [c for c in store.get("claims", []) if c.get("chunk_id") in chunks]

    if args.human:
        df = pd.read_csv(args.human).dropna(subset=["human_verdict"])
        df = df[df["human_verdict"].astype(str).str.strip() != ""]
        rows = df.to_dict("records")
        for r in rows:
            r["human_verdict"] = str(r["human_verdict"]).upper().strip()
        res = read_json(out_path, default={}) or {}
        res["human"] = summarise(rows, "human_verdict")
        if "llm_verdict" in df:
            res["human"]["auditor_agreement"] = float((df["human_verdict"].str.upper() == df["llm_verdict"].str.upper()).mean())
        if "pop_index" in df:
            res["human"]["model_assisted"] = model_assisted_precision(
                claims, [int(r["pop_index"]) for r in rows], [int(r["human_verdict"] == "CORRECT") for r in rows], store)
        write_json_atomic(out_path, res)
        print(res["human"]["overall"], res["human"].get("model_assisted", {}).get("estimate"))
        return

    if not claims:
        job.finish("failed", error="no triples in this workspace")
        sys.exit("No triples to audit.")
    sample_idx = random.Random(args.seed).sample(range(len(claims)), min(args.n, len(claims)))
    sample = [claims[i] for i in sample_idx]
    jobs = [{"i": i, "c": c, "pop_index": sample_idx[i], "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(
        chunk=chunks[c["chunk_id"]][:6000], quote=c.get("evidence_quote", ""), s=c["subject"], r=c["relation"], o=c["object"])}]}
        for i, c in enumerate(sample)]
    done = [0]
    job.update("audit", 0, len(jobs), note=f"auditor: {args.model}")

    def tick(_j, _r):
        done[0] += 1
        job.update("audit", done[0], len(jobs), note=f"auditor: {args.model}")

    try:
        rep = StageReport(len(jobs))
        outs = run_parallel(jobs, lambda j: chat_json(args.model, j["messages"], tag="kg_audit"), workers=args.workers,
                            desc="kg-audit", on_result=tick, report=rep)
        if rep.fatal:   # a partial sample would bias the precision estimate - stop instead
            raise FatalAPIError(f"{rep.fatal}\n{rep.ok}/{len(jobs)} audit calls are cached; run the audit again afterwards.")
        if rep.failed > 0.1 * len(jobs):
            raise RuntimeError(f"{rep.failed}/{len(jobs)} audit calls failed ({rep.summary()}); run again - finished calls are cached.")
        rows = []
        for j, o in zip(jobs, outs):
            if not isinstance(o, dict):
                continue
            c = j["c"]
            v = str(o.get("verdict", "INCORRECT")).upper().strip()
            rows.append({"pop_index": j["pop_index"], "subject": c["subject"], "relation": c["relation"], "object": c["object"],
                         "evidence_quote": c.get("evidence_quote", ""), "source_name": c.get("source_name", ""),
                         "chunk_id": c["chunk_id"], "evidence_match": c.get("evidence_match", "verbatim"),
                         "grounding_score": c.get("grounding_score"),
                         "llm_verdict": v if v in ("CORRECT", "PARTIALLY_CORRECT", "INCORRECT") else "INCORRECT",
                         "error_type": o.get("error_type", "none"), "reason": o.get("reason", "")})
        res = {"workspace": ws_id, "auditor_model": args.model, "sample_size": len(rows), "population": len(claims),
               "seed": args.seed, "created_at": datetime.now(timezone.utc).isoformat(),
               "llm": summarise(rows, "llm_verdict"), "items": rows}
        res["llm"]["model_assisted"] = model_assisted_precision(
            claims, [r["pop_index"] for r in rows], [int(r["llm_verdict"] == "CORRECT") for r in rows], store)
        prev = read_json(out_path, default={}) or {}
        if "human" in prev:
            res["human"] = prev["human"]
        write_json_atomic(out_path, res)
        if args.export_human:
            p = os.path.join(ws_dir, "kg_audit_sample.csv")
            pd.DataFrame(rows).assign(human_verdict="").to_csv(p, index=False)
            print(f"human annotation sheet -> {p}")
        o = res["llm"]["overall"]["strict"]
        print(f"KG precision (strict) = {o['p']:.3f}  95% CI [{o['lo']:.3f}, {o['hi']:.3f}]  n={o['n']}")
        job.finish("completed", result={"precision_strict": o, "precision_lenient": res["llm"]["overall"]["lenient"]})
    except Exception as e:
        import traceback
        traceback.print_exc()
        job.finish("failed", error=friendly(e))
        raise


if __name__ == "__main__":
    main()

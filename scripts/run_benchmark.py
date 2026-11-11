"""
Research benchmark runner (labelled; resumable; CPU-throttled). Operates on ONE corpus workspace.

  Stage 1  INTERCEPT  Phase 2 answer + claim extraction for every query   (I/O pool, cached)
  Stage 2  JUDGE      response-level label from an LLM judge that sees the gold evidence
                      (I/O pool, cached). y = 1 iff the verdict is SUPPORTED.
  Stage 3  FEATURES   Phase 3 CASSE vector for every response              (sequential, thread-capped)
  Stage 4  TRAIN      Phase 4 tournament + pool + conformal, ONCE
  Stage 5  REPORT     results CSV, summary JSON, paper tables/figures

Input: <workspace>/benchmark/benchmark.jsonl (from generate_benchmark.py) or any .txt file with one
question per line (optionally "ID | CATEGORY | ... | question"). Items without gold evidence are
judged against passages retrieved from the corpus.

Usage
  python scripts/run_benchmark.py                                   # active workspace
  python scripts/run_benchmark.py --bench my_queries.txt --limit 50
  python scripts/run_benchmark.py --export-annotation 100           # sample for human validation
  python scripts/run_benchmark.py --agreement <csv>
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(__file__))
from _llm import ROOT, StageReport, be_nice, chat_json, run_batch_api, run_parallel  # noqa: E402

os.chdir(ROOT)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kg_audit.components.decision import CLAIM_FEATURES, FEATURE_KEYS  # noqa: E402
from kg_audit.components.llm_interceptor import LLMInterceptor  # noqa: E402
from kg_audit.components.model_trainer import LABELLED_COLUMNS, split_of  # noqa: E402
from kg_audit.components.truth_evaluator import CorpusIndex  # noqa: E402
from kg_audit.pipelines.evaluation_pipeline import EvaluationPipeline  # noqa: E402
from kg_audit.utils import workspace as ws  # noqa: E402
from kg_audit.utils.api_errors import FatalAPIError, friendly, is_fatal  # noqa: E402
from kg_audit.utils.llm import model_for, provider_of, usage_snapshot  # noqa: E402
from kg_audit.utils.jobs import JobStatus  # noqa: E402
from kg_audit.utils.main_utils import embed, limit_cpu_threads, read_json, read_jsonl, write_json_atomic  # noqa: E402

JUDGE_SYSTEM = ("You are a meticulous scientific fact-checking judge. You decide whether an AI answer is faithful "
                "to the source papers. Reply with one JSON object.")
JUDGE_USER = """QUESTION:
{question}

QUESTION TYPE: {category} - {expectation}

REFERENCE INFORMATION:
{reference}

EVIDENCE FROM THE SOURCE PAPERS:
\"\"\"
{evidence}
\"\"\"

AI ANSWER:
\"\"\"
{answer}
\"\"\"

Instructions
1. List every factual claim the AI ANSWER makes about the methods/papers.
2. A claim is SUPPORTED if the EVIDENCE states or directly entails it, or if it is uncontroversial textbook
   background (e.g. the definition of stochastic gradient descent). It is UNSUPPORTED if it attributes to the
   papers something the evidence does not contain or contradicts.
3. If the question contains a false premise and the answer goes along with it, that is UNSUPPORTED.
   If the question is not answerable from the papers, inventing specifics is UNSUPPORTED; saying the papers
   do not report it is SUPPORTED.
4. Verdict: SUPPORTED (no unsupported claim), PARTIALLY_SUPPORTED (some unsupported claims), UNSUPPORTED
   (the core of the answer is unsupported).

Return JSON:
{{"claims_checked": <int>, "unsupported_claims": ["..."], "accepted_false_premise": <true|false|null>,
  "verdict": "SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED", "rationale": "<two sentences>"}}"""

EXPECTATION = {
    "FACTUAL": "answerable from the papers",
    "CROSS_DOC": "answerable by combining two papers",
    "BORDERLINE": "NOT answerable from the papers; a faithful answer says so",
    "FALSE_PREMISE": "contains a false premise; a faithful answer corrects it",
    "HALLUCINATION": "likely contains a false premise; a faithful answer corrects it",
    "USER": "unknown type; judge only against the evidence",
}


def load_items(path: str) -> List[Dict[str, Any]]:
    """.jsonl benchmark records, or a .txt file (plain questions or legacy 'ID | CAT | ... | question')."""
    if path.endswith(".jsonl"):
        return read_jsonl(path)
    items = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for k, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3 and re.match(r"^[A-Za-z]*\d+$", parts[0]):
                qid, cat, q = parts[0], parts[1].upper().replace(" ", "_"), parts[-1]
            else:
                qid, cat, q = f"U{k:04d}", "USER", line
            items.append({"id": qid, "category": cat, "question": q, "expected_behavior": "UNKNOWN",
                          "chunk_ids": [], "evidence_quotes": []})
    return items


class Progress:
    """Heartbeat to the run folder (CLI) and, when launched by the website, to the job folder."""

    def __init__(self, run_dir: str):
        self.path = os.path.join(run_dir, "progress.json")
        self.job = JobStatus(os.getenv("EPISTEME_JOB_DIR"))
        self.state: Dict[str, Any] = read_json(self.path, default={}) or {}

    def update(self, stage: str, done: int, total: int, note: str = "") -> None:
        usd = usage_snapshot()["usd_estimate"]
        if usd > 0:
            note = (note + " | " if note else "") + f"~${usd:.2f} API spend this run"
        self.state.update({"stage": stage, "done": done, "total": total, "note": note,
                           "updated_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()})
        write_json_atomic(self.path, self.state)
        self.job.update(stage, done, total, note)

    def finish(self, status: str = "completed", **extra) -> None:
        self.state.update({"stage": status, **extra, "updated_at": datetime.now(timezone.utc).isoformat()})
        write_json_atomic(self.path, self.state)
        self.job.finish(status if status in ("completed", "failed") else "completed", **extra)


class RunStopped(FatalAPIError):
    """A stage could not continue (e.g. no API credits). The message is shown verbatim on the website."""


def _count(run_dir: str, sub: str) -> int:
    d = os.path.join(run_dir, sub)
    return len([f for f in os.listdir(d) if f.endswith(".json")]) if os.path.isdir(d) else 0


def _record_failures(run_dir: str, stage: str, rep: "StageReport") -> None:
    path = os.path.join(run_dir, "failures.json")
    log = read_json(path, default={}) or {}
    log[stage] = {**rep.to_dict(), "at": datetime.now(timezone.utc).isoformat()}
    write_json_atomic(path, log)


def _saved_line(run_dir: str, n_items: int) -> str:
    feats = len(read_jsonl(os.path.join(run_dir, "features.jsonl")))
    return (f"Saved so far: {_count(run_dir, 'intercepts')}/{n_items} answers, {_count(run_dir, 'judgements')} judged, "
            f"{feats} feature vectors. Click Run again to resume - finished work is cached and is not re-paid.")


# ----------------------------------------------------------------------------------------- stages
def stage_intercept(items, run_dir, pipe: EvaluationPipeline, workers: int, prog: Progress, mode: str, label="intercept"):
    out_dir = os.path.join(run_dir, "intercepts")
    os.makedirs(out_dir, exist_ok=True)
    todo = [it for it in items if not os.path.exists(os.path.join(out_dir, f"{it['id']}.json"))]
    print(f"[{label}] {len(items) - len(todo)} already answered, {len(todo)} to run", flush=True)
    rep = StageReport(len(todo))
    if not todo:
        return rep

    def job(it):
        cfg = copy.copy(pipe.interceptor_config)
        cfg.captured_payload_file_path = os.path.join(run_dir, "payloads", f"{it['id']}.json")
        art, resp = LLMInterceptor(cfg, client=pipe.interceptor.client).intercept_and_generate(it["question"], mode=mode)
        if not str(art.llm_response or "").strip():
            raise ValueError("empty answer from the LLM")
        rec = {"answer": art.llm_response, "confidence": art.llm_self_reported_accuracy,
               "confidence_reasoning": art.llm_confidence_reasoning, "triples": resp["triples"],
               "token_nll": (resp.get("metadata") or {}).get("token_nll")}
        write_json_atomic(os.path.join(out_dir, f"{it['id']}.json"), rec)
        return rec

    prog.update(label, len(items) - len(todo), len(items))

    def tick(_j, _r):     # progress = answers actually on disk, never merely "attempted"
        prog.update(label, len(items) - len(todo) + rep.ok, len(items),
                    note=f"{rep.failed} failed" + (" - stopping: " + rep.fatal if rep.fatal else ""))

    run_parallel(todo, job, workers=workers, desc=label, on_result=tick, report=rep)
    _record_failures(run_dir, label, rep)
    if rep.fatal:
        raise RunStopped(f"{rep.fatal}\n{_saved_line(run_dir, len(items))}")
    return rep


_CHUNKS: Dict[str, str] = {}


def _chunk_map(chunks_path: str) -> Dict[str, str]:
    if not _CHUNKS:
        _CHUNKS.update({c["chunk_id"]: c["text"] for c in read_json(chunks_path, default=[]) or []})
    return _CHUNKS


def _judge_messages(it, ans, corpus: CorpusIndex, chunks_path: str) -> List[Dict[str, str]]:
    cat = it.get("category", "USER")
    ref = {"FACTUAL": f"Gold answer: {it.get('gold_answer', '')}",
           "CROSS_DOC": f"Gold answer: {it.get('gold_answer', '')}",
           "BORDERLINE": f"Why it is unanswerable: {it.get('why_unanswerable', '')}",
           "FALSE_PREMISE": f"False premise: {it.get('false_premise', '')}\nCorrection: {it.get('correction', '')}",
           }.get(cat, "No gold reference: judge only against the evidence below.")
    chunks = _chunk_map(chunks_path)
    evidence = [f"[{cid}] {chunks.get(cid, '')}" for cid in it.get("chunk_ids", []) if cid in chunks]
    if corpus.available and ans.get("answer"):      # passages relevant to what the answer says
        sims = corpus.sentence_emb @ embed(ans["answer"]).astype(np.float32)
        k = 6 if evidence else 12
        evidence.append("Relevant sentences from the papers:\n- " + "\n- ".join(corpus.sentences[i] for i in np.argsort(-sims)[:k]))
    user = JUDGE_USER.format(question=it["question"], category=cat, expectation=EXPECTATION.get(cat, EXPECTATION["USER"]),
                             reference=ref, evidence="\n\n".join(evidence)[:12000], answer=ans.get("answer", "")[:6000])
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def stage_judge(items, run_dir, pipe: EvaluationPipeline, model: str, workers: int, prog: Progress, use_batch: bool):
    out_dir = os.path.join(run_dir, "judgements")
    os.makedirs(out_dir, exist_ok=True)
    chunks_path = pipe.interceptor_config.latest_chunks_path
    corpus = CorpusIndex.get(chunks_path, pipe.pipeline_config.cache_dir)
    todo, n_judged, n_no_answer, n_rejudge = [], 0, 0, 0
    for it in items:
        jpath = os.path.join(out_dir, f"{it['id']}.json")
        if os.path.exists(jpath):
            old = read_json(jpath, default={}) or {}
            if old.get("judge_model", model) == model:
                n_judged += 1
                continue
            # judged by a different model: keep it (inter-judge agreement) but re-judge with the current judge
            arch = os.path.join(run_dir, "judgements_archive", re.sub(r"[^A-Za-z0-9._-]", "_", str(old.get("judge_model"))))
            os.makedirs(arch, exist_ok=True)
            os.replace(jpath, os.path.join(arch, f"{it['id']}.json"))
            n_rejudge += 1
        ans = read_json(os.path.join(run_dir, "intercepts", f"{it['id']}.json"))
        if ans:
            todo.append({"id": it["id"], "messages": _judge_messages(it, ans, corpus, chunks_path)})
        else:
            n_no_answer += 1
    print(f"[judge] ({model}): {n_judged} already judged, {len(todo)} to run ({n_rejudge} re-judged because an earlier "
          f"judge model was used), {n_no_answer} have no answer yet", flush=True)
    rep = StageReport(len(todo))
    if not todo:
        return rep
    prog.update("judge", n_judged, len(items))

    def save(j, o) -> bool:
        if isinstance(o, dict) and o.get("verdict"):
            v = str(o["verdict"]).upper().strip().replace(" ", "_")
            o["verdict"] = v if v in {"SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"} else "UNSUPPORTED"
            o["judge_model"] = model
            write_json_atomic(os.path.join(out_dir, f"{j['id']}.json"), o)
            return True
        return False

    if use_batch:
        prog.update("judge", n_judged, len(items), note="submitting to the Batch API (50% price)")
        try:
            outs = run_batch_api(todo, model, tag="bench_judge",
                                 progress=lambda d, t, note: prog.update("judge", n_judged + d, len(items), note=note))
        except Exception as e:
            if is_fatal(e):
                rep.fatal = friendly(e)
                _record_failures(run_dir, "judge", rep)
                raise RunStopped(f"{rep.fatal}\n{_saved_line(run_dir, len(items))}")
            raise
        for j, o in zip(todo, outs):
            if save(j, o):
                rep.ok += 1
            else:
                rep.add_error(j["id"], ValueError("batch result missing or without a verdict"))
    else:
        def call(j):     # verdict is written the moment it arrives: a crash later loses nothing
            o = chat_json(model, j["messages"], tag="bench_judge")
            if not save(j, o):
                raise ValueError("judge reply had no 'verdict' field")
            return o

        def tick(_j, _o):
            prog.update("judge", n_judged + rep.ok, len(items),
                        note=f"{rep.failed} failed" + (" - stopping: " + rep.fatal if rep.fatal else ""))

        run_parallel(todo, call, workers=workers, desc="judge", on_result=tick, report=rep)
    _record_failures(run_dir, "judge", rep)
    if rep.fatal:
        raise RunStopped(f"{rep.fatal}\n{_saved_line(run_dir, len(items))}")
    return rep


def crosscheck_sample(items, run_dir: str, n: int) -> List[str]:
    """Fixed, reproducible, category-stratified sample of JUDGED answers (same ids on every rerun)."""
    import hashlib
    path = os.path.join(run_dir, "crosscheck_ids.json")
    saved = read_json(path, default=None)
    if saved and saved.get("n") == n:
        return saved["ids"]
    judged = [it for it in items if os.path.exists(os.path.join(run_dir, "judgements", f"{it['id']}.json"))]
    if not judged:
        return []
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for it in judged:
        by_cat.setdefault(it.get("category", "USER"), []).append(it)
    key = lambda it: hashlib.sha256(f"crosscheck|{it['id']}".encode()).hexdigest()
    n = min(n, len(judged))
    quota = {c: max(1, round(n * len(v) / len(judged))) for c, v in by_cat.items()}
    while sum(quota.values()) > n:                     # fix rounding
        c = max(quota, key=lambda c: quota[c]); quota[c] -= 1
    ids = [it["id"] for c, v in sorted(by_cat.items()) for it in sorted(v, key=key)[: quota[c]]]
    write_json_atomic(path, {"n": n, "ids": ids, "rule": "per-category quota, lowest sha256('crosscheck|id') first"})
    return ids


def stage_crosscheck(items, run_dir, pipe: EvaluationPipeline, primary: str, second: str, n: int, prog: Progress,
                     use_batch: bool, bench_dir: str) -> Dict[str, Any]:
    """
    Re-judge a fixed stratified sample with a SECOND (stronger) judge and measure inter-judge agreement.
    It never changes the training labels (those come from the primary judge); it measures how reliable they are.
    """
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    ids = crosscheck_sample(items, run_dir, n)
    if not ids or second == primary:
        return {}
    by_id = {it["id"]: it for it in items}
    out_dir = os.path.join(run_dir, "crosscheck", re.sub(r"[^A-Za-z0-9._-]", "_", second))
    os.makedirs(out_dir, exist_ok=True)
    chunks_path = pipe.interceptor_config.latest_chunks_path
    corpus = CorpusIndex.get(chunks_path, pipe.pipeline_config.cache_dir)
    todo = []
    for i in ids:
        if os.path.exists(os.path.join(out_dir, f"{i}.json")):
            continue
        ans = read_json(os.path.join(run_dir, "intercepts", f"{i}.json"))
        if ans:
            todo.append({"id": i, "messages": _judge_messages(by_id[i], ans, corpus, chunks_path)})
    print(f"[cross-check] second judge {second}: {len(ids) - len(todo)} done, {len(todo)} to run (sample of {len(ids)})", flush=True)
    prog.update("cross-check", len(ids) - len(todo), len(ids), note=f"second judge {second}")
    rep = StageReport(len(todo))

    def save(j, o) -> bool:
        if isinstance(o, dict) and o.get("verdict"):
            v = str(o["verdict"]).upper().strip().replace(" ", "_")
            o["verdict"] = v if v in {"SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"} else "UNSUPPORTED"
            o["judge_model"] = second
            write_json_atomic(os.path.join(out_dir, f"{j['id']}.json"), o)
            return True
        return False

    if todo:
        if use_batch:
            try:
                outs = run_batch_api(todo, second, tag="bench_judge",
                                     progress=lambda d, t, note: prog.update("cross-check", len(ids) - len(todo) + d, len(ids), note=note))
            except Exception as e:
                if is_fatal(e):
                    raise RunStopped(f"{friendly(e)}\n{_saved_line(run_dir, len(items))}")
                raise
            for j, o in zip(todo, outs):
                rep.ok += 1 if save(j, o) else 0
        else:
            def call(j):
                o = chat_json(second, j["messages"], tag="bench_judge")
                if not save(j, o):
                    raise ValueError("judge reply had no 'verdict' field")
                return o
            run_parallel(todo, call, workers=4, desc="cross-check", report=rep)
            if rep.fatal:
                raise RunStopped(f"{rep.fatal}\n{_saved_line(run_dir, len(items))}")

    pairs = []
    for i in ids:
        a = read_json(os.path.join(run_dir, "judgements", f"{i}.json"), default={}) or {}
        b = read_json(os.path.join(out_dir, f"{i}.json"), default={}) or {}
        if a.get("verdict") and b.get("verdict"):
            pairs.append((i, by_id[i].get("category", "USER"), a["verdict"], b["verdict"]))
    if len(pairs) < 5:
        return {}
    va, vb = [p[2] for p in pairs], [p[3] for p in pairs]
    ba, bb = [int(v == "SUPPORTED") for v in va], [int(v == "SUPPORTED") for v in vb]
    agree_cat: Dict[str, float] = {}
    for c in sorted({p[1] for p in pairs}):
        sel = [k for k, p in enumerate(pairs) if p[1] == c]
        agree_cat[c] = round(float(np.mean([ba[k] == bb[k] for k in sel])), 4)
    kb = float(cohen_kappa_score(ba, bb)) if len(set(ba + bb)) > 1 else float("nan")
    res = {"primary_judge": primary, "second_judge": second, "n": len(pairs),
           "cohen_kappa_binary": kb, "cohen_kappa_3class": float(cohen_kappa_score(va, vb)),
           "raw_agreement_binary": float(np.mean([x == y for x, y in zip(ba, bb)])),
           "supported_rate_primary": float(np.mean(ba)), "supported_rate_second": float(np.mean(bb)),
           "confusion_binary[primary x second]": confusion_matrix(ba, bb, labels=[0, 1]).tolist(),
           "agreement_by_category": agree_cat,
           "disagreements": [p[0] for p, x, y in zip(pairs, ba, bb) if x != y],
           "note": "Measures label reliability only; training labels come from the primary judge."}
    write_json_atomic(os.path.join(run_dir, "judge_crosscheck.json"), res)
    write_json_atomic(os.path.join(bench_dir, "judge_crosscheck.json"), res)
    print(f"[cross-check] {primary} vs {second}: kappa={kb:.3f}, raw agreement={res['raw_agreement_binary']:.3f} on n={len(pairs)}", flush=True)
    return res


def stage_features(items, run_dir, pipe: EvaluationPipeline, prog: Progress, mode: str, require_judgement: bool = True,
                   label: str = "features") -> pd.DataFrame:
    feat_path = os.path.join(run_dir, "features.jsonl")
    done_ids = {r["id"] for r in read_jsonl(feat_path)}
    todo = [it for it in items if it["id"] not in done_ids
            and os.path.exists(os.path.join(run_dir, "intercepts", f"{it['id']}.json"))
            and (not require_judgement or os.path.exists(os.path.join(run_dir, "judgements", f"{it['id']}.json")))]
    print(f"[{label}] {len(done_ids)} cached, {len(todo)} to compute (sequential, CPU-capped)", flush=True)
    t0 = time.time()
    skipped: List[Dict[str, str]] = []
    for k, it in enumerate(todo, start=1):
        ans = read_json(os.path.join(run_dir, "intercepts", f"{it['id']}.json"))
        jd = read_json(os.path.join(run_dir, "judgements", f"{it['id']}.json"), default={}) or {}
        try:
            _, retr, ev = pipe.run_pipeline(
                it["question"], mode=mode, log_history=False, log_telemetry=False,
                precomputed_answer={"answer": ans["answer"], "confidence": ans["confidence"],
                                    "confidence_reasoning": ans.get("confidence_reasoning", ""),
                                    "_token_nll": ans.get("token_nll")},
                artifact_dir=os.path.join(run_dir, "artifacts", it["id"]))
        except Exception as e:
            if is_fatal(e):
                raise RunStopped(f"{friendly(e)}\n{_saved_line(run_dir, len(items))}")
            skipped.append({"id": it["id"], "error": str(e)[:300]})
            print(f"  [skip] {it['id']}: {e}", flush=True)
            continue
        d = ev.diagnostics
        rec = {"id": it["id"], "category": it.get("category", "USER"), "question": it["question"],
               "answer": ans["answer"][:2000], "s_kg_local": ev.local_kg_fidelity_score,
               "gt_nodes": len(retr.extracted_nodes), "n_claims": d.get("response_claims", 0),
               "asserted_claims": d.get("asserted_claims", 0), "unsupported_claims": d.get("unsupported_claims", 0),
               "hallucination_extent": d.get("hallucination_extent", 0.0),
               "alignment_rate": d.get("entity_alignment_rate", 0.0),
               "decision": ev.decision_status, "S_query": ev.query_belief_score, "S_final": ev.final_accuracy_score,
               "hallucination_probability": d.get("hallucination_probability"),
               "brain_transferred": bool((d.get("brain") or {}).get("transferred")), **ev.features,
               "claim_bag": d.get("claim_matrix", []),
               "claims": [{"source": c["source"], "relation": c["relation"], "target": c["target"], "claim_type": c["claim_type"],
                           "support_tier": c["support_tier"], "nli_entail": c.get("nli_entail"), "nli_contra": c.get("nli_contra"),
                           "learned_hallucination_score": c.get("learned_hallucination_score")} for c in ev.edge_audit_details]}
        if jd:
            rec.update({"judge_verdict": jd["verdict"], "label": int(jd["verdict"] == "SUPPORTED")})
        with open(feat_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        eta = (time.time() - t0) / k * (len(todo) - k)
        if k % 5 == 0 or k == len(todo):
            print(f"  [{label}] {k}/{len(todo)}  eta {eta / 60:.1f} min", flush=True)
        prog.update(label, len(done_ids) + k, len(items), note=f"eta {eta / 60:.1f} min")
    if skipped:
        log = read_json(os.path.join(run_dir, "failures.json"), default={}) or {}
        log[label] = {"skipped": len(skipped), "examples": skipped[:50], "at": datetime.now(timezone.utc).isoformat()}
        write_json_atomic(os.path.join(run_dir, "failures.json"), log)
        print(f"[{label}] {len(skipped)} responses skipped (see failures.json)", flush=True)
    return pd.DataFrame(read_jsonl(feat_path))


def attach_labels(feats: pd.DataFrame, run_dir: str) -> pd.DataFrame:
    if feats.empty:
        return feats
    feats = feats.copy()
    verdicts, labels = [], []
    for i in feats["id"]:
        jd = read_json(os.path.join(run_dir, "judgements", f"{i}.json"), default={}) or {}
        v = jd.get("verdict")
        verdicts.append(v)
        labels.append(float(v == "SUPPORTED") if v else np.nan)
    feats["judge_verdict"], feats["label"] = verdicts, labels
    # exact answer length (features.jsonl keeps a 2000-char excerpt): used only for the length BASELINE
    feats["answer_chars"] = [len(str((read_json(os.path.join(run_dir, "intercepts", f"{i}.json"), default={}) or {}).get("answer", "")))
                             for i in feats["id"]]
    return feats


def stage_train(feats: pd.DataFrame, items_by_id, pipe: EvaluationPipeline, prog: Progress):
    """Write the labelled store from scratch for this benchmark (no stale rows), then train once."""
    if "label" not in feats.columns:
        feats = feats.assign(label=np.nan, judge_verdict=None)
    lab = feats.dropna(subset=["label"])
    counts = Counter(int(v) for v in lab["label"])
    need_n, need_c = pipe.trainer_config.min_labelled_samples, 10
    if len(lab) < need_n or min(counts.get(0, 0), counts.get(1, 0)) < need_c:
        return {"skipped": True, "n": int(len(lab)), "faithful": counts.get(1, 0), "hallucinated": counts.get(0, 0),
                "reason": (f"Training needs >= {need_n} judged answers with >= {need_c} faithful AND >= {need_c} "
                           f"not-faithful; have {len(lab)} ({counts.get(1, 0)} faithful, {counts.get(0, 0)} not).")}
    prog.update("train", 0, 1, note="10-model nested-CV tournament")
    path = pipe.trainer_config.labelled_store_path
    rows = []
    for _, r in feats.dropna(subset=["label"]).iterrows():
        row = {k: r[k] for k in FEATURE_KEYS}
        row.update({"n_claims": r.get("n_claims"), "answer_chars": r.get("answer_chars", len(str(r.get("answer", ""))))})
        row.update({"query_id": r["id"], "category": r["category"], "label": int(r["label"]),
                    "judge_verdict": r["judge_verdict"], "s_kg_local": r["s_kg_local"],
                    "llm_model": pipe.interceptor_config.llm_model_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(), "query": items_by_id[r["id"]]["question"][:300]})
        rows.append(row)
    if os.path.exists(path):
        os.replace(path, path.replace(".csv", f"_backup_{datetime.now():%Y%m%d%H%M%S}.csv"))
    pd.DataFrame(rows)[LABELLED_COLUMNS].to_csv(path, index=False)
    with open(os.path.join(os.path.dirname(path), "labelled_bags.jsonl"), "w", encoding="utf-8") as f:
        for _, r in feats.dropna(subset=["label"]).iterrows():   # claim bags for the multiple-instance model
            f.write(json.dumps({"query_id": r["id"], "bag": r.get("claim_bag") or []}) + "\n")
    print(f"[train] {len(rows)} labelled responses -> {path}", flush=True)
    return pipe.retrain()


def stage_report(feats: pd.DataFrame, items_by_id, run_dir: str, pipe: EvaluationPipeline) -> Dict[str, Any]:
    """Score every response with the production brain; analysis must use split == 'test' rows only."""
    import pickle
    from kg_audit.components.decision import DecisionBundle
    prod = read_json(pipe.trainer_config.production_metrics_path, default={}) or {}
    bundle = DecisionBundle.from_dict(prod.get("decision_bundle"))
    with open(pipe.trainer_config.final_model_file_path, "rb") as f:
        model = pickle.load(f)
    df = feats.dropna(subset=["label"]).copy()
    bags = [np.asarray(b or [], dtype=float).reshape(-1, len(CLAIM_FEATURES)) for b in df.get("claim_bag", [[]] * len(df))]
    X = df[FEATURE_KEYS].to_numpy(float)
    s_q = (model.predict_proba_bags(X, bags) if hasattr(model, "predict_proba_bags") else model.predict_proba(X))[:, 1]
    s_f = bundle.pool(s_q, df["s_kg_local"].to_numpy(float))
    df["expected_behavior"] = [items_by_id.get(i, {}).get("expected_behavior", "") for i in df["id"]]
    df["S_query"], df["S_final"] = s_q, s_f
    df["hallucination_probability"] = 1.0 - s_f
    df["decision"] = [bundle.conformal.decide(p) for p in s_f]
    df["split"] = [split_of(i, pipe.trainer_config.test_fraction, pipe.trainer_config.calibration_fraction) for i in df["id"]]
    bench_dir = os.path.join(pipe.pipeline_config.workspace_dir, "benchmark")
    if hasattr(model, "claim_scores") and (getattr(model, "uses_bags", False) or getattr(model, "localizer", None) is not None):
        cs = model.claim_scores(X, bags)
        df["claim_scores"] = [json.dumps([round(float(v), 4) for v in c]) for c in cs]
    flat = df.drop(columns=[c for c in ("claim_bag", "claims") if c in df.columns])
    flat.to_csv(os.path.join(bench_dir, "results.csv"), index=False)
    flat.to_csv(os.path.join(run_dir, "results.csv"), index=False)

    test = prod.get("test_metrics", {})
    summary = {
        "run_dir": run_dir, "n_items": int(len(df)),
        "label_distribution": {k: int(v) for k, v in Counter(df["judge_verdict"]).items()},
        "label_by_category": df.groupby("category")["label"].mean().round(4).to_dict(),
        "champion": prod.get("winning_model"), "promoted": prod.get("is_promoted_to_production"),
        "test_S_final": test.get("S_final"), "test_conformal": test.get("conformal"),
        "llm_self_confidence_baseline": test.get("baselines", {}).get("llm_self_confidence"),
        "results_csv": os.path.join(bench_dir, "results.csv"),
    }
    write_json_atomic(os.path.join(run_dir, "summary.json"), summary)
    write_json_atomic(os.path.join(bench_dir, "summary.json"), summary)
    print("[report]\n" + json.dumps(summary, indent=2, default=str), flush=True)
    return summary


# ----------------------------------------------------------------------------------------- human validation
def export_annotation(run_dir: str, n: int) -> None:
    df = pd.read_csv(os.path.join(run_dir, "results.csv"))
    per = max(1, n // max(1, df.groupby(["category", "judge_verdict"]).ngroups))
    sample = df.groupby(["category", "judge_verdict"], group_keys=False).apply(lambda g: g.sample(min(len(g), per), random_state=0)).head(n)
    sample["human_verdict"] = ""
    out = os.path.join(os.path.dirname(os.path.dirname(run_dir)), "human_annotation_sample.csv")
    sample[["id", "category", "question", "answer", "judge_verdict", "human_verdict"]].to_csv(out, index=False)
    print(f"Wrote {len(sample)} items to {out}. Fill 'human_verdict' with SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED.")


def agreement(path: str, bench_dir: str) -> None:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    df = pd.read_csv(path).dropna(subset=["human_verdict"])
    df = df[df["human_verdict"].astype(str).str.strip() != ""]
    hb = (df["human_verdict"].str.upper() == "SUPPORTED").astype(int)
    jb = (df["judge_verdict"].str.upper() == "SUPPORTED").astype(int)
    res = {"n": int(len(df)),
           "cohen_kappa_3class": float(cohen_kappa_score(df["human_verdict"].str.upper(), df["judge_verdict"].str.upper())),
           "cohen_kappa_binary": float(cohen_kappa_score(hb, jb)),
           "raw_agreement_binary": float((hb == jb).mean()),
           "confusion_binary[human x judge]": confusion_matrix(hb, jb, labels=[0, 1]).tolist()}
    write_json_atomic(os.path.join(bench_dir, "judge_agreement.json"), res)
    print(json.dumps(res, indent=2))


# ----------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=None, help="workspace id (default: active)")
    ap.add_argument("--bench", default="", help=".jsonl or .txt (default: <workspace>/benchmark/benchmark.jsonl)")
    ap.add_argument("--run-name", default="latest")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6, help="concurrent API calls (I/O-bound)")
    ap.add_argument("--cpu-threads", type=int, default=2, help="BLAS/torch threads for feature + training stages")
    ap.add_argument("--mode", choices=["closed_book", "grounded"], default="closed_book")
    ap.add_argument("--judge-model", default="", help="default: JUDGE_MODEL from .env (claude-sonnet-5-5)")
    ap.add_argument("--use-batch-api", action="store_true", help="force batch judging (50%% price)")
    ap.add_argument("--no-batch", action="store_true", help="judge with normal calls (faster start, full price)")
    ap.add_argument("--crosscheck-n", type=int, default=int(os.getenv("CROSSCHECK_N", "100") or 0),
                    help="re-judge this many answers with a second judge to measure label reliability (0 = off)")
    ap.add_argument("--crosscheck-model", default=os.getenv("CROSSCHECK_MODEL", "claude-sonnet-5-5"))
    ap.add_argument("--reuse-answers", action="store_true",
                    help="only judge/score answers already in this run (keeps the run's original answering model)")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--export-annotation", type=int, default=0)
    ap.add_argument("--agreement", default="")
    args = ap.parse_args()

    ws_id = args.workspace or ws.active_workspace_id()
    bench_dir = os.path.join(ws.workspace_dir(ws_id), "benchmark")
    run_dir = os.path.join(bench_dir, "runs", args.run_name)
    if args.agreement:
        return agreement(args.agreement, bench_dir)
    if args.export_annotation:
        return export_annotation(run_dir, args.export_annotation)

    be_nice()
    os.environ["EPISTEME_CPU_THREADS"] = str(args.cpu_threads)
    limit_cpu_threads(args.cpu_threads)
    os.makedirs(run_dir, exist_ok=True)
    items = load_items(args.bench or os.path.join(bench_dir, "benchmark.jsonl"))
    if args.limit:
        items = items[: args.limit]
    if not items:
        sys.exit("No benchmark items. Generate a benchmark or pass --bench <file>.")
    items_by_id = {it["id"]: it for it in items}
    pipe = EvaluationPipeline(use_mlflow=False, workspace_id=ws_id)
    pipe.trainer_config.n_jobs = args.cpu_threads
    judge_model = args.judge_model.strip() or model_for("judge")
    use_batch = (args.use_batch_api or provider_of(judge_model) == "anthropic") and not args.no_batch

    # One run = one audited model. Never mix answers of two different LLMs in one dataset.
    cur = {"answer": pipe.interceptor_config.llm_model_name, "extract": pipe.interceptor_config.extractor_model_name}
    rec = read_json(os.path.join(run_dir, "models.json"))
    if rec and args.reuse_answers:
        pipe.interceptor_config.llm_model_name, pipe.interceptor_config.extractor_model_name = rec["answer"], rec["extract"]
    elif rec and rec.get("answer") != cur["answer"] and _count(run_dir, "intercepts") > 0:
        slug = re.sub(r"[^A-Za-z0-9._-]", "_", rec["answer"])
        dest = os.path.join(bench_dir, "runs", f"{args.run_name}__{slug}")
        if os.path.exists(dest):
            dest += datetime.now().strftime("_%Y%m%d%H%M%S")
        os.replace(run_dir, dest)
        os.makedirs(run_dir, exist_ok=True)
        print(f"[run] earlier answers came from {rec['answer']}; they are kept in {dest}. "
              f"This run audits {cur['answer']} from scratch.", flush=True)
        rec = None
    if not rec:
        rec = cur
        write_json_atomic(os.path.join(run_dir, "models.json"), rec)
    write_json_atomic(os.path.join(run_dir, "config.json"), {**vars(args), "workspace": ws_id, "n_items": len(items),
                                                             "answer_model": rec["answer"], "judge_model": judge_model,
                                                             "batch_judge": use_batch,
                                                             "started_at": datetime.now(timezone.utc).isoformat()})
    prog = Progress(run_dir)
    print(f"[run] audited model {rec['answer']} | claim extractor {rec['extract']} | judge {judge_model}"
          f"{' (batch, 50% price)' if use_batch else ''}", flush=True)
    try:
        if not args.reuse_answers:
            stage_intercept(items, run_dir, pipe, args.workers, prog, args.mode)
        stage_judge(items, run_dir, pipe, judge_model, args.workers, prog, use_batch)
        xc = {}
        if args.crosscheck_n > 0 and args.crosscheck_model.strip():
            xc = stage_crosscheck(items, run_dir, pipe, judge_model, args.crosscheck_model.strip(), args.crosscheck_n, prog,
                                  use_batch, bench_dir)
        feats = attach_labels(stage_features(items, run_dir, pipe, prog, args.mode), run_dir)
        n_ans, n_jud = _count(run_dir, "intercepts"), _count(run_dir, "judgements")
        gaps = ""
        if n_ans < len(items) or n_jud < n_ans:
            gaps = (f"{len(items) - n_ans} questions still have no answer and {n_ans - n_jud} answers are not judged "
                    f"(transient API errors - see failures.json). Click Run again to retry only those.")
        if args.skip_train or feats.empty:
            prog.finish("completed", note="features only", warning=gaps or None)
            return
        tr = stage_train(feats, items_by_id, pipe, prog)
        if isinstance(tr, dict) and tr.get("skipped"):
            msg = f"Training not run yet. {tr['reason']} " + (gaps or "Every question is already processed, so "
                  "add more questions (e.g. generate extra FALSE_PREMISE / BORDERLINE items) and run again.")
            print(f"[train] {msg}", flush=True)
            prog.finish("completed", note="stopped before training", warning=msg)
            return
        summary = stage_report(feats, items_by_id, run_dir, pipe)
        if xc:
            summary["judge_crosscheck"] = {k: xc[k] for k in ("primary_judge", "second_judge", "n", "cohen_kappa_binary", "raw_agreement_binary")}
            if not (xc.get("cohen_kappa_binary", 0) >= 0.6):
                gaps = (gaps + " " if gaps else "") + (
                    f"Inter-judge agreement is low (kappa={xc.get('cohen_kappa_binary', float('nan')):.2f} on n={xc['n']}). "
                    f"Consider re-judging everything with {xc['second_judge']} (set JUDGE_MODEL in .env and click Run: "
                    "answers and features are cached, only judging is paid) - retraining itself costs nothing.")
        try:
            import export_paper_assets
            export_paper_assets.main(["--workspace", ws_id])
        except Exception as e:
            print(f"[report] paper asset export skipped: {e}", flush=True)
        # usage.json ACCUMULATES across runs of this dataset (a zero-cost retrain must not erase what the data cost)
        prev = read_json(os.path.join(run_dir, "usage.json"), default={}) or {}
        snap, per = usage_snapshot(), dict(prev.get("per_model") or {})
        for mname, u in snap["per_model"].items():
            acc = dict(per.get(mname) or {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0})
            for k in ("calls", "input_tokens", "output_tokens", "usd"):
                acc[k] = acc.get(k, 0) + u.get(k, 0)
            per[mname] = acc
        write_json_atomic(os.path.join(run_dir, "usage.json"), {"per_model": per, "usd_estimate": round(sum(v["usd"] for v in per.values()), 4),
                                                                 "ledger_total_usd": snap["ledger_total_usd"], "budget_usd": snap["budget_usd"]})
        prog.finish("completed", summary=summary, warning=gaps or None, usage=usage_snapshot())
    except RunStopped as e:
        print(f"[stopped] {e}", flush=True)
        prog.finish("failed", error=str(e))
        sys.exit(2)
    except Exception as e:
        import traceback
        traceback.print_exc()
        prog.finish("failed", error=friendly(e) + "\n" + _saved_line(run_dir, len(items)))
        sys.exit(1)


if __name__ == "__main__":
    main()

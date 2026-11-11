"""
Generate an N-query (default 500) hallucination benchmark from the ingested PDFs.

Categories (default mix for N = 500)
  FACTUAL        150  answerable from ONE chunk; gold answer + verbatim evidence quote
  CROSS_DOC       50  needs facts from chunks of TWO different papers
  BORDERLINE     100  on-topic, specific, but NOT answerable from the papers (ideal: say so)
  FALSE_PREMISE  200  embeds a premise contradicted by the chunk (ideal: reject the premise)

Quality controls
  * every item carries an evidence quote that must fuzzy-match its source chunk (ratio >= 0.85)
  * false premises are phrased naturally (no "Assume ..." template that a classifier could learn)
  * near-duplicates removed by MiniLM cosine >= 0.92
  * stratified trimming keeps papers balanced inside each category

CPU / cost
  * one API call yields `--per-call` questions; calls run in a small I/O thread pool
  * every call is disk-cached -> re-running is free
  * --use-batch-api submits everything as one batch (Claude or OpenAI; ~50% cheaper, zero local load)

Usage
  python scripts/generate_benchmark.py --n 500
  python scripts/generate_benchmark.py --n 500 --use-batch-api
Outputs
  <workspace>/benchmark/benchmark.jsonl         rich records (used by run_benchmark.py and the website)
  <workspace>/benchmark/benchmark_queries.txt   pipe format: ID | CATEGORY | prior | EXPECTED | question
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(__file__))
from _llm import ROOT, StageReport, be_nice, chat_json, run_batch_api, run_parallel  # noqa: E402
from kg_audit.utils.api_errors import FatalAPIError, friendly  # noqa: E402
from kg_audit.utils.llm import model_for  # noqa: E402

from kg_audit.utils import workspace as ws  # noqa: E402
from kg_audit.utils.jobs import JobStatus  # noqa: E402
from kg_audit.utils.main_utils import embed_many, limit_cpu_threads, read_json  # noqa: E402

MIX = {"FACTUAL": 0.30, "CROSS_DOC": 0.10, "BORDERLINE": 0.20, "FALSE_PREMISE": 0.40}
EXPECTED = {"FACTUAL": "ANSWER", "CROSS_DOC": "ANSWER", "BORDERLINE": "ABSTAIN", "FALSE_PREMISE": "REJECT_PREMISE"}
LEGACY_PRIOR = {"FACTUAL": (1, "VERIFIED"), "CROSS_DOC": (1, "VERIFIED"), "BORDERLINE": (1, "ABSTAIN"),
                "FALSE_PREMISE": (0, "HALLUCINATED")}

SYSTEM = "You write evaluation questions for a hallucination benchmark over scientific papers. Reply with one JSON object."

COMMON = """Rules for every question:
- Self-contained: name the method/model explicitly (e.g. "node2vec", "LINE", "DeepWalk"); never say
  "the passage", "the chunk", "paper1" or "this text".
- Vary the phrasing (why / how / what / compare / under which conditions ...). Do not start questions
  with "Assume".
- "evidence_quote" must be ONE sentence copied character-for-character from the PASSAGE.
- "difficulty": easy | medium | hard."""

PROMPTS = {
    "FACTUAL": """Write {k} questions that are fully answerable from the PASSAGE alone.
{common}
Return {{"items": [{{"question": "...", "gold_answer": "<2-3 sentence answer grounded in the passage>",
  "evidence_quote": "...", "difficulty": "..."}}]}}

PASSAGE ({source}):
\"\"\"{text}\"\"\"""",

    "CROSS_DOC": """Write {k} questions that can ONLY be answered by combining PASSAGE A and PASSAGE B
(e.g. contrast how the two methods handle the same concept).
{common}
Give one evidence quote from each passage.
Return {{"items": [{{"question": "...", "gold_answer": "...", "evidence_quote": "<from A>",
  "evidence_quote_b": "<from B>", "difficulty": "..."}}]}}

PASSAGE A ({source}):
\"\"\"{text}\"\"\"

PASSAGE B ({source_b}):
\"\"\"{text_b}\"\"\"""",

    "BORDERLINE": """Write {k} questions that are on the SAME topic as the PASSAGE and sound answerable,
but whose specific answer is NOT contained in it (e.g. ask for an exact number, ablation, dataset,
hardware detail or extension the authors never report). A faithful model should say the paper does
not provide this rather than invent it.
{common}
"evidence_quote" = the sentence the question is anchored to.
Return {{"items": [{{"question": "...", "why_unanswerable": "...", "evidence_quote": "...",
  "difficulty": "..."}}]}}

PASSAGE ({source}):
\"\"\"{text}\"\"\"""",

    "FALSE_PREMISE": """Write {k} questions that silently presuppose something the PASSAGE contradicts
(wrong mechanism, wrong objective, wrong dataset, wrong comparison result, swapped attribution, a
technique the method does not use ...). The false premise must be embedded naturally, e.g.
"Why does X's <wrong mechanism> make it ...", not announced. Mix subtle and blatant errors.
{common}
"evidence_quote" = the sentence that refutes the premise.
Return {{"items": [{{"question": "...", "false_premise": "...", "correction": "...",
  "evidence_quote": "...", "difficulty": "..."}}]}}

PASSAGE ({source}):
\"\"\"{text}\"\"\"""",
}


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(s).lower()).split())


def quote_ok(quote: str, text: str, thr: float = 0.85) -> bool:
    q, t = norm(quote), norm(text)
    if not q or len(q) < 20:
        return False
    if q in t:
        return True
    sents = re.split(r"(?<=[.!?])\s+", text)
    windows = [norm(s) for s in sents] + [norm(a + " " + b) for a, b in zip(sents, sents[1:])]
    return max((SequenceMatcher(None, q, w).ratio() for w in windows), default=0.0) >= thr


def plan_jobs(chunks: List[Dict[str, Any]], n: int, per_call: int, overgen: float, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        if len(c.get("text", "")) >= 400:
            by_doc[c["source_name"]].append(c)
    docs = sorted(by_doc)
    for d in docs:
        rng.shuffle(by_doc[d])
    cursor = {d: 0 for d in docs}

    def next_chunk(doc):
        lst = by_doc[doc]
        c = lst[cursor[doc] % len(lst)]
        cursor[doc] += 1
        return c

    jobs = []
    for cat, frac in MIX.items():
        calls = max(1, round(n * frac * overgen / per_call))
        for i in range(calls):
            doc = docs[i % len(docs)]
            c = next_chunk(doc)
            job = {"id": f"{cat}_{i:03d}", "category": cat, "chunk": c, "k": per_call}
            if cat == "CROSS_DOC":
                other = docs[(i + 1 + i // len(docs)) % len(docs)]
                if other == doc:
                    other = docs[(docs.index(doc) + 1) % len(docs)]
                job["chunk_b"] = next_chunk(other)
            jobs.append(job)
    return jobs


def build_messages(job: Dict[str, Any]) -> List[Dict[str, str]]:
    c, cb = job["chunk"], job.get("chunk_b") or {}
    user = PROMPTS[job["category"]].format(
        k=job["k"], common=COMMON, source=c["source_name"], text=c["text"],
        source_b=cb.get("source_name", ""), text_b=cb.get("text", ""))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def validate(job: Dict[str, Any], out: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = out.get("items", []) if isinstance(out, dict) else []
    keep = []
    for it in items:
        if not isinstance(it, dict) or len(str(it.get("question", ""))) < 25:
            continue
        cat, c = job["category"], job["chunk"]
        if not quote_ok(it.get("evidence_quote", ""), c["text"]):
            continue
        if cat == "CROSS_DOC" and not quote_ok(it.get("evidence_quote_b", ""), job["chunk_b"]["text"]):
            continue
        if cat == "FALSE_PREMISE" and not (it.get("false_premise") and it.get("correction")):
            continue
        if cat == "BORDERLINE" and not it.get("why_unanswerable"):
            continue
        if re.match(r"^\s*assume\b", it["question"], re.I):
            continue
        rec = {
            "category": cat, "question": " ".join(str(it["question"]).split()),
            "expected_behavior": EXPECTED[cat], "difficulty": it.get("difficulty", "medium"),
            "gold_answer": it.get("gold_answer", ""), "false_premise": it.get("false_premise", ""),
            "correction": it.get("correction", ""), "why_unanswerable": it.get("why_unanswerable", ""),
            "evidence_quotes": [it["evidence_quote"]] + ([it["evidence_quote_b"]] if cat == "CROSS_DOC" else []),
            "chunk_ids": [c["chunk_id"]] + ([job["chunk_b"]["chunk_id"]] if cat == "CROSS_DOC" else []),
            "source_names": [c["source_name"]] + ([job["chunk_b"]["source_name"]] if cat == "CROSS_DOC" else []),
        }
        keep.append(rec)
    return keep


def dedup(items: List[Dict[str, Any]], thr: float = 0.92) -> List[Dict[str, Any]]:
    if not items:
        return items
    E = embed_many([it["question"] for it in items])
    kept_idx: List[int] = []
    for i in range(len(items)):
        if not kept_idx or float((E[kept_idx] @ E[i]).max()) < thr:
            kept_idx.append(i)
    return [items[i] for i in kept_idx]


def stratified_trim(items: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    out = []
    for cat, frac in MIX.items():
        target = round(n * frac)
        pool = [it for it in items if it["category"] == cat]
        by_doc: Dict[str, List] = defaultdict(list)
        for it in pool:
            by_doc[it["source_names"][0]].append(it)
        for v in by_doc.values():
            rng.shuffle(v)
        chosen, docs = [], sorted(by_doc)
        while len(chosen) < target and any(by_doc.values()):
            for d in docs:
                if by_doc[d] and len(chosen) < target:
                    chosen.append(by_doc[d].pop())
        if len(chosen) < target:
            print(f"  [warn] {cat}: only {len(chosen)}/{target} valid items - raise --overgen", flush=True)
        out.extend(chosen)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=None, help="workspace id (default: active)")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--per-call", type=int, default=4)
    ap.add_argument("--overgen", type=float, default=1.35, help="over-generation factor before filtering")
    ap.add_argument("--workers", type=int, default=6, help="concurrent API calls (I/O-bound, ~0 CPU)")
    ap.add_argument("--model", default=model_for("generate"))
    ap.add_argument("--use-batch-api", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="", help="default: <workspace>/benchmark/benchmark.jsonl")
    args = ap.parse_args()

    be_nice()
    limit_cpu_threads(2)
    ws_id = args.workspace or ws.active_workspace_id()
    ws_dir = ws.workspace_dir(ws_id)
    out = args.out or os.path.join(ws_dir, "benchmark", "benchmark.jsonl")
    job = JobStatus(os.getenv("EPISTEME_JOB_DIR"))
    chunks = read_json(os.path.join(ws_dir, "latest_chunks.json"), default=[]) or []
    if not chunks:
        job.finish("failed", error="no corpus in this workspace - build the knowledge graph first")
        sys.exit("No corpus found: build the knowledge graph first (Phase 1).")

    try:
        jobs = plan_jobs(chunks, args.n, args.per_call, args.overgen, args.seed)
        print(f"[gen] {len(jobs)} generation calls for ~{args.n} items with {args.model}", flush=True)
        for j in jobs:
            j["messages"] = build_messages(j)
        job.update("generate", 0, len(jobs))

        if args.use_batch_api:
            job.update("generate", 0, len(jobs), note="submitted to the Batch API (50% price)")
            outs = run_batch_api([{"id": j["id"], "messages": j["messages"]} for j in jobs], args.model, tag="bench_gen")
        else:
            done = [0]

            def tick(_j, _r):
                done[0] += 1
                job.update("generate", done[0], len(jobs))
            rep = StageReport(len(jobs))
            outs = run_parallel(jobs, lambda j: chat_json(args.model, j["messages"], tag="bench_gen", temperature=0.7),
                                workers=args.workers, desc="generate", on_result=tick, report=rep)
            if rep.fatal:   # never overwrite an existing benchmark with a partial one
                raise FatalAPIError(f"{rep.fatal}\n{rep.ok}/{len(jobs)} generation calls finished and are cached; "
                                    "click Generate again after fixing this - they will not be re-paid. "
                                    "The existing benchmark file was left untouched.")
            if rep.failed > 0.2 * len(jobs):
                raise RuntimeError(f"{rep.failed}/{len(jobs)} generation calls failed ({rep.summary()}). "
                                   "Click Generate again: successful calls are cached. Existing benchmark left untouched.")

        job.update("validate", len(jobs), len(jobs), note="evidence check + de-duplication")
        items = []
        for j, o in zip(jobs, outs):
            if isinstance(o, Exception):
                print(f"  [skip] {j['id']}: {o}", flush=True)
                continue
            items.extend(validate(j, o))
        print(f"[gen] {len(items)} items passed evidence validation", flush=True)
        items = dedup(items)
        print(f"[gen] {len(items)} after de-duplication", flush=True)
        items = stratified_trim(items, args.n, args.seed)
        random.Random(args.seed).shuffle(items)
        for i, it in enumerate(sorted(items, key=lambda r: list(MIX).index(r["category"])), start=1):
            it["id"] = f"B{i:04d}"

        if not items:
            raise RuntimeError("No generated item passed evidence validation; the existing benchmark was left untouched.")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        items.sort(key=lambda r: r["id"])
        with open(out, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        with open(out.replace(".jsonl", "_queries.txt"), "w", encoding="utf-8") as f:
            for it in items:
                prior, exp = LEGACY_PRIOR[it["category"]]
                f.write(f"{it['id']} | {it['category']} | {prior} | {exp} | {it['question']}\n")

        counts = defaultdict(int)
        for it in items:
            counts[it["category"]] += 1
        print(f"[gen] wrote {len(items)} items -> {out}\n      {dict(counts)}", flush=True)
        job.finish("completed", result={"path": out, "n": len(items), "by_category": dict(counts)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        job.finish("failed", error=friendly(e))
        raise


if __name__ == "__main__":
    main()

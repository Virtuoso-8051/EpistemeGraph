"""
CPU-light LLM request layer shared by the benchmark scripts.

Why the CPU stayed pegged before: every query re-instantiated the pipeline, re-loaded models,
re-downloaded the whole KG, retrained 9 models every 3 queries (GridSearchCV n_jobs=-1) and
ran torch with all cores. API calls themselves are I/O-bound and cost ~0 CPU.

This module therefore:
  * runs API calls in a small thread pool (threads sleep on the socket -> ~0% CPU),
  * caches every response on disk (sha256 of model+messages) -> re-runs cost $0 and no CPU,
  * retries with exponential backoff + jitter on 429/5xx,
  * optionally submits everything through the OpenAI Batch API (50% cheaper, zero local load),
  * lowers the process priority so the machine stays responsive.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv  # noqa: E402

from kg_audit.utils.api_errors import FatalAPIError, as_fatal, classify, friendly, retry_delay  # noqa: E402
from kg_audit.utils.llm import (_record, anthropic_params, check_budget, default_client, estimate_batch_usd,  # noqa: E402
                                parse_json, provider_of)
from kg_audit.utils.main_utils import read_json, sha256_text, write_json_atomic  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))
CACHE_DIR = os.path.join(ROOT, "Artifacts", "script_cache")


def be_nice() -> None:
    """Lower process priority (Windows: BELOW_NORMAL, POSIX: nice +10)."""
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass


def _openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=90.0, max_retries=0)


def _key(model: str, messages: List[Dict[str, str]], tag: str) -> str:
    return sha256_text(json.dumps([tag, model, messages], sort_keys=True))


def _cache_path(model: str, messages: List[Dict[str, str]], tag: str) -> str:
    return os.path.join(CACHE_DIR, tag or "misc", _key(model, messages, tag) + ".json")


def _split(messages: List[Dict[str, str]]):
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
    user = "\n\n".join(m["content"] for m in messages if m.get("role") != "system")
    return system, user


def _parse_json(text: str) -> Dict[str, Any]:
    return parse_json(text)


def chat_json(model: str, messages: List[Dict[str, str]], tag: str = "", temperature: Optional[float] = 0.0,
              max_attempts: int = 10, max_tokens: int = 12000) -> Dict[str, Any]:
    """
    One cached JSON completion (Claude or OpenAI, chosen by the model name).
    Fatal errors (no credits, bad key, unknown model) raise FatalAPIError immediately; rate limits and
    overload back off honouring Retry-After; malformed JSON is retried once. Cached on disk forever.
    """
    path = _cache_path(model, messages, tag)
    hit = read_json(path)
    if hit is not None:
        return hit
    system, user = _split(messages)
    out, _meta = default_client().complete_json(model, system, user, temperature=temperature,
                                                max_tokens=max_tokens, max_attempts=max_attempts)
    write_json_atomic(path, out)
    return out


class SkippedAfterFatal(FatalAPIError):
    """Raised for jobs never sent because an earlier job hit a fatal error."""


class StageReport:
    """What a fan-out stage actually achieved - never 'done' when requests silently failed."""

    def __init__(self, n_jobs: int):
        self.n_jobs, self.ok, self.failed, self.skipped = n_jobs, 0, 0, 0
        self.errors: Dict[str, int] = {}
        self.fatal: Optional[str] = None
        self.failed_ids: List[str] = []

    def add_error(self, job_id: str, e: BaseException, skipped: bool = False) -> None:
        if skipped:
            self.skipped += 1
        else:
            self.failed += 1
            key = friendly(e)[:300]
            self.errors[key] = self.errors.get(key, 0) + 1
        self.failed_ids.append(job_id)

    def summary(self) -> str:
        s = f"{self.ok} ok, {self.failed} failed, {self.skipped} skipped of {self.n_jobs}"
        if self.errors:
            top = max(self.errors, key=self.errors.get)
            s += f" | most common error ({self.errors[top]}x): {top}"
        return s

    def to_dict(self) -> Dict[str, Any]:
        return {"n_jobs": self.n_jobs, "ok": self.ok, "failed": self.failed, "skipped": self.skipped,
                "fatal": self.fatal, "errors": self.errors, "failed_ids": self.failed_ids[:1000]}


def run_parallel(jobs: List[Dict[str, Any]], fn: Callable[[Dict[str, Any]], Any], workers: int = 6,
                 desc: str = "requests", on_result: Optional[Callable[[Dict[str, Any], Any], None]] = None,
                 report: Optional[StageReport] = None) -> List[Any]:
    """
    Thread-pool fan-out for I/O-bound jobs with a live counter. Failures are returned as exceptions
    (and tallied in `report`). The first FATAL error (e.g. no credits) trips a stop flag: every job not
    yet started is skipped instantly instead of hitting the dead account, and the stage ends in seconds.
    """
    import threading
    results: List[Any] = [None] * len(jobs)
    rep = report or StageReport(len(jobs))
    stop = threading.Event()
    t0 = time.time()

    def guarded(j):
        if stop.is_set():
            raise SkippedAfterFatal(rep.fatal or "stopped after a fatal API error")
        try:
            return fn(j)
        except Exception as e:
            if classify(e) == "fatal" and not stop.is_set():
                rep.fatal = friendly(e)
                stop.set()
                print(f"  [{desc}] FATAL - stopping this stage: {rep.fatal}", flush=True)
            raise

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        fut = {ex.submit(guarded, j): i for i, j in enumerate(jobs)}
        for done, f in enumerate(as_completed(fut), start=1):
            i = fut[f]
            try:
                results[i] = f.result()
                rep.ok += 1
            except Exception as e:  # keep going; caller decides
                results[i] = e
                rep.add_error(str(jobs[i].get("id", i)) if isinstance(jobs[i], dict) else str(i), e,
                              skipped=isinstance(e, SkippedAfterFatal))
            if on_result:
                try:
                    on_result(jobs[i], results[i])
                except Exception as cb:          # a progress-file hiccup must never kill the stage
                    print(f"  [{desc}] progress callback error ignored: {cb}", flush=True)
            if done % 10 == 0 or done == len(jobs):
                rate = done / max(1e-6, time.time() - t0)
                print(f"  [{desc}] {done}/{len(jobs)}  ({rate:.1f}/s)  ok={rep.ok} failed={rep.failed} skipped={rep.skipped}", flush=True)
    if jobs:
        print(f"  [{desc}] {rep.summary()}", flush=True)
    return results


# --------------------------------------------------------------------------- Batch API path
def run_batch_api(jobs: List[Dict[str, Any]], model: str, tag: str, poll_s: int = 20,
                  progress: Optional[Callable[[int, int, str], None]] = None) -> List[Dict[str, Any]]:
    """
    jobs: [{"id": str, "messages": [...]}]. Uncached jobs go out as ONE asynchronous batch at 50 % of the
    normal price (Anthropic Message Batches or OpenAI Batch API, by model name). Results land in the same
    disk cache chat_json() reads, so the rest of the pipeline is identical. The batch id is saved to disk:
    if this process is stopped, the next run resumes polling the SAME batch (nothing is paid twice).
    Anything the batch could not answer is retried with normal calls at the end.
    """
    if provider_of(model) == "anthropic":
        _anthropic_batch(jobs, model, tag, poll_s, progress)
    else:
        _openai_batch(jobs, model, tag, poll_s)
    leftover = [j for j in jobs if read_json(_cache_path(model, j["messages"], tag)) is None]
    if leftover:
        print(f"  [batch] {len(leftover)} request(s) not answered by the batch - sending them directly", flush=True)
        rep = StageReport(len(leftover))
        run_parallel(leftover, lambda j: chat_json(model, j["messages"], tag=tag), workers=4, desc="batch-fallback",
                     report=rep)
        if rep.fatal:
            raise FatalAPIError(rep.fatal)
    return [read_json(_cache_path(model, j["messages"], tag), default={}) for j in jobs]


def _anthropic_batch(jobs, model, tag, poll_s, progress) -> None:
    import anthropic
    state_path = os.path.join(CACHE_DIR, "batches", f"{tag}_{model}_pending.json")
    by_cid = {_key(model, j["messages"], tag)[:64]: j for j in jobs}
    state = read_json(state_path)
    try:
        client = default_client()._sdk("anthropic")
        if not state:
            pending = [j for j in jobs if read_json(_cache_path(model, j["messages"], tag)) is None]
            if not pending:
                return
            chars = sum(len(m["content"]) for j in pending for m in j["messages"])
            check_budget(estimate_batch_usd(model, chars, len(pending)))     # refuse BEFORE paying for anything
            reqs = []
            for j in pending:
                system, user = _split(j["messages"])
                params = anthropic_params(model, system, user, 12000,
                                          "low")
                reqs.append({"custom_id": _key(model, j["messages"], tag)[:64], "params": params})
            batch = client.messages.batches.create(requests=reqs)
            state = {"batch_id": batch.id, "n": len(reqs), "model": model, "created": time.time()}
            write_json_atomic(state_path, state)
            print(f"  [batch] submitted {len(reqs)} requests to Anthropic Message Batches ({batch.id}), 50% price", flush=True)
        else:
            print(f"  [batch] resuming earlier batch {state['batch_id']} (not re-submitted, not re-paid)", flush=True)
        t0 = time.time()
        while True:
            b = client.messages.batches.retrieve(state["batch_id"])
            rc = b.request_counts
            done = int(rc.succeeded + rc.errored + rc.canceled + rc.expired)
            note = f"Anthropic batch {b.processing_status}: {done}/{state['n']} done ({int(time.time() - t0)}s)"
            print(f"  [batch] {note}", flush=True)
            if progress:
                try:
                    progress(done, state["n"], note)
                except Exception:
                    pass
            if b.processing_status == "ended":
                break
            time.sleep(poll_s)
        n_ok = 0
        for r in client.messages.batches.results(state["batch_id"]):
            j = by_cid.get(r.custom_id)
            if j is None or r.result.type != "succeeded":
                continue
            msg = r.result.message
            u = getattr(msg, "usage", None)
            _record(model, int(getattr(u, "input_tokens", 0) or 0), int(getattr(u, "output_tokens", 0) or 0), batch=True)
            text = "".join(c.text for c in msg.content if getattr(c, "type", "") == "text")
            try:
                write_json_atomic(_cache_path(model, j["messages"], tag), parse_json(text))
                n_ok += 1
            except Exception:
                pass
        print(f"  [batch] {n_ok} results stored in the cache", flush=True)
        try:
            os.remove(state_path)
        except OSError:
            pass
    except anthropic.NotFoundError:
        try:
            os.remove(state_path)     # stale id (e.g. results older than 29 days) - fall back to direct calls
        except OSError:
            pass
    except Exception as e:
        if classify(e) == "fatal":
            raise as_fatal(e) from e
        raise


def _openai_batch(jobs, model, tag, poll_s) -> None:
    client = _openai_client()
    pending = [j for j in jobs if read_json(_cache_path(model, j["messages"], tag)) is None]
    if not pending:
        return
    os.makedirs(os.path.join(CACHE_DIR, "batches"), exist_ok=True)
    jl = os.path.join(CACHE_DIR, "batches", f"{tag}_{int(time.time())}.jsonl")
    with open(jl, "w", encoding="utf-8") as f:
        for j in pending:
            f.write(json.dumps({"custom_id": j["id"], "method": "POST", "url": "/v1/chat/completions",
                                "body": {"model": model, "messages": j["messages"],
                                         "response_format": {"type": "json_object"}}}) + "\n")
    try:
        batch_file = client.files.create(file=open(jl, "rb"), purpose="batch")
        batch = client.batches.create(input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h")
    except Exception as e:
        if classify(e) == "fatal":
            raise as_fatal(e) from e
        raise
    print(f"  [batch] submitted {len(pending)} requests as {batch.id}; polling every {poll_s}s ...", flush=True)
    while batch.status not in ("completed", "failed", "expired", "cancelled"):
        time.sleep(poll_s)
        batch = client.batches.retrieve(batch.id)
        rc = batch.request_counts
        print(f"  [batch] {batch.status}: {getattr(rc, 'completed', 0)}/{getattr(rc, 'total', 0)}", flush=True)
    if batch.status != "completed" or not batch.output_file_id:
        print(f"  [batch] {batch.id} ended with status {batch.status}", flush=True)
        return
    by_id = {j["id"]: j for j in pending}
    for line in client.files.content(batch.output_file_id).text.splitlines():
        rec = json.loads(line)
        j = by_id.get(rec.get("custom_id"))
        body = (rec.get("response") or {}).get("body") or {}
        if j and body.get("choices"):
            try:
                write_json_atomic(_cache_path(model, j["messages"], tag), parse_json(body["choices"][0]["message"]["content"]))
            except Exception:
                pass

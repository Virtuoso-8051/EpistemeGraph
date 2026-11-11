"""
Provider-neutral JSON completion used by every phase (KG extraction, interception, judging, generation).

The provider is chosen by the MODEL NAME, not by a global switch:
    claude-*  -> Anthropic Messages API (default for everything new)
    gpt-*/o*  -> OpenAI Chat Completions (kept so cached OpenAI-era answers stay readable)
so a run that was answered by an OpenAI model can still be read from cache while new calls go to Claude.

Roles and their default models (override any of them in .env):
    answer     LLM_MODEL        claude-haiku-5-5    the model UNDER AUDIT (cheap, fast, hallucinates enough)
    extract    EXTRACTOR_MODEL  claude-haiku-5-5    claim extraction from an answer
    kg         KG_MODEL         claude-haiku-5-5    triple extraction from PDF chunks
    judge      JUDGE_MODEL      claude-haiku-5-5    benchmark labels + KG precision audit (the judge SEES the gold evidence)
    generate   GENERATOR_MODEL  claude-haiku-5-5    benchmark question generation
  A stronger judge (claude-sonnet-5-5, ~15-20x the cost) is optional: set JUDGE_MODEL in .env.

Every call:
  * fatal errors (no credits, bad key, unknown model) raise FatalAPIError at once - never retried;
  * rate limits / overload / 5xx back off honouring Retry-After (rate-limit waits cost nothing);
  * a reply that is not valid JSON, or that hit max_tokens, is retried once (with more room);
  * token usage is accumulated per model so scripts can report what a run cost.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

from kg_audit.utils.api_errors import FatalAPIError, as_fatal, classify, retry_delay

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(_ROOT, ".env"))

# ------------------------------------------------------------------------------------------ models
_ANTHROPIC_DEFAULTS = {
    "answer": "claude-haiku-5-5", "extract": "claude-haiku-5-5", "kg": "claude-haiku-5-5",
    "judge": "claude-haiku-5-5", "generate": "claude-haiku-5-5",
}
_ENV = {"answer": "LLM_MODEL", "extract": "EXTRACTOR_MODEL", "kg": "KG_MODEL",
        "judge": "JUDGE_MODEL", "generate": "GENERATOR_MODEL"}
_LEGACY_OPENAI_ENV = {"answer": "OPENAI_MODEL", "extract": "OPENAI_EXTRACTOR_MODEL", "kg": "OPENAI_MODEL",
                      "judge": "OPENAI_JUDGE_MODEL", "generate": "OPENAI_GENERATOR_MODEL"}

# USD per 1M tokens (input, output), first-party list prices; batch = 50 %.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00), "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5": (2.00, 10.00), "claude-opus-5": (5.00, 25.00), "claude-haiku-4-5": (1.00, 5.00),
}


def default_provider() -> str:
    p = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if p in ("anthropic", "claude"):
        return "anthropic"
    if p == "openai":
        return "openai"
    return "openai" if (os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY")) else "anthropic"


def model_for(role: str) -> str:
    """Model id for a pipeline role, from .env or the provider defaults."""
    v = (os.getenv(_ENV[role]) or "").strip()
    if v:
        return v
    if default_provider() == "openai":
        legacy = os.getenv(_LEGACY_OPENAI_ENV[role]) or os.getenv("OPENAI_MODEL") or "gpt-5.4-mini"
        return legacy.strip()
    return _ANTHROPIC_DEFAULTS[role]


def provider_of(model: str) -> str:
    return "anthropic" if str(model).lower().startswith("claude") else "openai"


def key_status() -> Dict[str, bool]:
    return {"anthropic": bool((os.getenv("ANTHROPIC_API_KEY") or "").strip()),
            "openai": bool((os.getenv("OPENAI_API_KEY") or "").strip())}


# ------------------------------------------------------------------------------------------ usage + spend cap
_USAGE: Dict[str, Dict[str, float]] = {}
_USAGE_LOCK = threading.Lock()
_LEDGER = os.path.join(_ROOT, "Artifacts", "spend_ledger.json")


def budget_usd() -> Optional[float]:
    """Hard cap on the TOTAL estimated spend recorded in the ledger (all runs). Unset / 0 = no cap."""
    try:
        v = float(os.getenv("EPISTEME_BUDGET_USD", "") or 0)
    except ValueError:
        return None
    return v if v > 0 else None


def ledger_total() -> float:
    try:
        with open(_LEDGER, "r", encoding="utf-8") as f:
            return float(json.load(f).get("usd", 0.0))
    except Exception:
        return 0.0


def _ledger_add(usd: float) -> None:
    if usd <= 0:
        return
    try:
        os.makedirs(os.path.dirname(_LEDGER), exist_ok=True)
        tot = ledger_total() + usd
        tmp = _LEDGER + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"usd": round(tot, 5), "note": "estimated from token counts at list prices; delete this file to reset"}, f)
        os.replace(tmp, _LEDGER)
    except Exception:
        pass


def _cost(model: str, inp: int, out: int, batch: bool = False) -> float:
    p = PRICES.get(model)
    return (inp * p[0] + out * p[1]) / 1e6 * (0.5 if batch else 1.0) if p else 0.0


def check_budget(extra_usd: float = 0.0) -> None:
    """Raise FatalAPIError BEFORE a call/batch if it would push the recorded spend over the cap."""
    cap = budget_usd()
    if cap is None:
        return
    spent = ledger_total()
    if spent + extra_usd > cap:
        raise FatalAPIError(
            f"Spend cap reached: ~${spent:.2f} already spent (cap ${cap:.2f}"
            f"{', this step needs about $%.2f more' % extra_usd if extra_usd else ''}). Nothing was lost - finished work is "
            f"cached. Raise EPISTEME_BUDGET_USD in .env to continue, or delete Artifacts/spend_ledger.json to reset the counter.")


def estimate_batch_usd(model: str, texts_chars: int, n: int, out_tokens_each: int = 900) -> float:
    return _cost(model, int(texts_chars / 3.3), n * out_tokens_each, batch=True)


def _record(model: str, inp: int, out: int, batch: bool = False) -> None:
    usd = _cost(model, inp, out, batch)
    with _USAGE_LOCK:
        u = _USAGE.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0})
        u["calls"] += 1
        u["input_tokens"] += inp
        u["output_tokens"] += out
        u["usd"] += usd
        _ledger_add(usd)


def usage_snapshot() -> Dict[str, Any]:
    with _USAGE_LOCK:
        per = {m: dict(v) for m, v in _USAGE.items()}
    return {"per_model": per, "usd_estimate": round(sum(v["usd"] for v in per.values()), 4),
            "ledger_total_usd": round(ledger_total(), 4), "budget_usd": budget_usd()}


# ------------------------------------------------------------------------------------------ parsing
_JSON_RULE = ("\n\nOutput format: reply with exactly one JSON object and nothing else - no markdown fences, "
              "no text before or after it.")


def parse_json(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S).strip()
    try:
        out = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            raise
        out = json.loads(m.group(0))
    if not isinstance(out, dict):
        raise ValueError("model reply is JSON but not an object")
    return out


class _Truncated(ValueError):
    pass


# ------------------------------------------------------------------------------------------ requests
def anthropic_params(model: str, system: str, user: str, max_tokens: int, effort: Optional[str] = None) -> Dict[str, Any]:
    """Request body for one JSON task (shared by direct calls and the Message Batches API)."""
    params: Dict[str, Any] = {
        "model": model, "max_tokens": int(max_tokens),
        "system": (system or "") + _JSON_RULE,
        "messages": [{"role": "user", "content": user}],
    }
    if effort:
        params["output_config"] = {"effort": effort}
    return params


def _effort_for(model: str, role_hint: Optional[str]) -> Optional[str]:
    # extraction / answering on Haiku: light thinking keeps it fast and cheap;
    # judging on Sonnet: default ("high") thinking for careful verdicts.
    if "haiku" in model:
        return "low"
    if "sonnet" in model or "opus" in model:
        return "low"         # hidden thinking is billed as output: keep it small
    return None


class LLMClient:
    """One object per process (thread-safe); `api_key` overrides the .env key (website bring-your-own-key)."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 120.0):
        self.api_key = (api_key or "").strip() or None
        self.timeout = timeout
        self._clients: Dict[str, Any] = {}
        self._lock = threading.Lock()

    # -- raw SDK clients (lazy, never retried by the SDK: retries live here)
    def _sdk(self, provider: str):
        with self._lock:
            if provider in self._clients:
                return self._clients[provider]
            if provider == "anthropic":
                key = self.api_key if (self.api_key or "").startswith("sk-ant") else os.getenv("ANTHROPIC_API_KEY", "").strip()
                if not key:
                    raise FatalAPIError("ANTHROPIC_API_KEY is empty. Open the .env file in the project folder, paste your "
                                        "Claude API key after 'ANTHROPIC_API_KEY=', save, and restart the website.")
                import anthropic
                c = anthropic.Anthropic(api_key=key, timeout=self.timeout, max_retries=0)
            else:
                key = (self.api_key if self.api_key and not self.api_key.startswith("sk-ant") else None) \
                    or os.getenv("OPENAI_API_KEY", "").strip()
                if not key:
                    raise FatalAPIError("This step needs an OpenAI model but OPENAI_API_KEY is empty. Set the model "
                                        "names in .env to Claude models (claude-...) instead.")
                from openai import OpenAI
                c = OpenAI(api_key=key, timeout=self.timeout, max_retries=0)
            self._clients[provider] = c
            return c

    def complete_json(self, model: str, system: str, user: str, *, temperature: Optional[float] = None,
                      logprobs: bool = False, max_tokens: int = 8000, effort: Optional[str] = None,
                      max_attempts: int = 10) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Returns (parsed JSON object, meta{latency_ms, prompt_tokens, completion_tokens, token_nll})."""
        provider = provider_of(model)
        check_budget()
        bad_json = 0
        drop: set = set()
        last: Optional[BaseException] = None
        for attempt in range(max_attempts):
            t0 = time.perf_counter()
            try:
                if provider == "anthropic":
                    text, meta = self._anthropic(model, system, user, max_tokens, effort if effort is not None
                                                 else _effort_for(model, None), drop)
                else:
                    text, meta = self._openai(model, system, user, temperature, logprobs, drop)
                meta["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
                return parse_json(text), meta
            except _Truncated as e:                    # ran out of room: retry once with double max_tokens
                last = e
                if bad_json >= 1:
                    raise ValueError(f"{model}: reply truncated at max_tokens={max_tokens}") from e
                bad_json += 1
                max_tokens = min(64000, max_tokens * 2)
            except (json.JSONDecodeError, ValueError) as e:
                last = e
                if bad_json >= 1:
                    raise
                bad_json += 1
            except FatalAPIError:
                raise
            except Exception as e:
                last = e
                msg = str(e).lower()
                fixed = False
                for p in ("temperature", "logprobs", "output_config", "effort"):
                    if p in msg and p not in drop and classify(e) != "fatal":
                        drop.add(p)
                        fixed = True
                if fixed:
                    continue
                kind = classify(e)
                if kind == "fatal":
                    raise as_fatal(e) from e
                if kind != "retryable" or attempt == max_attempts - 1:
                    raise
                time.sleep(retry_delay(e, min(attempt, 5), cap=60.0))
        raise last or RuntimeError("no attempt made")

    def _anthropic(self, model, system, user, max_tokens, effort, drop) -> Tuple[str, Dict[str, Any]]:
        params = anthropic_params(model, system, user, max_tokens,
                                  None if ("output_config" in drop or "effort" in drop) else effort)
        resp = self._sdk("anthropic").messages.create(**params)
        u = getattr(resp, "usage", None)
        inp, out = int(getattr(u, "input_tokens", 0) or 0), int(getattr(u, "output_tokens", 0) or 0)
        _record(model, inp, out)
        if resp.stop_reason == "refusal":
            raise ValueError(f"{model} declined this request (refusal)")
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        if resp.stop_reason == "max_tokens":
            raise _Truncated("max_tokens")
        return text, {"cache_hit": False, "prompt_tokens": inp, "completion_tokens": out, "token_nll": None}

    def _openai(self, model, system, user, temperature, logprobs, drop) -> Tuple[str, Dict[str, Any]]:
        import numpy as np
        kw: Dict[str, Any] = dict(model=model, response_format={"type": "json_object"},
                                  messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        if temperature is not None and "temperature" not in drop:
            kw["temperature"] = temperature
        if logprobs and "logprobs" not in drop:
            kw["logprobs"] = True
        resp = self._sdk("openai").chat.completions.create(**kw)
        u = getattr(resp, "usage", None)
        inp, out = int(getattr(u, "prompt_tokens", 0) or 0), int(getattr(u, "completion_tokens", 0) or 0)
        _record(model, inp, out)
        nll = None
        if kw.get("logprobs"):
            lp = getattr(resp.choices[0], "logprobs", None)
            toks = [t.logprob for t in (getattr(lp, "content", None) or []) if t.logprob is not None]
            nll = float(-np.mean(toks)) if toks else None
        return (resp.choices[0].message.content or ""), {"cache_hit": False, "prompt_tokens": inp,
                                                          "completion_tokens": out, "token_nll": nll}


_DEFAULT: Optional[LLMClient] = None


def default_client() -> LLMClient:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = LLMClient()
    return _DEFAULT

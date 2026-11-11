"""
One place that decides what an LLM-API failure means, so no stage ever wastes time retrying a
request that cannot succeed, and no failure is ever swallowed silently.

  fatal      -> stop the whole stage at once (no credits, bad key, unknown model, permission)
  retryable  -> back off and try again (rate limit, 5xx, timeout, dropped connection)
  other      -> fail this one request only (malformed output, context too long, ...)

Detection is by status code AND by message text, because wrappers such as KGAuditException keep
the original message but not the original exception type.
"""
from __future__ import annotations

import random
from typing import Optional

_FATAL_MARKERS = (
    "credit balance is too low", "purchase credits", "authentication_error", "invalid x-api-key",
    "anthropic_api_key is empty", "openai_api_key is empty", "not_found_error",
    "insufficient_quota", "credit_balance_exhausted", "no credits remaining", "exceeded your current quota",
    "billing_hard_limit", "billing hard limit", "account_deactivated", "invalid_api_key", "incorrect api key",
    "no api key", "api_key client option must be set", "model_not_found", "does not exist or you do not have access",
    "permission_denied", "unsupported_country", "organization has been disabled",
)
_RETRY_MARKERS = (
    "overloaded", "rate_limit", "rate limit", "too many requests", "timeout", "timed out", "temporarily", "overloaded",
    "server_error", "bad gateway", "service unavailable", "connection", "502", "503", "504",
)


class FatalAPIError(RuntimeError):
    """A failure no retry can fix. Carries a plain-English, actionable message."""


def _status(e: BaseException) -> Optional[int]:
    for attr in ("status_code", "http_status", "status"):
        v = getattr(e, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(e, "response", None)
    v = getattr(resp, "status_code", None)
    return v if isinstance(v, int) else None


def classify(e: BaseException) -> str:
    if isinstance(e, FatalAPIError):
        return "fatal"
    msg = str(e).lower()
    if any(m in msg for m in _FATAL_MARKERS):
        return "fatal"
    code = _status(e)
    if code in (401, 403, 404):
        return "fatal"
    if code in (408, 409, 429) or (code is not None and code >= 500) or any(m in msg for m in _RETRY_MARKERS):
        return "retryable"
    if type(e).__name__ in ("APIConnectionError", "APITimeoutError", "ConnectionError", "ReadTimeout"):
        return "retryable"
    return "other"


def is_fatal(e: BaseException) -> bool:
    return classify(e) == "fatal"


def friendly(e: BaseException) -> str:
    """Actionable one-paragraph explanation for the website / log."""
    if isinstance(e, FatalAPIError):
        return str(e)
    msg = str(e).lower()
    if "credit balance is too low" in msg or "purchase credits" in msg:
        return ("Claude API credits exhausted: Anthropic answered 'credit balance is too low'. Add credits at "
                "console.anthropic.com > Settings > Billing, then click Run again. Everything already finished is "
                "cached on disk and will NOT be paid for twice.")
    if "authentication_error" in msg or "invalid x-api-key" in msg:
        return ("Anthropic rejected the API key. Open .env, check the line ANTHROPIC_API_KEY=sk-ant-... (no quotes, "
                "no spaces), save, and restart the website.")
    if "not_found_error" in msg and "model" in msg:
        return "This Claude model id is not available to your key. Check LLM_MODEL / JUDGE_MODEL in .env."
    if any(m in msg for m in ("insufficient_quota", "credit_balance_exhausted", "no credits remaining",
                              "exceeded your current quota", "billing")):
        return ("OpenAI credits exhausted: the API answered 'no credits remaining'. Add credits at "
                "platform.openai.com > Settings > Billing, then click Run again. Everything already finished is "
                "cached on disk and will NOT be paid for twice.")
    if any(m in msg for m in ("invalid_api_key", "incorrect api key", "no api key", "api_key client option")):
        return "OpenAI rejected the API key. Check OPENAI_API_KEY in .env (or the key pasted in the website), then run again."
    if any(m in msg for m in ("model_not_found", "does not exist or you do not have access")):
        return "The configured OpenAI model is not available to this key. Check OPENAI_MODEL / the judge model name, then run again."
    if "permission" in msg or _status(e) == 403:
        return "OpenAI refused the request (permission / region). Check the key's project permissions, then run again."
    return str(e)[:400]


def as_fatal(e: BaseException) -> FatalAPIError:
    return e if isinstance(e, FatalAPIError) else FatalAPIError(friendly(e))


def retry_delay(e: BaseException, attempt: int, cap: float = 30.0) -> float:
    """Honour the server's Retry-After header when present, else exponential backoff with jitter."""
    resp = getattr(e, "response", None)
    headers = getattr(resp, "headers", None) or {}
    try:
        ra = float(headers.get("retry-after") or headers.get("Retry-After") or 0)
        if ra > 0:
            return min(cap, ra + random.random())
    except (TypeError, ValueError):
        pass
    return min(cap, (2 ** attempt) + random.random())

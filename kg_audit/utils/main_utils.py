"""
Shared utilities for EpistemeGraph.

* CPU governance  : caps BLAS / OpenMP / torch intra-op threads so that batch
                    evaluation of hundreds of queries never saturates every core.
* Embedding cache : one process-wide SentenceTransformer with an LRU cache, so a
                    string (entity name, relation, claim) is encoded at most once.
* Text helpers    : deterministic normalisation / tokenisation used by every phase.
* JSON helpers    : atomic writes (no half-written state files if Streamlit reruns).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import unicodedata
from functools import lru_cache
from typing import Any, Iterable, List, Optional

import numpy as np

# --------------------------------------------------------------------------------------
# CPU governance
# --------------------------------------------------------------------------------------
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def default_cpu_budget() -> int:
    """Half the logical cores (min 1, max 4): leaves the machine responsive."""
    env = os.getenv("EPISTEME_CPU_THREADS")
    if env and env.isdigit():
        return max(1, int(env))
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def limit_cpu_threads(n_threads: Optional[int] = None) -> int:
    """
    Must be called *before* heavy numerical work. Safe to call repeatedly.
    Returns the number of threads actually configured.
    """
    n = n_threads or default_cpu_budget()
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, str(n))
    try:
        import torch
        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # can only be set once per process
    except Exception:
        pass
    return n


# --------------------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset("""
a an the and or of for with from that this these those into over under how what why which when
where who whom does do did is are was were be been being can could would should may might must
will shall it its they them their there here than then also such as at by on in to via using use
used uses based between across both each every any all some more most other only very not no
paper papers section table figure we our you your one two three first second new
""".split())


def fold_ascii(text: Any) -> str:
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii").lower()


def stem(token: str) -> str:
    """Tiny suffix stripper: deterministic and dependency-free (good enough for IDF matching)."""
    for suf, min_len in (("ing", 5), ("ed", 4), ("es", 4), ("s", 3)):
        if len(token) > min_len and token.endswith(suf) and not token.endswith("ss"):
            return token[: -len(suf)]
    return token


def content_stems(text: str, min_len: int = 3) -> List[str]:
    """Tokenise -> drop stopwords -> stem. Hyphenated words are split ('skip-gram' -> skip, gram)."""
    clean = fold_ascii(text).replace("-", " ")
    return [stem(t) for t in _TOKEN_RE.findall(clean) if len(t) >= min_len and t not in STOPWORDS]


def sha256_text(text: str, n: int = 64) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def sha256_file(path: str, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:n]


# --------------------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------------------
def read_json(path: str, default: Any = None) -> Any:
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def write_json_atomic(path: str, payload: Any, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent, default=_json_default)
        # Windows refuses to replace a file that another process (e.g. the website polling progress) has open
        # at that instant; retry briefly, and as a last resort write in place.
        import time
        for attempt in range(12):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 11:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=indent, default=_json_default)
                    break
                time.sleep(0.05 * (attempt + 1))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


# --------------------------------------------------------------------------------------
# Embeddings (process-wide singleton + LRU cache)
# --------------------------------------------------------------------------------------
EMBED_DIM = 384
_EMBEDDER = None
_EMBEDDER_LOCK = threading.Lock()


def get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Returns a SentenceTransformer or None (hash fallback) - loaded once per process."""
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER if _EMBEDDER is not False else None
    with _EMBEDDER_LOCK:
        if _EMBEDDER is None:
            limit_cpu_threads()
            try:
                from sentence_transformers import SentenceTransformer
                _EMBEDDER = SentenceTransformer(model_name, device="cpu")
            except Exception:
                _EMBEDDER = False
    return _EMBEDDER if _EMBEDDER is not False else None


def _hash_embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float64)
    for i in range(max(1, len(text) - 2)):
        d = hashlib.sha256(text[i:i + 3].encode("utf-8")).digest()
        vec[int.from_bytes(d[:4], "little") % dim] += 1.0 if d[4] % 2 == 0 else -1.0
    return vec


@lru_cache(maxsize=50_000)
def _embed_cached(text: str) -> bytes:
    model = get_embedder()
    if model is not None:
        v = np.asarray(model.encode(text, show_progress_bar=False), dtype=np.float64)
    else:
        v = _hash_embed(text)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v.tobytes()


def embed(text: str) -> np.ndarray:
    clean = " ".join(str(text).lower().split())
    if not clean:
        return np.zeros(EMBED_DIM, dtype=np.float64)
    return np.frombuffer(_embed_cached(clean), dtype=np.float64).copy()


def embed_many(texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
    """Batched encoding that also populates the single-string LRU cache."""
    texts = [" ".join(str(t).lower().split()) for t in texts]
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float64)
    model = get_embedder()
    uniq = [t for t in dict.fromkeys(texts) if t]
    if model is not None and len(uniq) > 1:
        vecs = np.asarray(model.encode(uniq, batch_size=batch_size, show_progress_bar=False), dtype=np.float64)
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        lookup = dict(zip(uniq, vecs))
        return np.vstack([lookup[t] if t else np.zeros(EMBED_DIM) for t in texts])
    return np.vstack([embed(t) for t in texts])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))

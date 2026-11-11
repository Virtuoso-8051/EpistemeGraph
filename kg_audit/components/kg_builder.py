import os
import re
import sys
import glob
import json
import time
import hashlib
import threading
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Callable, Optional

import networkx as nx
import numpy as np
from neo4j import GraphDatabase
from pypdf import PdfReader

from kg_audit.entity.config_entity import KGBuilderConfig
from kg_audit.entity.artifact_entity import KGBuilderArtifact
from kg_audit.exceptions.exception import KGAuditException
from kg_audit.utils.api_errors import FatalAPIError, as_fatal, classify as classify_api_error
from kg_audit.utils.llm import default_client
from kg_audit.logging.logger import logging
from kg_audit.utils.main_utils import read_json, sha256_file, sha256_text, write_json_atomic

import importlib.util as _ilu

# Only CHECK that the package exists: importing sentence-transformers pulls in transformers/torch (~25 s on a
# laptop) and must happen lazily, when an embedding is actually needed (main_utils.get_embedder).
SENTENCE_TRANSFORMERS_AVAILABLE = _ilu.find_spec("sentence_transformers") is not None

_GLOBAL_EMBEDDER = None

PROMPT_VERSION = "v4"
QUOTE_ACCEPT = 0.82
ENTITY_SUPPORT_MIN = 0.85
ANCHOR_MIN = 0.62
PARAPHRASE_ENT_MIN = 0.85
PARAPHRASE_SEM_MIN = 0.28
PARAPHRASE_PENALTY = 0.88
SEM_LOW, SEM_HIGH = 0.15, 0.65
COV_TAU = 0.60
MIN_EXTRACTION_CONFIDENCE = 0.50
NEO4J_BATCH = 500

_ABBREV = ("et al.", "e.g.", "i.e.", "fig.", "eq.", "vs.", "etc.", "no.", "sec.", "approx.")

_GENERIC_MENTIONS = {
    "it", "they", "we", "he", "she", "i", "this", "that", "these", "those", "which",
    "its", "our", "their", "paper", "this paper", "the paper", "authors", "the authors",
    "model", "the model", "this model", "our model", "method", "the method", "this method",
    "our method", "approach", "the approach", "this approach", "our approach", "system",
    "the system", "this work", "the study", "this study", "work", "study", "algorithm",
    "the algorithm", "framework", "the framework", "experiment", "experiments", "results"
}

# Comprehensive stop-words for acronym matching (fixes BERT, LLMs, GCNs, etc.)
_STOP_WORDS_ACR = {
    "OF", "THE", "AND", "IN", "FOR", "TO", "A", "AN", "FROM", "WITH",
    "BY", "ON", "AT", "INTO", "OVER", "VIA", "USING", "AS"
}

VAGUE_RELATIONS = {"RELATED_TO", "ASSOCIATED_WITH", "LINKED_TO", "CONNECTED_TO", "MENTIONS", "INVOLVES"}

PREFERRED_RELATIONS = [
    "USES", "PROPOSES", "OUTPERFORMS", "PART_OF", "CONTAINS", "DEFINED_AS", "IS_A",
    "TRAINED_ON", "EVALUATED_ON", "ACHIEVES", "BASED_ON", "IMPROVES", "REDUCES",
    "CAUSES", "REQUIRES", "ENABLES", "MEASURED_BY", "COMPARED_WITH", "PRESERVES",
    "OPTIMIZES", "APPROXIMATES", "GENERATES"
]

RELATION_SYNONYMS = {
    "UTILIZES": "USES", "UTILISES": "USES", "EMPLOYS": "USES", "LEVERAGES": "USES",
    "APPLIES": "USES", "ADOPTS": "USES", "USED_FOR": "USES", "USED_IN": "USES",
    "BEATS": "OUTPERFORMS", "SURPASSES": "OUTPERFORMS", "EXCEEDS": "OUTPERFORMS",
    "IMPROVES_ON": "OUTPERFORMS", "IMPROVES_UPON": "OUTPERFORMS",
    "IS_PART_OF": "PART_OF", "COMPONENT_OF": "PART_OF", "BELONGS_TO": "PART_OF",
    "INCLUDES": "CONTAINS", "COMPRISES": "CONTAINS", "CONSISTS_OF": "CONTAINS",
    "IS_DEFINED_AS": "DEFINED_AS", "DENOTES": "DEFINED_AS",
    "INTRODUCES": "PROPOSES", "PRESENTS": "PROPOSES", "DEVELOPS": "PROPOSES",
    "TRAINED_WITH": "TRAINED_ON", "LEARNED_FROM": "TRAINED_ON",
    "TESTED_ON": "EVALUATED_ON", "BENCHMARKED_ON": "EVALUATED_ON",
    "OBTAINS": "ACHIEVES", "REACHES": "ACHIEVES", "ATTAINS": "ACHIEVES",
    "BUILT_ON": "BASED_ON", "DERIVED_FROM": "BASED_ON",
    "INSTANCE_OF": "IS_A", "TYPE_OF": "IS_A", "IS_TYPE_OF": "IS_A",
    "ENHANCES": "IMPROVES", "BOOSTS": "IMPROVES",
    "DECREASES": "REDUCES", "LOWERS": "REDUCES", "MINIMIZES": "OPTIMIZES",
    "MAXIMIZES": "OPTIMIZES"
}

_CUE_RE = re.compile(
    r"\b(propos|introduc|present|achiev|outperform|improv|us(?:e|es|ed|ing)|train|evaluat|compar|"
    r"demonstrat|defin|consist|based on|dataset|accuracy|benchmark|result|optimiz|preserv|sampl)\w*",
    re.I,
)

EXTRACTION_PROMPT = """You are an expert scientific Knowledge Graph extractor.

Work in TWO STEPS and return both in ONE JSON object.

STEP 1 - "entities": list all key named technical entities that appear in the chunk
(methods, algorithms, models, datasets, metrics, mathematical objectives, concepts).
STEP 2 - "claims": extract up to 10 factual triplets. Both the subject and the object
MUST be copied from your STEP 1 entity list.

Rules:
1. Extract ONLY facts explicitly stated in the chunk. Use no outside knowledge.
2. Never use pronouns or generic phrases as entities ("it", "we", "this paper", "the model",
   "our method"). Resolve them to the specific technical name from the chunk, or skip the claim.
3. Entity types in UPPER_SNAKE_CASE (ALGORITHM, MODEL, DATASET, METRIC, SYSTEM, PERSON,
   ORGANIZATION, CONCEPT).
4. Canonical Naming & Acronyms: If the chunk gives both a full name and an acronym (e.g.,
   "Large-scale Information Network Embedding (LINE)" or "Large Language Models (LLMs)"),
   put the singular full expansion as "name" ("Large-scale Information Network Embedding")
   and the singular acronym as "alias" ("LINE"). Do NOT keep "(LINE)" inside the name.
5. Relations: concise UPPER_SNAKE_CASE verbs. Prefer this vocabulary when it fits:
   __RELATIONS__. Create a new verb only if none fits. Never use vague relations such as
   RELATED_TO or ASSOCIATED_WITH.
6. "is_functional": true ONLY if the relation can mathematically have only one unique object per subject.
7. "evidence_quote": ONE sentence copied character-for-character from the chunk that
   supports the triplet.
8. "extraction_confidence": 0.0-1.0.

Return ONLY this JSON:
{
  "entities": [
    {"name": "Large Language Model", "alias": "LLM", "type": "MODEL"}
  ],
  "claims": [
    {
      "subject": "Large Language Model",
      "subject_alias": "LLM",
      "subject_type": "MODEL",
      "relation": "TRAINED_ON",
      "is_functional": false,
      "object": "BooksCorpus",
      "object_alias": "",
      "object_type": "DATASET",
      "evidence_quote": "Exact verbatim sentence from the chunk.",
      "extraction_confidence": 0.95
    }
  ]
}

<chunk>
__CHUNK__
</chunk>
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()



class _ChunkFailed(RuntimeError):
    """A single chunk could not be extracted (non-fatal); the build retries it once, then stops without saving."""

class KGBuilder:
    def __init__(self, config: KGBuilderConfig):
        try:
            global _GLOBAL_EMBEDDER
            self.config = config
            self.llm = default_client()        # Claude or OpenAI by model name; retries + fatal stop inside
            os.makedirs(self.config.llm_cache_dir, exist_ok=True)

            self._usage: Counter = Counter()
            self._usage_lock = threading.Lock()
            self._last_rejected: List[Dict[str, Any]] = []
            self._last_match_counts: Counter = Counter()

            # one process-wide MiniLM instance shared with Phases 2-3 (thread-capped)
            from kg_audit.utils.main_utils import get_embedder
            self.embedder = get_embedder() if SENTENCE_TRANSFORMERS_AVAILABLE else None
            if self.embedder is None:
                print(
                    "[EpistemeGraph] WARNING: sentence-transformers not installed -> falling back to hashed "
                    "n-gram vectors. Run: pip install sentence-transformers",
                    flush=True,
                )
        except FatalAPIError:
            raise
        except Exception as e:
            raise KGAuditException(e, sys)

    def _encode_text(self, text: str, dim: int = 384) -> np.ndarray:
        clean = str(text).strip().lower()
        if not clean:
            return np.zeros(dim, dtype=np.float64)
        if self.embedder is not None:
            vec = self.embedder.encode(clean, show_progress_bar=False)
            return vec / (np.linalg.norm(vec) + 1e-12)

        vec = np.zeros(dim, dtype=np.float64)
        tokens = [clean[i:i + 3] for i in range(max(1, len(clean) - 2))]
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[idx] += sign
        return vec / (np.linalg.norm(vec) + 1e-12)

    def _encode_many(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float64)
        if self.embedder is not None:
            vecs = self.embedder.encode([t.lower() for t in texts], batch_size=64, show_progress_bar=False)
            return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        return np.vstack([self._encode_text(t) for t in texts])

    @staticmethod
    def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-12
        return float(np.dot(v1, v2) / denom)

    @staticmethod
    def _strip_plural_acronym(token: str) -> str:
        """Normalizes plural acronyms like 'LLMs' -> 'LLM', 'CNNs' -> 'CNN'."""
        t = token.strip().replace(".", "")
        if len(t) >= 3 and t.endswith("s") and t[:-1].isupper():
            return t[:-1]
        return t.upper()

    @classmethod
    def _initials(cls, name: str) -> str:
        """Computes canonical initials for multi-word technical terms (e.g., 'Large Language Models' -> 'LLM')."""
        cleaned = re.sub(r"\([^)]*\)", " ", name).strip()
        words = [
            w for w in cleaned.upper().replace(".", "").replace("-", " ").split()
            if w and w not in _STOP_WORDS_ACR
        ]
        if len(words) < 2:
            return ""
        return "".join(w[0] for w in words)

    @classmethod
    def _is_short_token(cls, s: str) -> bool:
        t = cls._strip_plural_acronym(s)
        return 2 <= len(t) <= 8 and " " not in t and any(ch.isalpha() for ch in t)

    @classmethod
    def _is_acronym_match(cls, s1: str, s2: str) -> bool:
        short_s, long_s = (s1, s2) if len(s1) < len(s2) else (s2, s1)
        if not cls._is_short_token(short_s):
            return False
        acr = cls._strip_plural_acronym(short_s)
        init = cls._initials(long_s)
        return bool(init) and (acr == init)

    @staticmethod
    def _split_inline_parenthetical(mention: str, alias: str) -> Tuple[str, str]:
        """
        If mention is 'Large Language Model (LLM)' or 'LLM (Large Language Model)',
        automatically separates the full expansion and the short alias.
        """
        m = re.match(r"^(.+?)\s*\(([^()]+)\)\s*$", mention.strip())
        if m:
            part1, part2 = m.group(1).strip(), m.group(2).strip()
            if len(part1) >= len(part2):
                return part1, alias or part2
            return part2, alias or part1
        return mention, alias

    @staticmethod
    def _singularize_phrase(text: str) -> str:
        """Normalizes simple trailing plural 's' on multi-word entity names ('Large Language Models' -> 'Large Language Model')."""
        words = text.split()
        if not words:
            return text
        last = words[-1]
        if len(last) > 4 and last.lower().endswith("s") and not last.lower().endswith(("ss", "is", "us", "ics")):
            words[-1] = last[:-1]
        return " ".join(words)

    @staticmethod
    def document_hash(item: str) -> str:
        if os.path.exists(item):
            return sha256_file(item, n=16)
        return sha256_text(str(item), n=16)

    @staticmethod
    def _clean_pdf_text(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
        return " ".join(text.split())

    @staticmethod
    def _norm(text: str) -> str:
        text = unicodedata.normalize("NFKC", str(text)).lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return " ".join(text.split())

    @staticmethod
    def _stem(tok: str) -> str:
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            return tok[:-1]
        return tok

    def _stem_set(self, norm_text: str) -> set:
        return {self._stem(t) for t in norm_text.split()}

    @classmethod
    def _clean_mention(cls, text: Any) -> str:
        t = unicodedata.normalize("NFKC", str(text or "")).strip().strip(".,;:()[]\"'")
        t = re.sub(r"^(the|a|an)\s+", "", t, flags=re.I)
        t = " ".join(t.split())
        if cls._is_short_token(t):
            return cls._strip_plural_acronym(t)
        return cls._singularize_phrase(t)

    @staticmethod
    def _normalize_relation(rel: Any) -> str:
        r = re.sub(r"[^A-Z0-9]+", "_", str(rel or "").upper()).strip("_")
        return RELATION_SYNONYMS.get(r, r) or "RELATED_TO"

    @staticmethod
    def _triple_text(subj: str, rel: str, obj: str) -> str:
        return f"{subj} {rel.replace('_', ' ').lower()} {obj}"

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\(\[])', text)
        merged: List[str] = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if merged and merged[-1].lower().endswith(_ABBREV):
                merged[-1] += " " + p
            else:
                merged.append(p)
        return merged

    @staticmethod
    def _pack_sentences(sentences: List[str], max_chars: int, overlap_sents: int = 1) -> List[str]:
        segments: List[str] = []
        cur: List[str] = []
        cur_len = 0
        for s in sentences:
            if cur and cur_len + len(s) + 1 > max_chars:
                segments.append(" ".join(cur))
                cur = cur[-overlap_sents:] if overlap_sents else []
                cur_len = sum(len(x) + 1 for x in cur)
            cur.append(s)
            cur_len += len(s) + 1
        if cur:
            segments.append(" ".join(cur))
        return segments

    @staticmethod
    def _alpha_ratio(s: str) -> float:
        return sum(ch.isalpha() for ch in s) / max(1, len(s))

    @staticmethod
    def _density(seg: str) -> float:
        words = seg.split()
        if not words:
            return 0.0
        cues = len(_CUE_RE.findall(seg))
        caps = sum(1 for w in words[1:] if w[:1].isupper() and len(w) > 2)
        nums = sum(1 for w in words if any(ch.isdigit() for ch in w))
        return (cues + 0.5 * caps + 0.3 * nums) / (len(words) / 100.0)

    def _select_segments(self, segs: List[str], k: Optional[int]) -> List[str]:
        if k is None or k <= 0 or len(segs) <= k:
            return segs
        scores = np.array([self._density(s) for s in segs])
        picked = []
        for idx_bin in np.array_split(np.arange(len(segs)), k):
            if len(idx_bin):
                picked.append(int(idx_bin[np.argmax(scores[idx_bin])]))
        return [segs[i] for i in sorted(set(picked))]

    def parse_and_chunk_documents(
        self, file_paths_or_texts: List[str], max_chunks_per_doc: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Parses all uploaded PDFs/TXTs, strips trailing Bibliography/References sections,
        and packs sentences into overlapping chunks. When max_chunks_per_doc is None (default),
        100% of all valid chunks in every document are processed.
        """
        try:
            chunks: List[Dict[str, Any]] = []
            overlap_sents = 1 if getattr(self.config, "chunk_overlap", 0) > 0 else 0

            for doc_idx, item in enumerate(file_paths_or_texts):
                # Content-addressed document id: stable across uploads, so chunk ids never collide
                # and re-uploading the same PDF is idempotent.
                doc_id = f"DOC_{self.document_hash(item)[:10]}"

                if os.path.exists(item) and item.lower().endswith(".pdf"):
                    print(f"[Phase 1] Reading PDF [{doc_id}]: {os.path.basename(item)}...", flush=True)
                    reader = PdfReader(item)
                    raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)
                    source_name = os.path.basename(item)
                elif os.path.exists(item):
                    with open(item, "r", encoding="utf-8", errors="ignore") as f:
                        raw_text = f.read()
                    source_name = os.path.basename(item)
                else:
                    raw_text = str(item)
                    source_name = f"inline_text_{doc_idx + 1}"

                clean_text = self._clean_pdf_text(raw_text)
                if not clean_text:
                    continue

                # Strip bibliography/references section in the final 35% of the document
                tail_start = int(len(clean_text) * 0.65)
                ref_matches = list(re.finditer(r"\b(references|bibliography)\b", clean_text[tail_start:], flags=re.I))
                if ref_matches:
                    clean_text = clean_text[:tail_start + ref_matches[-1].start()]

                sentences = self._split_sentences(clean_text)
                doc_segments = self._pack_sentences(sentences, self.config.chunk_size, overlap_sents)
                doc_segments = [s for s in doc_segments if len(s) >= 80 and self._alpha_ratio(s) >= 0.55]
                doc_segments = self._select_segments(doc_segments, max_chunks_per_doc)

                for idx_c, seg in enumerate(doc_segments, start=1):
                    chunks.append({
                        "document_id": doc_id,
                        "source_name": source_name,
                        "chunk_id": f"{doc_id}_CH_{idx_c:03d}",
                        "text": seg,
                    })

            print(
                f"[Phase 1] Prepared {len(chunks)} total sentence-aligned chunks across {len(file_paths_or_texts)} document(s).",
                flush=True,
            )
            return chunks
        except FatalAPIError:
            raise
        except Exception as e:
            raise KGAuditException(e, sys)

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """One JSON extraction call; usage is tallied for the fidelity report."""
        parsed, meta = self.llm.complete_json(self.config.llm_model_name, "You output strictly valid JSON.", prompt,
                                              temperature=0.0, max_tokens=16000)
        with self._usage_lock:
            self._usage["prompt_tokens"] += int(meta.get("prompt_tokens", 0))
            self._usage["completion_tokens"] += int(meta.get("completion_tokens", 0))
            self._usage["calls"] += 1
        return parsed

    def _filter_claims(self, claims: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        known = set()
        for e in entities or []:
            if not isinstance(e, dict):
                continue
            raw_name, raw_alias = self._split_inline_parenthetical(
                str(e.get("name", "")), str(e.get("alias", ""))
            )
            for val in (raw_name, raw_alias):
                if val:
                    known.add(self._norm(self._clean_mention(val)))

        seen, kept = set(), []
        for c in claims or []:
            if not isinstance(c, dict):
                continue
            c = dict(c)
            subj_raw, subj_al_raw = self._split_inline_parenthetical(
                str(c.get("subject", "")), str(c.get("subject_alias", ""))
            )
            obj_raw, obj_al_raw = self._split_inline_parenthetical(
                str(c.get("object", "")), str(c.get("object_alias", ""))
            )

            c["subject"] = self._clean_mention(subj_raw)
            c["object"] = self._clean_mention(obj_raw)
            c["subject_alias"] = self._clean_mention(subj_al_raw)
            c["object_alias"] = self._clean_mention(obj_al_raw)
            c["relation"] = self._normalize_relation(c.get("relation"))

            s, o = self._norm(c["subject"]), self._norm(c["object"])
            if not s or not o or s == o:
                continue
            if s in _GENERIC_MENTIONS or o in _GENERIC_MENTIONS:
                continue
            if c["relation"] in VAGUE_RELATIONS:
                continue
            try:
                conf = float(c.get("extraction_confidence", 0.9))
            except (TypeError, ValueError):
                conf = 0.9
            c["extraction_confidence"] = conf
            if conf < MIN_EXTRACTION_CONFIDENCE:
                continue
            if known:
                s_ok = s in known or self._norm(c["subject_alias"]) in known
                o_ok = o in known or self._norm(c["object_alias"]) in known
                if not (s_ok and o_ok):
                    continue
            key = (s, c["relation"], o)
            if key in seen:
                continue
            seen.add(key)
            kept.append(c)
        return kept

    def extract_from_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        if getattr(self, "_abort_reason", None):          # an earlier chunk hit a fatal API error
            raise FatalAPIError(self._abort_reason)
        try:
            chunk_hash = hashlib.sha256(
                f"{PROMPT_VERSION}_{self.config.llm_model_name}_{chunk['text']}".encode("utf-8")
            ).hexdigest()
            cache_file = os.path.join(self.config.llm_cache_dir, f"{chunk_hash}.json")

            parsed: Optional[Dict[str, Any]] = None
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    parsed = json.load(f)
                print(f"  -> [CACHE HIT] {chunk['chunk_id']} ({chunk['source_name']}) - $0 API cost", flush=True)

            if parsed is None:
                prompt = (
                    EXTRACTION_PROMPT
                    .replace("__RELATIONS__", ", ".join(PREFERRED_RELATIONS))
                    .replace("__CHUNK__", chunk["text"])
                )
                parsed = self._call_llm(prompt)
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(parsed, f, indent=2)

            claims = self._filter_claims(parsed.get("claims", []), parsed.get("entities", []))
            for c in claims:
                c["document_id"] = chunk["document_id"]
                c["chunk_id"] = chunk["chunk_id"]
                c["source_name"] = chunk["source_name"]
                c["extracted_at"] = _utcnow()

            print(f"  -> [EXTRACTED] {chunk['chunk_id']} ({chunk['source_name']}): {len(claims)} triplets", flush=True)
            return claims

        except FatalAPIError:
            raise
        except Exception as e:
            if classify_api_error(e) == "fatal":
                raise as_fatal(e) from e
            print(f"  -> [WARNING] Extraction failed on {chunk['chunk_id']}: {e}", flush=True)
            logging.warning(f"Extraction error on chunk {chunk['chunk_id']}: {e}")
            raise _ChunkFailed(str(e)[:300]) from e

    def _prep_chunk(self, chunk_text: str) -> Dict[str, Any]:
        raw_sents = self._split_sentences(chunk_text)
        raw_windows = raw_sents + [f"{a} {b}" for a, b in zip(raw_sents, raw_sents[1:])]
        norm_text = self._norm(chunk_text)
        return {
            "norm_text": norm_text,
            "raw_windows": raw_windows,
            "norm_windows": [self._norm(w) for w in raw_windows],
            "tokens": self._stem_set(norm_text),
            "emb": None,
        }

    @staticmethod
    def _quote_similarity(quote_norm: str, chunk_norm: str, windows: List[str]) -> float:
        if not quote_norm:
            return 0.0
        if quote_norm in chunk_norm:
            return 1.0
        return max((SequenceMatcher(None, quote_norm, w).ratio() for w in windows), default=0.0)

    def _entity_support(self, name: str, alias: str, q_tokens: set, c_tokens: set) -> float:
        best = 0.0
        for cand in (name, alias):
            toks = [self._stem(t) for t in self._norm(cand).split()]
            if not toks:
                continue
            in_q = sum(t in q_tokens for t in toks) / len(toks)
            in_c = sum(t in c_tokens for t in toks) / len(toks)
            if in_q >= 0.65:
                best = max(best, 1.0)
            elif in_c >= 0.65:
                best = max(best, 0.55)
        return best

    def _verify_claims(
        self, raw_claims: List[Dict[str, Any]], chunks_by_id: Dict[str, str]
    ) -> Tuple[List[Dict[str, Any]], List[float], List[Dict[str, Any]]]:
        grounding_scores: List[float] = []
        verified: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        chunk_cache: Dict[str, Dict[str, Any]] = {}
        match_counts: Counter = Counter()

        def reject(claim: Dict[str, Any], reason: str, score: float) -> None:
            rejected.append({
                "chunk_id": claim.get("chunk_id"),
                "subject": claim.get("subject"),
                "relation": claim.get("relation"),
                "object": claim.get("object"),
                "evidence_quote": claim.get("evidence_quote"),
                "reason": reason,
                "grounding": round(score, 4),
            })

        prelim: List[Dict[str, Any]] = []
        for claim in raw_claims:
            subj = str(claim.get("subject", "")).strip()
            obj = str(claim.get("object", "")).strip()
            rel = self._normalize_relation(claim.get("relation", "RELATED_TO"))
            quote = str(claim.get("evidence_quote", "")).strip()
            chunk_text = chunks_by_id.get(claim.get("chunk_id", ""), "")

            if not subj or not obj or not chunk_text:
                grounding_scores.append(0.0)
                reject(claim, "empty_field_or_missing_chunk", 0.0)
                continue

            if claim["chunk_id"] not in chunk_cache:
                chunk_cache[claim["chunk_id"]] = self._prep_chunk(chunk_text)
            info = chunk_cache[claim["chunk_id"]]

            q_norm = self._norm(quote)
            q_sim = self._quote_similarity(q_norm, info["norm_text"], info["norm_windows"])
            p = {
                "claim": claim, "subj": subj, "obj": obj, "rel": rel, "quote": quote, "chunk_text": chunk_text,
                "info": info, "q_sim": q_sim, "verbatim": q_sim >= QUOTE_ACCEPT,
                "anchor_text": None, "anchor_sim": 0.0,
                "triple": self._triple_text(subj, rel, obj),
            }
            if p["verbatim"]:
                p["evidence"] = quote
                p["ent"] = 0.5 * (
                    self._entity_support(subj, claim.get("subject_alias", ""), self._stem_set(q_norm), info["tokens"])
                    + self._entity_support(obj, claim.get("object_alias", ""), self._stem_set(q_norm), info["tokens"])
                )
            prelim.append(p)

        to_anchor = [p for p in prelim if not p["verbatim"] and p["quote"]]
        if to_anchor:
            QV = self._encode_many([p["quote"] for p in to_anchor])
            for p, qv in zip(to_anchor, QV):
                info = p["info"]
                if info["emb"] is None:
                    info["emb"] = self._encode_many(info["raw_windows"])
                if info["emb"].shape[0] == 0:
                    continue
                sims = info["emb"] @ qv
                j = int(np.argmax(sims))
                p["anchor_text"], p["anchor_sim"] = info["raw_windows"][j], float(sims[j])

        for p in prelim:
            if p["verbatim"]:
                continue
            claim, info = p["claim"], p["info"]
            if p["anchor_text"]:
                a_tokens = self._stem_set(self._norm(p["anchor_text"]))
                p["evidence"] = p["anchor_text"]
                p["ent"] = 0.5 * (
                    self._entity_support(p["subj"], claim.get("subject_alias", ""), a_tokens, info["tokens"])
                    + self._entity_support(p["obj"], claim.get("object_alias", ""), a_tokens, info["tokens"])
                )
            else:
                p["evidence"] = p["quote"] or p["chunk_text"]
                p["ent"] = 0.0

        if prelim:
            T = self._encode_many([p["triple"] for p in prelim])
            E = self._encode_many([p["evidence"] for p in prelim])
            cos_all = np.sum(T * E, axis=1)
        else:
            cos_all = np.zeros(0)

        for p, cos in zip(prelim, cos_all):
            claim = p["claim"]
            sem = float(np.clip((float(cos) - SEM_LOW) / (SEM_HIGH - SEM_LOW), 0.0, 1.0))
            ent = p["ent"]

            if p["verbatim"]:
                accepted = ent >= ENTITY_SUPPORT_MIN
                reason = None if accepted else "entities_not_in_quote"
                ev_score = 1.0
                match_type = "verbatim"
            else:
                anchored = p["anchor_sim"] >= ANCHOR_MIN
                if not anchored:
                    reason = "quote_not_found_in_chunk"
                elif ent < PARAPHRASE_ENT_MIN:
                    reason = "entities_not_in_anchor"
                elif sem < PARAPHRASE_SEM_MIN:
                    reason = "low_semantic_support"
                else:
                    reason = None
                accepted = reason is None
                ev_score = (
                    PARAPHRASE_PENALTY * min(1.0, p["anchor_sim"]) if accepted
                    else 0.6 * max(p["q_sim"], p["anchor_sim"])
                )
                match_type = "paraphrase_anchored"

            grounding = float(ev_score * (0.4 * ent + 0.6 * sem))
            grounding_scores.append(grounding)

            if accepted:
                claim["relation"] = p["rel"]
                claim["grounding_score"] = grounding
                claim["evidence_match"] = match_type
                if match_type == "paraphrase_anchored":
                    claim["llm_quote"] = p["quote"]
                    claim["evidence_quote"] = p["anchor_text"]
                match_counts[match_type] += 1
                verified.append(claim)
            else:
                reject(claim, reason or "rejected", grounding)

        self._last_match_counts = match_counts
        return verified, grounding_scores, rejected

    def _resolve_entities(
        self,
        verified_claims: List[Dict[str, Any]],
        existing_entities: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Merges synonymous entities across chunks & PDFs using:
        1. Exact case-insensitive / singularized lookup
        2. Acronym <-> Full Expansion bidirectional matching (e.g., 'LLM' <-> 'Large Language Model')
        3. Dense Cosine Similarity >= entity_sim_threshold
        Always prefers the descriptive full name as the canonical key while storing all acronyms/variants in aliases.

        existing_entities (incremental mode): canonical entities already in the KG. They are
        pre-loaded into the resolver so a mention in a NEW PDF is merged into the EXISTING node
        instead of creating a duplicate. Existing canonical names are never renamed.
        """
        thr_same = float(self.config.entity_sim_threshold)
        thr_diff = min(0.96, thr_same + 0.06)

        canonical: Dict[str, Dict[str, Any]] = {}
        alias_lookup: Dict[str, str] = {}
        names: List[str] = []
        types: List[str] = []
        emb_mat: Optional[np.ndarray] = None
        n_ent = 0
        initials_idx: Dict[str, List[str]] = {}
        short_idx: Dict[str, List[str]] = {}

        def index_strings(c_name: str, strings) -> None:
            for s in strings:
                if not s:
                    continue
                if self._is_short_token(s):
                    acr_key = self._strip_plural_acronym(s)
                    if c_name not in short_idx.setdefault(acr_key, []):
                        short_idx[acr_key].append(c_name)
                init = self._initials(s)
                if 2 <= len(init) <= 8:
                    if c_name not in initials_idx.setdefault(init, []):
                        initials_idx[init].append(c_name)

        def acronym_probe(s: str) -> Optional[str]:
            if not s:
                return None
            if self._is_short_token(s):
                hit = initials_idx.get(self._strip_plural_acronym(s))
                if hit:
                    return hit[0]
            init = self._initials(s)
            if init:
                hit = short_idx.get(init)
                if hit:
                    return hit[0]
            return None

        def register_new(name: str, ent_type: str, vec: np.ndarray) -> None:
            nonlocal emb_mat, n_ent
            if emb_mat is None:
                emb_mat = np.zeros((256, len(vec)), dtype=np.float64)
            if n_ent == emb_mat.shape[0]:
                emb_mat = np.vstack([emb_mat, np.zeros_like(emb_mat)])
            emb_mat[n_ent] = vec
            names.append(name)
            types.append(ent_type)
            n_ent += 1

        # ---- seed the resolver with the entities already in the KG (incremental ingestion)
        if existing_entities:
            ex_names = list(existing_entities.keys())
            ex_vecs = self._encode_many(ex_names)
            for name, vec in zip(ex_names, ex_vecs):
                data = existing_entities[name]
                aliases = set(data.get("aliases", [])) | {name}
                canonical[name] = {
                    "name": name,
                    "type": data.get("type", "CONCEPT"),
                    "embedding": vec,
                    "aliases": aliases,
                    "source_pdfs": set(data.get("source_pdfs", [])),
                }
                for a in aliases:
                    alias_lookup.setdefault(str(a).lower(), name)
                register_new(name, data.get("type", "CONCEPT"), vec)
                index_strings(name, list(aliases))

        for claim in verified_claims:
            doc_source = claim["source_name"]
            for role, alias_key, type_key in (
                ("subject", "subject_alias", "subject_type"),
                ("object", "object_alias", "object_type"),
            ):
                mention = str(claim.get(role, "")).strip()
                alias = str(claim.get(alias_key, "")).strip()
                ent_type = str(claim.get(type_key, "CONCEPT")).strip().upper().replace(" ", "_")

                # Prefer the longer descriptive string as primary mention and shorter as alias
                if alias and len(alias) > len(mention) and self._is_short_token(mention):
                    mention, alias = alias, mention

                low_mention = mention.lower()
                low_alias = alias.lower() if alias else ""

                if low_mention in alias_lookup:
                    name = alias_lookup[low_mention]
                    if alias:
                        alias_lookup[low_alias] = name
                        canonical[name]["aliases"].add(alias)
                    canonical[name]["source_pdfs"].add(doc_source)
                    claim[role] = name
                    continue
                if low_alias and low_alias in alias_lookup:
                    name = alias_lookup[low_alias]
                    alias_lookup[low_mention] = name
                    canonical[name]["aliases"].add(mention)
                    canonical[name]["source_pdfs"].add(doc_source)
                    claim[role] = name
                    continue

                m_vec = self._encode_text(mention)
                matched = acronym_probe(mention) or acronym_probe(alias)

                if matched is None and n_ent > 0 and emb_mat is not None:
                    sims = emb_mat[:n_ent] @ m_vec
                    thr_vec = np.where(np.array(types) != ent_type, thr_diff, thr_same)
                    ok = np.where(sims >= thr_vec)[0]
                    if ok.size:
                        matched = names[int(ok[np.argmax(sims[ok])])]

                if matched:
                    alias_lookup[low_mention] = matched
                    if low_alias:
                        alias_lookup[low_alias] = matched
                    canonical[matched]["aliases"].add(mention)
                    if alias:
                        canonical[matched]["aliases"].add(alias)
                    canonical[matched]["source_pdfs"].add(doc_source)
                    index_strings(matched, [mention, alias])
                    claim[role] = matched
                else:
                    # Only surface forms actually observed in the text become aliases. Auto-generated
                    # initials are used for *matching* (index_strings) but are no longer stored,
                    # which previously produced junk aliases such as 'LSINE' or '1ST'.
                    aliases_set = {mention}
                    if alias:
                        aliases_set.add(alias)

                    canonical[mention] = {
                        "name": mention,
                        "type": ent_type,
                        "embedding": m_vec,
                        "aliases": aliases_set,
                        "source_pdfs": {doc_source},
                    }
                    alias_lookup[low_mention] = mention
                    if low_alias:
                        alias_lookup[low_alias] = mention
                    register_new(mention, ent_type, m_vec)
                    index_strings(mention, [mention, alias])
                    claim[role] = mention

        # Remove post-resolution self-loops if subject and object resolved to the same canonical entity
        verified_claims[:] = [c for c in verified_claims if c["subject"] != c["object"]]
        return canonical

    def validate_and_resolve_entities(
        self,
        raw_claims: List[Dict[str, Any]],
        chunks_by_id: Dict[str, str],
        existing_entities: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[float]]:
        try:
            print(f"[Phase 1] Validating quotes & resolving entities across {len(raw_claims)} raw claims...", flush=True)
            verified, scores, rejected = self._verify_claims(raw_claims, chunks_by_id)
            self._last_rejected = rejected
            canonical = self._resolve_entities(verified, existing_entities)
            reasons = Counter(r["reason"] for r in rejected)
            if rejected:
                print(f"[Phase 1] Rejected {len(rejected)} claims: {dict(reasons)}", flush=True)
            return verified, canonical, scores
        except FatalAPIError:
            raise
        except Exception as e:
            raise KGAuditException(e, sys)

    # ==================================================================================
    # S_KG : corpus-level knowledge-graph fidelity (decomposable, so it is incremental)
    # ==================================================================================
    def chunk_coverage(
        self, chunks: List[Dict[str, Any]], verified_claims: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Per-chunk semantic coverage (recall proxy). For chunk c with sentences s_1..s_m and
        triples T_c extracted from it:
            cov(c) = (1/m) sum_i  min(1, max_{t in T_c} cos(phi(s_i), phi(t)) / tau_cov)
        Chunks that yielded no triple contribute cov = 0. Weight w_c = log(1 + |c|).
        Stored per chunk so that S_KG can be re-aggregated when new PDFs arrive without
        re-encoding the old corpus.
        """
        by_chunk: Dict[str, List[str]] = {}
        for c in verified_claims:
            by_chunk.setdefault(c["chunk_id"], []).append(self._triple_text(c["subject"], c["relation"], c["object"]))
        out: Dict[str, Dict[str, float]] = {}
        for ch in chunks:
            w = float(np.log1p(len(ch["text"])))
            triples = by_chunk.get(ch["chunk_id"], [])
            sents = [s for s in self._split_sentences(ch["text"]) if len(s.split()) >= 6]
            if not triples or not sents:
                out[ch["chunk_id"]] = {"w": w, "cov": 0.0}
                continue
            sims = self._encode_many(sents) @ self._encode_many(triples).T
            cov = float(np.mean(np.clip(sims.max(axis=1) / COV_TAU, 0.0, 1.0)))
            out[ch["chunk_id"]] = {"w": w, "cov": cov}
        return out

    @staticmethod
    def aggregate_fidelity(store: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
        """
        Four parameter-free pillars, each in [0, 1]:

          Phi_prov = mean grounding score over ALL extracted claims (rejected ones included),
                     i.e. an extraction-precision proxy.
          Phi_cov  = sum_c w_c cov(c) / sum_c w_c                (corpus recall proxy)
          Phi_topo = 1 - (k(G) - 1) / (|V| - 1)                  (k = #connected components;
                     1 for a connected KG, 0 for a set of isolated triples)
          Phi_cons = 1 - (#functional-constraint violations) / |claims|

          S_KG = (Phi_prov * Phi_cov * Phi_topo * Phi_cons)^(1/4)

        The unweighted geometric mean is the unique symmetric, scale-invariant aggregator
        that vanishes when any pillar vanishes - no tuned exponents (the old 0.7/0.3 are gone).
        """
        claims = store.get("claims", [])
        g = store.get("grounding_scores", [])
        cov = store.get("chunk_coverage", {})

        phi_prov = float(np.mean(g)) if g else 0.0
        tw = sum(v["w"] for v in cov.values())
        phi_cov = float(sum(v["w"] * v["cov"] for v in cov.values()) / tw) if tw > 0 else 0.0

        G = nx.Graph()
        for c in claims:
            if c["subject"] != c["object"]:
                G.add_edge(c["subject"], c["object"])
        n = G.number_of_nodes()
        phi_topo = float(1.0 - (nx.number_connected_components(G) - 1) / (n - 1)) if n > 1 else 0.0

        functional: Dict[Tuple[str, str], set] = {}
        conflicts = 0
        for c in claims:
            if bool(c.get("is_functional", False)):
                key = (c["subject"], c["relation"])
                objs = functional.setdefault(key, set())
                if objs and c["object"] not in objs:
                    conflicts += 1
                objs.add(c["object"])
        phi_cons = float(1.0 - conflicts / max(1, len(claims)))

        s_kg = float(np.prod([max(p, 1e-12) for p in (phi_prov, phi_cov, phi_topo, phi_cons)]) ** 0.25)
        return phi_prov, phi_cov, phi_topo, phi_cons, s_kg

    def compute_kg_fidelity_metrics(
        self,
        chunks: List[Dict[str, Any]],
        verified_claims: List[Dict[str, Any]],
        grounding_scores: List[float],
    ) -> Tuple[float, float, float, float, float]:
        """Stand-alone (non-incremental) S_KG for a single batch - kept for notebooks/back-compat."""
        try:
            store = {
                "claims": verified_claims,
                "grounding_scores": grounding_scores,
                "chunk_coverage": self.chunk_coverage(chunks, verified_claims),
            }
            return self.aggregate_fidelity(store)
        except FatalAPIError:
            raise
        except Exception as e:
            raise KGAuditException(e, sys)

    # ==================================================================================
    # Cumulative local store (source of truth mirrored into Neo4j)
    # ==================================================================================
    @staticmethod
    def _empty_store() -> Dict[str, Any]:
        return {"entities": {}, "claims": [], "grounding_scores": [], "chunk_coverage": {}, "rejected": []}

    def _load_store(self) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
        """Returns (kg_store, corpus_registry, corpus_chunks). Migrates a legacy single-run KG once."""
        store = read_json(self.config.kg_store_path)
        registry = read_json(self.config.corpus_registry_path, default={}) or {}
        chunks = read_json(self.config.latest_chunks_path, default=[]) or []
        if store is None:
            store = self._empty_store()
            legacy = read_json(self.config.kg_fidelity_state_path, default={}) or {}
            legacy_kg = read_json(legacy.get("latest_extracted_kg_path", ""), default=None)
            if legacy_kg and chunks:
                print("[Phase 1] Migrating legacy single-run KG into the cumulative store...", flush=True)
                for n in legacy_kg.get("nodes", []):
                    store["entities"][n["name"]] = {
                        "type": n.get("type", "CONCEPT"),
                        "aliases": [a for a in n.get("aliases", []) if len(a) >= 2],
                        "source_pdfs": n.get("source_pdfs", []),
                    }
                store["claims"] = legacy_kg.get("edges", [])
                rejected = read_json(os.path.join(os.path.dirname(self.config.kg_fidelity_state_path), "rejected_claims.json"), default=[]) or []
                store["grounding_scores"] = [float(c.get("grounding_score", 0.0)) for c in store["claims"]] + \
                                            [float(r.get("grounding", 0.0)) for r in rejected]
                store["chunk_coverage"] = self.chunk_coverage(chunks, store["claims"])
                for src in sorted({c["source_name"] for c in chunks}):
                    path = os.path.join(self.config.data_uploads_dir, src)
                    if os.path.exists(path):
                        registry[self.document_hash(path)] = {
                            "source_name": src, "ingested_at": legacy.get("timestamp", _utcnow()),
                            "chunks": sum(1 for c in chunks if c["source_name"] == src), "migrated": True,
                        }
        return store, registry, chunks

    # ==================================================================================
    # Neo4j: corpus-namespaced, idempotent MERGE. Each workspace owns the nodes whose
    # kg_id equals its own; no statement here can touch another corpus' graph.
    # ==================================================================================
    CYPHER_SCHEMA = [
        # v1 constraints made names unique across the WHOLE database, which forbids two corpora
        # from both containing e.g. "DeepWalk". Replace them with a per-corpus composite key.
        "DROP CONSTRAINT entity_name_unique IF EXISTS",
        "DROP CONSTRAINT entity_key_unique IF EXISTS",
        # one-off migration: nodes written before namespacing belong to the 'legacy' corpus
        "MATCH (e:Entity) WHERE e.kg_id IS NULL SET e.kg_id = 'legacy'",
        "MATCH (e:Entity) WHERE e.name_key IS NULL SET e.name_key = toLower(trim(e.name))",
        "MATCH (:Entity)-[r:RELATED_TO]->(v:Entity) WHERE r.kg_id IS NULL SET r.kg_id = v.kg_id",
        "CREATE CONSTRAINT entity_kg_key IF NOT EXISTS FOR (e:Entity) REQUIRE (e.kg_id, e.name_key) IS UNIQUE",
        "CREATE INDEX entity_kg_idx IF NOT EXISTS FOR (e:Entity) ON (e.kg_id)",
    ]

    CYPHER_UPSERT_ENTITIES = """
        UNWIND $batch AS row
        MERGE (e:Entity {kg_id: row.kg_id, name_key: row.name_key})
          ON CREATE SET e.name        = row.name,
                        e.type        = row.type,
                        e.aliases     = row.aliases,
                        e.source_pdfs = row.source_pdfs,
                        e.created_at  = row.ts
          ON MATCH SET  e.type        = coalesce(e.type, row.type),
                        e.aliases     = coalesce(e.aliases, []) + [a IN row.aliases WHERE NOT a IN coalesce(e.aliases, [])],
                        e.source_pdfs = coalesce(e.source_pdfs, []) + [p IN row.source_pdfs WHERE NOT p IN coalesce(e.source_pdfs, [])]
        SET e.updated_at        = row.ts,
            e.pdf_mention_count = size(e.source_pdfs)
    """

    CYPHER_UPSERT_EDGES = """
        UNWIND $batch AS row
        MATCH (u:Entity {kg_id: row.kg_id, name_key: row.subj_key})
        MATCH (v:Entity {kg_id: row.kg_id, name_key: row.obj_key})
        MERGE (u)-[r:RELATED_TO {type: row.rel}]->(v)
          ON CREATE SET r.created_at = row.ts, r.kg_id = row.kg_id,
                        r.chunk_ids = [], r.evidence_quotes = [], r.source_pdfs = [], r.document_ids = []
        WITH r, row, (r.grounding_score IS NULL OR row.grounding >= r.grounding_score) AS better
        SET r.weight          = CASE WHEN r.weight IS NULL OR row.conf > r.weight THEN row.conf ELSE r.weight END,
            r.grounding_score = CASE WHEN better THEN row.grounding ELSE r.grounding_score END,
            r.evidence_quote  = CASE WHEN better THEN row.best_quote ELSE r.evidence_quote END,
            r.is_functional   = coalesce(r.is_functional, false) OR row.is_functional,
            r.evidence_quotes = (coalesce(r.evidence_quotes, []) + [q IN row.quotes WHERE NOT q IN coalesce(r.evidence_quotes, [])])[..5],
            r.source_pdfs     = coalesce(r.source_pdfs, [])  + [p IN row.sources   WHERE NOT p IN coalesce(r.source_pdfs, [])],
            r.document_ids    = coalesce(r.document_ids, []) + [d IN row.doc_ids   WHERE NOT d IN coalesce(r.document_ids, [])],
            r.chunk_ids       = coalesce(r.chunk_ids, [])    + [c IN row.chunk_ids WHERE NOT c IN coalesce(r.chunk_ids, [])],
            r.updated_at      = row.ts
        SET r.support_count = size(r.chunk_ids)
    """

    # scoped reset: deletes ONLY this corpus' nodes (and their relationships)
    CYPHER_RESET_CORPUS = "MATCH (e:Entity {kg_id: $kg_id}) DETACH DELETE e"

    @staticmethod
    def name_key(name: str) -> str:
        return " ".join(str(name).lower().split())

    @staticmethod
    def _aggregate_edges(verified_claims: List[Dict[str, Any]], s_kg_score: float = 0.0) -> List[Dict[str, Any]]:
        agg: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for c in verified_claims:
            key = (c["subject"], c["relation"], c["object"])
            g = float(c.get("grounding_score", 1.0))
            row = agg.setdefault(key, {
                "subj": c["subject"], "obj": c["object"], "rel": c["relation"],
                "subj_key": KGBuilder.name_key(c["subject"]), "obj_key": KGBuilder.name_key(c["object"]),
                "conf": 0.0, "grounding": -1.0, "best_quote": "", "quotes": [], "is_functional": False,
                "sources": [], "doc_ids": [], "chunk_ids": [],
            })
            row["conf"] = max(row["conf"], float(c.get("extraction_confidence", 0.9)))
            row["is_functional"] = row["is_functional"] or bool(c.get("is_functional", False))
            q = c.get("evidence_quote", "")
            if q and q not in row["quotes"] and len(row["quotes"]) < 5:
                row["quotes"].append(q)
            if g > row["grounding"]:
                row["grounding"], row["best_quote"] = g, q
            for lst, val in (("sources", c["source_name"]), ("doc_ids", c["document_id"]), ("chunk_ids", c["chunk_id"])):
                if val not in row[lst]:
                    row[lst].append(val)
        return list(agg.values())

    def push_to_neo4j_cloud(
        self,
        canonical_entities: Dict[str, Dict[str, Any]],
        verified_claims: List[Dict[str, Any]],
        s_kg_score: float,
        clear_existing: Optional[bool] = None,
        fidelity_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Corpus-namespaced upsert. MERGE keys:
          (:Entity {kg_id, name_key})                - one node per entity per corpus
          (u)-[:RELATED_TO {type}]->(v)              - one relationship per (u, relation, v)
        List properties are unioned, scalar evidence is replaced only by better-grounded evidence,
        and support_count = |chunk_ids| stays correct because chunk ids are content-addressed.
        `clear_existing` deletes only the nodes of THIS corpus (kg_id), never the whole database.
        """
        driver = None
        kg = getattr(self.config, "kg_id", "legacy")
        try:
            if clear_existing is None:
                clear_existing = bool(getattr(self.config, "clear_neo4j_on_build", False))
            edge_batch = self._aggregate_edges(verified_claims)
            now_ts = _utcnow()
            for e in edge_batch:
                e.update({"ts": now_ts, "kg_id": kg})
            entity_batch = [
                {"name": name, "name_key": self.name_key(name), "type": d.get("type", "CONCEPT"), "kg_id": kg,
                 "aliases": sorted(d.get("aliases", [])), "source_pdfs": sorted(d.get("source_pdfs", [])), "ts": now_ts}
                for name, d in canonical_entities.items()
            ]
            print(f"[Phase 1] Upserting {len(entity_batch)} entities & {len(edge_batch)} relations into Neo4j "
                  f"corpus kg_id='{kg}' ({'scoped rebuild' if clear_existing else 'MERGE'})...", flush=True)

            driver = GraphDatabase.driver(self.config.neo4j_uri, auth=(self.config.neo4j_user, self.config.neo4j_password))
            driver.verify_connectivity()
            with driver.session(database=self.config.neo4j_database) as session:
                for stmt in self.CYPHER_SCHEMA:
                    try:
                        session.run(stmt).consume()
                    except Exception as ce:
                        logging.warning(f"Neo4j schema statement skipped ({ce}): {stmt[:60]}")
                if clear_existing:
                    session.run(self.CYPHER_RESET_CORPUS, kg_id=kg).consume()
                for i in range(0, len(entity_batch), NEO4J_BATCH):
                    session.execute_write(lambda tx, b=entity_batch[i:i + NEO4J_BATCH]: tx.run(self.CYPHER_UPSERT_ENTITIES, batch=b).consume())
                for i in range(0, len(edge_batch), NEO4J_BATCH):
                    session.execute_write(lambda tx, b=edge_batch[i:i + NEO4J_BATCH]: tx.run(self.CYPHER_UPSERT_EDGES, batch=b).consume())
                fs = fidelity_state or {}
                session.run(
                    """
                    MERGE (m:GraphMeta {id: $kg_id})
                    SET m.overall_kg_accuracy_score = $s_kg, m.verified_triplets = $count,
                        m.documents = $docs, m.updated_at = $ts
                    """,
                    kg_id=kg, s_kg=s_kg_score, count=int(fs.get("verified_triplets_pushed", len(verified_claims))),
                    docs=int(fs.get("documents_in_corpus", 0)), ts=now_ts,
                ).consume()
            print(f"[Phase 1] SUCCESS: Neo4j corpus '{kg}' updated.", flush=True)
            return True
        except Exception as e:
            print(f"[Phase 1] ERROR pushing to Neo4j Cloud: {e}", flush=True)
            logging.error(f"Neo4j Cloud push failed: {e}")
            return False
        finally:
            if driver is not None:
                try:
                    driver.close()
                except Exception:
                    pass

    # ==================================================================================
    # Orchestration
    # ==================================================================================
    def initiate_kg_from_directory(self, directory_path: str = None) -> KGBuilderArtifact:
        target_dir = directory_path or self.config.data_uploads_dir
        os.makedirs(target_dir, exist_ok=True)
        all_files = sorted(glob.glob(os.path.join(target_dir, "*.pdf"))) + sorted(glob.glob(os.path.join(target_dir, "*.txt")))
        if not all_files:
            raise ValueError(f"No .pdf or .txt files found in {target_dir}. Place your PDFs there first!")
        return self.initiate_kg_construction(all_files)

    def initiate_kg_construction(
        self,
        file_paths_or_texts: List[str],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        reset: Optional[bool] = None,
    ) -> KGBuilderArtifact:
        """
        Incremental ingestion:
          1. content-hash every input; documents already in corpus_registry.json are skipped
          2. chunk + extract (parallel, cached) only the NEW documents
          3. verify quotes, then resolve new mentions AGAINST the existing canonical entities
          4. append to the cumulative store, recompute S_KG over the whole corpus
          5. MERGE the delta into Neo4j (no wipe)
        """
        try:
            reset = bool(self.config.clear_neo4j_on_build) if reset is None else bool(reset)
            print("\n=== Phase 1: Incremental Multi-PDF KG Construction ===", flush=True)
            if reset:
                store, registry, old_chunks = self._empty_store(), {}, []
            else:
                store, registry, old_chunks = self._load_store()

            new_items, skipped = [], []
            for item in file_paths_or_texts:
                h = self.document_hash(item)
                (skipped if h in registry else new_items).append(item)
                if h in registry:
                    print(f"[Phase 1] Skipping already-ingested document: {os.path.basename(str(item))[:60]}", flush=True)

            max_chunks = getattr(self.config, "max_chunks_per_doc", None)
            workers = int(getattr(self.config, "extraction_workers", 5))
            chunks = self.parse_and_chunk_documents(new_items, max_chunks_per_doc=max_chunks) if new_items else []
            chunks_by_id = {ch["chunk_id"]: ch["text"] for ch in chunks}

            results: List[Optional[List[Dict[str, Any]]]] = [None] * len(chunks)
            self._abort_reason = None
            failed: Dict[int, str] = {}
            if chunks:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    fut = {executor.submit(self.extract_from_chunk, ch): i for i, ch in enumerate(chunks)}
                    for done, f in enumerate(as_completed(fut), start=1):
                        i = fut[f]
                        try:
                            results[i] = f.result()
                        except FatalAPIError as fe:
                            if not self._abort_reason:
                                self._abort_reason = str(fe)
                                print(f"[Phase 1] FATAL API error - stopping extraction: {fe}", flush=True)
                        except _ChunkFailed as ce:
                            failed[i] = str(ce)
                        if progress_callback:
                            progress_callback(done, len(chunks), chunks[i]["chunk_id"])
                if self._abort_reason:
                    n_ok = sum(r is not None for r in results)
                    raise FatalAPIError(f"{self._abort_reason}\nKG build stopped before saving anything: {n_ok}/{len(chunks)} "
                                        "chunk extractions finished and are cached, so the next build only pays for the rest.")
                for i in sorted(failed):             # one sequential second chance for transient glitches
                    try:
                        results[i] = self.extract_from_chunk(chunks[i])
                        failed.pop(i)
                    except _ChunkFailed as ce:
                        failed[i] = str(ce)
                if len(failed) > 0.05 * len(chunks):
                    bad_docs = sorted({chunks[i]["source_name"] for i in failed})
                    raise RuntimeError(f"{len(failed)}/{len(chunks)} chunks could not be extracted (e.g. "
                                       f"{next(iter(failed.values()))}). Affected: {', '.join(bad_docs)[:300]}. "
                                       "Nothing was saved; run the build again - finished chunks are cached.")
            self._failed_chunks = [{"chunk_id": chunks[i]["chunk_id"], "error": e} for i, e in failed.items()]
            if failed:
                print(f"[Phase 1] WARNING: {len(failed)} chunk(s) could not be extracted and were left out "
                      f"(<= 5%, recorded in fidelity_state.failed_chunks)", flush=True)
            raw_claims = [c for r in results for c in (r or [])]

            existing_entities = {k: v for k, v in store["entities"].items()}
            if raw_claims:
                verified, canonical, grounding = self.validate_and_resolve_entities(raw_claims, chunks_by_id, existing_entities)
            else:
                verified, canonical, grounding = [], {}, []

            # ---- update cumulative store --------------------------------------------------
            touched: Dict[str, Dict[str, Any]] = {}
            for name, d in canonical.items():
                entry = {"type": d["type"], "aliases": sorted(d["aliases"]), "source_pdfs": sorted(d["source_pdfs"])}
                if store["entities"].get(name) != entry:
                    touched[name] = entry
                store["entities"][name] = entry
            store["claims"].extend(verified)
            store["grounding_scores"].extend(grounding)
            store["chunk_coverage"].update(self.chunk_coverage(chunks, verified))
            store["rejected"].extend(self._last_rejected if raw_claims else [])
            for item in new_items:
                registry[self.document_hash(item)] = {
                    "source_name": os.path.basename(str(item)) if os.path.exists(str(item)) else "inline_text",
                    "ingested_at": _utcnow(),
                    "chunks": sum(1 for c in chunks if c["document_id"] == f"DOC_{self.document_hash(item)[:10]}"),
                }
            known_ids = {c["chunk_id"] for c in old_chunks}
            all_chunks = old_chunks + [c for c in chunks if c["chunk_id"] not in known_ids]

            phi_prov, phi_cov, phi_topo, phi_cons, s_kg = self.aggregate_fidelity(store)
            shared = sorted(n for n, d in store["entities"].items() if len(d.get("source_pdfs", [])) > 1)
            rejected_by_reason = dict(Counter(r["reason"] for r in store["rejected"]))

            fidelity_state = {
                "overall_kg_accuracy_score": s_kg,
                "provenance_grounding_score": phi_prov,
                "semantic_coverage_score": phi_cov,
                "topological_coherence_score": phi_topo,
                "internal_consistency_score": phi_cons,
                "formula": "S_KG = (Phi_prov * Phi_cov * Phi_topo * Phi_cons)^(1/4)",
                "failed_chunks": getattr(self, "_failed_chunks", []),
                "documents_in_corpus": len(registry),
                "documents": sorted(v["source_name"] for v in registry.values()),
                "total_chunks_processed": len(all_chunks),
                "raw_triplets_extracted": len(store["grounding_scores"]),
                "verified_triplets_pushed": len(store["claims"]),
                "resolved_entity_count": len(store["entities"]),
                "cross_pdf_shared_entities_count": len(shared),
                "cross_pdf_shared_entities": shared,
                "latest_extracted_kg_path": self.config.extracted_kg_file_path,
                "last_run": {
                    "new_documents": [os.path.basename(str(i)) for i in new_items],
                    "skipped_documents": [os.path.basename(str(i)) for i in skipped],
                    "new_chunks": len(chunks), "new_raw_claims": len(raw_claims), "new_verified_claims": len(verified),
                    "new_or_updated_entities": len(touched),
                },
                "diagnostics": {
                    "claim_acceptance_rate": len(store["claims"]) / max(1, len(store["grounding_scores"])),
                    "rejected_by_reason": rejected_by_reason,
                    "accepted_by_evidence_match": dict(Counter(c.get("evidence_match", "verbatim") for c in store["claims"])),
                    "llm_calls": int(self._usage.get("calls", 0)),
                    "llm_prompt_tokens": int(self._usage.get("prompt_tokens", 0)),
                    "llm_completion_tokens": int(self._usage.get("completion_tokens", 0)),
                    "prompt_version": PROMPT_VERSION,
                },
                "timestamp": _utcnow(),
            }

            # ---- persist (atomic writes) -------------------------------------------------
            write_json_atomic(self.config.kg_store_path, store)
            write_json_atomic(self.config.corpus_registry_path, registry)
            write_json_atomic(self.config.latest_chunks_path, all_chunks)
            write_json_atomic(self.config.kg_fidelity_state_path, fidelity_state, indent=4)
            write_json_atomic(os.path.join(os.path.dirname(self.config.kg_fidelity_state_path), "rejected_claims.json"), store["rejected"])
            names = list(store["entities"].keys())
            write_json_atomic(self.config.extracted_kg_file_path, {
                "fidelity_metrics": fidelity_state,
                "nodes": [{"id": i, "name": n, **store["entities"][n]} for i, n in enumerate(names)],
                "edges": store["claims"],
            }, indent=2)

            pushed_ok = True
            if verified or touched or reset:
                delta_entities = touched if not reset else store["entities"]
                delta_claims = verified if not reset else store["claims"]
                pushed_ok = self.push_to_neo4j_cloud(delta_entities, delta_claims, s_kg, clear_existing=reset,
                                                     fidelity_state=fidelity_state)

            print(f"=== Phase 1 Complete | S_KG = {s_kg * 100:.2f}% | docs={len(registry)} "
                  f"(+{len(new_items)} new, {len(skipped)} skipped) | triples={len(store['claims'])} "
                  f"| entities={len(store['entities'])} | shared={len(shared)} ===\n", flush=True)

            return KGBuilderArtifact(
                total_chunks_processed=len(all_chunks),
                raw_triplets_extracted=len(store["grounding_scores"]),
                verified_triplets_pushed=len(store["claims"]),
                resolved_entity_count=len(store["entities"]),
                provenance_grounding_score=phi_prov,
                semantic_coverage_score=phi_cov,
                topological_coherence_score=phi_topo,
                internal_consistency_score=phi_cons,
                overall_kg_accuracy_score=s_kg,
                kg_fidelity_state_path=self.config.kg_fidelity_state_path,
                extracted_kg_file_path=self.config.extracted_kg_file_path,
                neo4j_cloud_pushed=pushed_ok,
                new_documents=[os.path.basename(str(i)) for i in new_items],
                skipped_documents=[os.path.basename(str(i)) for i in skipped],
            )
        except FatalAPIError:
            raise
        except Exception as e:
            raise KGAuditException(e, sys)

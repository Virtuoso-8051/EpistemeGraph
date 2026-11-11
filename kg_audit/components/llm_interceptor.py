"""
Phase 2 - Query interception (two-stage, prompt-version-pinned, fully cached).

Stage A  ANSWER     : the LLM answers the user exactly as it would in production, and verbalises
                      a confidence (Xiong et al., 2024). Modes:
                        closed_book - no context (audits the model's parametric knowledge)
                        grounded    - top-k corpus sentences retrieved by embedding similarity
                                      are supplied (audits a RAG system)
Stage B  EXTRACT    : a SEPARATE extraction call converts the answer text into the response
                      graph G' with claim polarity (ASSERTED / SPECULATIVE / REFUTED).
                      Decoupling matters: when one call both answers and lists its own claims,
                      the model can silently omit its fabrications from G'.

Dynamic prompting (no hard-coded domain wording):
  * the relation vocabulary is mined from the live KG (most frequent relation types),
  * entity naming hints are the KG entities nearest to the query+answer in embedding space,
  * grounded-mode evidence is retrieved from the indexed corpus.

Every LLM response is cached on disk under sha256(prompt_version | model | prompt), so re-running
a benchmark or reloading the UI costs $0.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from kg_audit.components.kg_builder import (
    PREFERRED_RELATIONS,
    RELATION_SYNONYMS,
    VAGUE_RELATIONS,
    _GENERIC_MENTIONS,
    KGBuilder,
)
from kg_audit.entity.artifact_entity import LLMInterceptorArtifact
from kg_audit.entity.config_entity import LLMInterceptorConfig
from kg_audit.exceptions.exception import KGAuditException
from kg_audit.utils.api_errors import FatalAPIError
from kg_audit.utils.llm import LLMClient, default_client
from kg_audit.logging.logger import logging
from kg_audit.utils.main_utils import embed, embed_many, read_json, sha256_text, write_json_atomic

PROMPT_VERSION = "intercept-v4"

ANSWER_SYSTEM = "You are a knowledgeable research assistant. Reply with a single valid JSON object."
ANSWER_USER = """Answer the question below as accurately and completely as you can.
{context_block}
Then report how confident you are that every factual statement in your answer is correct.

Return JSON:
{{"answer": "<your answer>",
  "confidence": <number between 0 and 1>,
  "confidence_reasoning": "<one sentence>"}}

QUESTION:
{query}"""

EXTRACT_SYSTEM = "You convert text into a knowledge graph. Reply with a single valid JSON object."
EXTRACT_USER = """Extract every technical factual claim made in the ANSWER as directed triples.

Rules
1. Extract claims from the ANSWER only - not from the question, not from your own knowledge.
2. Entities: specific technical names (methods, models, datasets, metrics, objectives, concepts).
   Never pronouns or generic phrases ("it", "the model", "this paper"). If a full name and an
   acronym both appear, use the full name as "source"/"target" and put the acronym in the alias field.
3. Relations: UPPER_SNAKE_CASE verbs. Prefer this vocabulary, which is the vocabulary of the
   reference knowledge graph: {relations}. Never use vague verbs such as RELATED_TO.
4. claim_type:
   ASSERTED    - the answer states it as true
   SPECULATIVE - the answer hedges ("may", "could", "one could extend ...") or extrapolates
   REFUTED     - the answer explicitly denies it ("X does not use Y")
5. Extract between 1 and 15 triples; include EVERY claim, also doubtful ones.
{entity_hints}
Return JSON:
{{"query_entities": ["..."],
  "triples": [{{"source": "...", "source_alias": "", "type": "USES", "target": "...",
               "target_alias": "", "claim_type": "ASSERTED"}}]}}

QUESTION (for entity naming only):
{query}

ANSWER:
{answer}"""


class LLMInterceptor:
    _entity_matrix_cache: Dict[str, Any] = {}

    def __init__(self, config: LLMInterceptorConfig, client: Optional[LLMClient] = None):
        try:
            self.config = config
            self.client = client or default_client()     # Claude or OpenAI, chosen per model name
            os.makedirs(config.query_cache_dir, exist_ok=True)
        except Exception as e:
            raise KGAuditException(e, sys)

    # ------------------------------------------------------------------ LLM call with cache
    def _cached_json_call(self, model: str, system: str, user: str, logprobs: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        With logprobs=True the call also returns token log-probabilities (no extra cost) and the cache stores
        `_token_nll` = mean negative log-likelihood of the generated tokens, a model-intrinsic uncertainty
        signal (Kadavath et al., 2022; Farquhar et al., 2024). Models that do not expose logprobs (e.g. some
        reasoning models) are retried without it and `_token_nll` is None.
        """
        key = sha256_text(f"{PROMPT_VERSION}|{model}|{system}|{user}")
        path = os.path.join(self.config.query_cache_dir, f"{key}.json")
        cached = read_json(path)
        if cached is not None:
            return cached, {"cache_hit": True, "latency_ms": 0.0, "prompt_tokens": 0, "completion_tokens": 0}

        parsed, meta = self.client.complete_json(model, system, user, temperature=self.config.temperature,
                                                 logprobs=logprobs, max_tokens=8000)
        if logprobs:      # Claude does not expose log-probabilities -> None (feature imputed downstream)
            parsed["_token_nll"] = meta.get("token_nll")
        write_json_atomic(path, parsed)
        return parsed, {"cache_hit": False, "latency_ms": meta.get("latency_ms", 0.0),
                        "prompt_tokens": meta.get("prompt_tokens", 0), "completion_tokens": meta.get("completion_tokens", 0)}

    # ------------------------------------------------------------------ dynamic prompt parts
    def _kg_store(self) -> Dict[str, Any]:
        return read_json(self.config.kg_store_path, default={}) or {}

    def dynamic_relation_vocabulary(self, top_n: int = 24) -> List[str]:
        """Most frequent non-vague relation types of the live KG, padded with the core ontology."""
        claims = self._kg_store().get("claims", [])
        freq = Counter(str(c.get("relation", "")).upper() for c in claims)
        mined = [r for r, _ in freq.most_common() if r and r not in VAGUE_RELATIONS][:top_n]
        return mined + [r for r in PREFERRED_RELATIONS if r not in mined][: max(0, top_n - len(mined))]

    def dynamic_entity_hints(self, text: str) -> str:
        """KG entities nearest (cosine) to the query+answer; tells the extractor the canonical names."""
        ents = self._kg_store().get("entities", {})
        if not ents:
            return ""
        names = list(ents.keys())
        key = sha256_text("|".join(names), 16)
        if LLMInterceptor._entity_matrix_cache.get("key") != key:   # encode the KG vocabulary once
            LLMInterceptor._entity_matrix_cache = {"key": key, "mat": embed_many(names)}
        sims = LLMInterceptor._entity_matrix_cache["mat"] @ embed(text)
        top = [names[i] for i in np.argsort(-sims)[: self.config.max_entity_hints] if sims[i] > 0.25]
        if not top:
            return ""
        shown = []
        for n in top:
            short = [a for a in ents[n].get("aliases", []) if a.lower() != n.lower() and 2 <= len(a) <= 10 and a.isupper()]
            shown.append(f"{n} ({short[0]})" if short else n)
        return ("6. If the answer mentions one of these reference entities, use exactly this canonical name "
                "(other entities are allowed): " + "; ".join(shown))

    def retrieve_context(self, query: str, k: int = 8) -> List[str]:
        """Dense retrieval over the disk-cached corpus sentence index shared with Phase 3."""
        from kg_audit.components.truth_evaluator import CorpusIndex  # local import: avoids a cycle
        idx = CorpusIndex.get(self.config.latest_chunks_path, self.config.cache_dir)
        if idx.sentence_emb.shape[0] == 0:
            return []
        sims = idx.sentence_emb @ embed(query).astype(np.float32)
        return [idx.sentences[i] for i in np.argsort(-sims)[:k]]

    # ------------------------------------------------------------------ normalisation
    @staticmethod
    def _normalize_entity(raw_name: Any, raw_alias: Any = "") -> Tuple[str, str]:
        name_str, alias_str = KGBuilder._split_inline_parenthetical(str(raw_name or ""), str(raw_alias or ""))
        clean_name = KGBuilder._clean_mention(name_str)
        clean_alias = KGBuilder._clean_mention(alias_str) if alias_str else ""
        if clean_alias and len(clean_alias) > len(clean_name) and KGBuilder._is_short_token(clean_name):
            clean_name, clean_alias = clean_alias, clean_name
        return clean_name, clean_alias

    @staticmethod
    def _normalize_relation(rel: Any) -> str:
        r = re.sub(r"[^A-Z0-9]+", "_", str(rel or "").upper()).strip("_")
        r = RELATION_SYNONYMS.get(r, r) or "USES"
        return "USES" if r in VAGUE_RELATIONS else r

    def build_response_graph(self, raw_query_entities, raw_triples) -> Dict[str, Any]:
        aliases: Dict[str, set] = {}

        def register(raw_n: Any, raw_a: Any = "") -> Optional[str]:
            name, alias = self._normalize_entity(raw_n, raw_a)
            if not name or name.lower() in _GENERIC_MENTIONS:
                return None
            for existing, al in aliases.items():
                al_low = {a.lower() for a in al}
                if (name.lower() == existing.lower() or name.lower() in al_low
                        or KGBuilder._is_acronym_match(name, existing)
                        or (alias and (alias.lower() == existing.lower() or KGBuilder._is_acronym_match(alias, existing)))):
                    al.update(x for x in (name, alias) if x)
                    return existing
            aliases[name] = {x for x in (name, alias) if x}
            return name

        q_ents = [e for e in (register(q) for q in raw_query_entities or []) if e]
        triples, seen, counts = [], set(), Counter()
        for t in raw_triples or []:
            if not isinstance(t, dict):
                continue
            s = register(t.get("source"), t.get("source_alias", ""))
            o = register(t.get("target"), t.get("target_alias", ""))
            if not s or not o or s == o:
                continue
            rel = self._normalize_relation(t.get("type"))
            ct = str(t.get("claim_type", "ASSERTED")).upper()
            ct = ct if ct in {"ASSERTED", "SPECULATIVE", "REFUTED"} else "ASSERTED"
            key = (s.lower(), rel, o.lower(), ct)
            if key in seen:
                continue
            seen.add(key)
            counts[ct] += 1
            triples.append({"source": s, "target": o, "type": rel, "claim_type": ct})

        names = list(aliases.keys())
        idx = {n: i for i, n in enumerate(names)}
        return {
            "entities": names,
            "query_entities": list(dict.fromkeys(q_ents)),
            "nodes": [{"id": i, "name": n, "aliases": sorted(aliases[n])} for i, n in enumerate(names)],
            "edges": [{"source": idx[t["source"]], "target": idx[t["target"]], "type": t["type"],
                       "claim_type": t["claim_type"], "weight": 1.0} for t in triples],
            "triples": triples,
            "polarity_counts": dict(counts),
        }

    # ------------------------------------------------------------------ main entry
    def intercept_and_generate(
        self,
        user_query: str,
        source_documents: Optional[List[str]] = None,
        mode: str = "closed_book",
        precomputed_answer: Optional[Dict[str, Any]] = None,
    ) -> Tuple[LLMInterceptorArtifact, Dict[str, Any]]:
        """
        precomputed_answer lets the benchmark runner inject an answer produced elsewhere
        (e.g. a different LLM under audit) and still run the identical extraction stage.
        """
        try:
            source_documents = list(source_documents or [])
            context = source_documents + (self.retrieve_context(user_query) if mode == "grounded" else [])
            ctx_block = ("\nUse the following context where relevant:\n\"\"\"\n" + "\n".join(context) + "\n\"\"\"\n") if context else ""

            if precomputed_answer is None:
                ans, a_meta = self._cached_json_call(self.config.llm_model_name, ANSWER_SYSTEM,
                                                     ANSWER_USER.format(context_block=ctx_block, query=user_query), logprobs=True)
            else:
                ans, a_meta = precomputed_answer, {"cache_hit": True, "latency_ms": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
            answer = str(ans.get("answer", "")).strip()
            try:
                conf = float(ans.get("confidence", ans.get("self_reported_accuracy", 0.5)))
            except (TypeError, ValueError):
                conf = 0.5
            conf = float(np.clip(conf / 100.0 if conf > 1.0 else conf, 0.0, 1.0))
            reason = str(ans.get("confidence_reasoning", "")).strip()

            ext, e_meta = self._cached_json_call(
                self.config.extractor_model_name, EXTRACT_SYSTEM,
                EXTRACT_USER.format(relations=", ".join(self.dynamic_relation_vocabulary()),
                                    entity_hints=self.dynamic_entity_hints(f"{user_query} {answer}"),
                                    query=user_query, answer=answer))
            graph = self.build_response_graph(ext.get("query_entities", []), ext.get("triples", []))
            n_trip = max(1, len(graph["triples"]))
            pc = graph["polarity_counts"]

            metadata = {
                "user_query": user_query,
                "llm_response": answer,
                "llm_extraction_confidence": conf,
                "llm_confidence_reasoning": reason,
                "llm_model_name": self.config.llm_model_name,
                "extractor_model_name": self.config.extractor_model_name,
                "mode": mode,
                "prompt_version": PROMPT_VERSION,
                "answer_call": a_meta,
                "extract_call": e_meta,
                "asserted_claim_count": pc.get("ASSERTED", 0),
                "speculative_claim_count": pc.get("SPECULATIVE", 0),
                "refuted_claim_count": pc.get("REFUTED", 0),
                "speculative_ratio": round(pc.get("SPECULATIVE", 0) / n_trip, 4),
                "token_nll": ans.get("_token_nll"),
                "query_entities": graph["query_entities"],
            }
            payload = {
                "user_query": user_query, "mode": mode, "llm_response": answer,
                "llm_self_reported_accuracy": conf, "llm_confidence_reasoning": reason,
                "extracted_query_entities": graph["query_entities"],
                "extracted_response_entities": graph["entities"],
                "extracted_response_triples": graph["triples"],
                "context_used": context, "telemetry_metadata": metadata,
            }
            write_json_atomic(self.config.captured_payload_file_path, payload, indent=4)

            artifact = LLMInterceptorArtifact(
                user_query=user_query, llm_response=answer, llm_self_reported_accuracy=conf,
                llm_confidence_reasoning=reason, source_documents=context,
                captured_payload_path=self.config.captured_payload_file_path, mode=mode,
            )
            graph_payload = {
                "entities": list(dict.fromkeys(graph["query_entities"] + graph["entities"])),
                "query_entities": graph["query_entities"],
                "nodes": graph["nodes"], "edges": graph["edges"], "triples": graph["triples"],
                "llm_self_reported_accuracy": conf, "metadata": metadata,
            }
            logging.info(f"Intercepted '{user_query[:60]}' -> {len(graph['triples'])} triples "
                         f"(answer cache={a_meta['cache_hit']}, extract cache={e_meta['cache_hit']})")
            return artifact, graph_payload
        except FatalAPIError:
            raise                                # keep the type: callers stop the whole stage on it
        except Exception as e:
            raise KGAuditException(e, sys)

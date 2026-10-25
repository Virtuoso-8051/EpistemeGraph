import os
import re
import sys
import json
import time
import unicodedata
from typing import List, Dict, Any, Tuple, Optional
from openai import OpenAI

from kg_audit.entity.config_entity import LLMInterceptorConfig
from kg_audit.entity.artifact_entity import LLMInterceptorArtifact
from kg_audit.components.kg_builder import (
    PREFERRED_RELATIONS,
    RELATION_SYNONYMS,
    VAGUE_RELATIONS,
    _GENERIC_MENTIONS,
    KGBuilder,
)
from kg_audit.exceptions.exception import KGAuditException
from kg_audit.logging.logger import logging


INTERCEPTION_SYSTEM_PROMPT = """You are a rigorous scientific AI assistant under epistemic audit.
You must answer the user's question and extract the exact directed knowledge graph induced by your answer.
Output strictly valid JSON with no markdown wrappers."""

INTERCEPTION_USER_PROMPT = """Answer the user's research query thoroughly and accurately.
Along with your natural language answer, extract the structured Knowledge Graph (G') representing every technical claim made in your response.

Rules for Knowledge Graph (G') Extraction:
1. "self_reported_accuracy": Your estimated probability (0.00 to 1.00) that your answer is completely factually accurate.
2. "confidence_reasoning": One concise sentence explaining why you assigned that confidence score.
3. "query_entities": Key technical entities explicitly mentioned in the user's query (use canonical singular names, e.g., "Large-scale Information Network Embedding", "node2vec", "breadth-first search").
4. "response_entities": All key technical entities in both the query and your answer. Never use generic pronouns ("it", "this model", "the paper").
5. "response_triples": Extract 4 to 10 directed technical claims from your answer.
   - If an entity has both a full name and an acronym (e.g., "Breadth-First Search" and "BFS"), provide the full name as "source"/"target" and the acronym in "source_alias"/"target_alias".
   - Prefer normalized relation verbs from: [__RELATIONS__]. Avoid vague verbs like RELATED_TO.
   - "claim_type":
       * "ASSERTED": A positive factual claim you assert as true in the original algorithm/literature.
       * "SPECULATIVE": A hypothetical extension, hybrid design, or extrapolation beyond the original formulation.
       * "REFUTED": A relationship you explicitly state is false or NOT used by the method (e.g., "X does NOT use Y").
   - "weight": Your confidence (0.0 to 1.0) in this specific triple.

__KNOWN_ENTITIES_HINT__

Return ONLY valid JSON matching this schema:
{
  "answer": "Detailed technical response...",
  "self_reported_accuracy": 0.95,
  "confidence_reasoning": "One-sentence explanation of confidence.",
  "query_entities": ["Entity1", "Entity2"],
  "response_entities": ["Entity1", "Entity2", "Entity3"],
  "response_triples": [
    {
      "source": "Full Entity Name 1",
      "source_alias": "ACR1",
      "target": "Full Entity Name 2",
      "target_alias": "ACR2",
      "type": "USES",
      "claim_type": "ASSERTED",
      "weight": 0.95
    }
  ]
}

CONTEXT (if provided):
\"\"\"__CONTEXT__\"\"\"

USER QUERY:
\"\"\"__QUERY__\"\"\"
"""


class LLMInterceptor:
    def __init__(self, config: LLMInterceptorConfig):
        try:
            self.config = config
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            raise KGAuditException(e, sys)

    def _load_canonical_entity_hints(self, user_query: str, max_hints: int = 30) -> str:
        """
        Loads canonical entity names from the Phase 1 KG state (if available) that share tokens
        with the user query so the extractor uses consistent canonical naming for known entities,
        without restricting the LLM from introducing new/hallucinated entities.
        """
        try:
            fidelity_path = os.path.join("final_model", "kg_fidelity_state.json")
            if not os.path.exists(fidelity_path):
                return ""
            with open(fidelity_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            kg_path = state.get("latest_extracted_kg_path", "")
            if not kg_path or not os.path.exists(kg_path):
                return ""
            with open(kg_path, "r", encoding="utf-8") as f:
                kg_data = json.load(f)

            nodes = kg_data.get("nodes", [])
            if not nodes:
                return ""

            q_tokens = {
                w.lower() for w in re.sub(r"[^\w\s]", " ", user_query).split() if len(w) >= 2
            }
            scored = []
            for n in nodes:
                name = str(n.get("name", ""))
                aliases = [str(a) for a in n.get("aliases", [])]
                all_toks = {
                    w.lower()
                    for phrase in ([name] + aliases)
                    for w in re.sub(r"[^\w\s]", " ", phrase).split()
                    if len(w) >= 2
                }
                overlap = len(q_tokens & all_toks)
                scored.append((overlap, len( n.get("source_pdfs", [])), name, aliases))

            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            selected = []
            for _, _, name, aliases in scored[:max_hints]:
                short_als = [a for a in aliases if a.lower() != name.lower() and len(a) <= 10]
                if short_als:
                    selected.append(f"{name} (alias: {short_als[0]})")
                else:
                    selected.append(name)

            if not selected:
                return ""
            return (
                "When referring to concepts that match these canonical corpus entities, "
                "use their exact canonical name (you may still introduce other entities if needed):\n"
                + ", ".join(selected)
            )
        except Exception:
            return ""

    @staticmethod
    def _normalize_entity(raw_name: Any, raw_alias: Any = "") -> Tuple[str, str]:
        name_str, alias_str = KGBuilder._split_inline_parenthetical(
            str(raw_name or ""), str(raw_alias or "")
        )
        clean_name = KGBuilder._clean_mention(name_str)
        clean_alias = KGBuilder._clean_mention(alias_str) if alias_str else ""
        if clean_alias and len(clean_alias) > len(clean_name) and KGBuilder._is_short_token(clean_name):
            clean_name, clean_alias = clean_alias, clean_name
        return clean_name, clean_alias

    @staticmethod
    def _normalize_relation(rel: Any) -> str:
        r = re.sub(r"[^A-Z0-9]+", "_", str(rel or "").upper()).strip("_")
        norm_r = RELATION_SYNONYMS.get(r, r) or "USES"
        if norm_r in VAGUE_RELATIONS:
            return "USES"
        return norm_r

    def intercept_and_generate(
        self,
        user_query: str,
        source_documents: Optional[List[str]] = None
    ) -> Tuple[LLMInterceptorArtifact, Dict[str, Any]]:
        """
        Phase 2 Single-Pass Query Interceptor:
        1. Sends user query (+ optional context) to OpenAI LLM.
        2. Captures natural language response, self-reported accuracy score (P_LLM), and rationale.
        3. Extracts ontology-normalized induced response graph G' with claim polarity & telemetry metadata.
        """
        try:
            logging.info(f"Intercepting user query: '{user_query}'")
            if source_documents is None:
                source_documents = []

            context_str = "\n".join(source_documents)
            entity_hints = self._load_canonical_entity_hints(user_query)

            prompt = (
                INTERCEPTION_USER_PROMPT
                .replace("__RELATIONS__", ", ".join(PREFERRED_RELATIONS))
                .replace("__KNOWN_ENTITIES_HINT__", entity_hints)
                .replace("__CONTEXT__", context_str)
                .replace("__QUERY__", user_query)
            )

            import hashlib
            query_cache_dir = os.path.join("Artifacts", "query_cache")
            os.makedirs(query_cache_dir, exist_ok=True)
            q_hash = hashlib.sha256(
                f"v2_{self.config.llm_model_name}_{user_query.strip()}_{context_str}".encode("utf-8")
            ).hexdigest()
            q_cache_file = os.path.join(query_cache_dir, f"{q_hash}.json")

            t0 = time.perf_counter()
            if os.path.exists(q_cache_file):
                with open(q_cache_file, "r", encoding="utf-8") as f:
                    parsed = json.load(f)
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                usage = None
                logging.info(f"[Query Cache HIT] Loaded cached LLM response for '{user_query[:60]}...' ($0 API cost)")
            else:
                response = self.client.chat.completions.create(
                    model=self.config.llm_model_name,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": INTERCEPTION_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature
                )
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                raw_content = response.choices[0].message.content.strip()
                match = re.search(r"\{.*\}", raw_content, re.S)
                parsed = json.loads(match.group(0) if match else raw_content)
                usage = getattr(response, "usage", None)
                with open(q_cache_file, "w", encoding="utf-8") as f:
                    json.dump(parsed, f, indent=2)

            llm_response = str(parsed.get("answer", "")).strip()
            self_acc = float(parsed.get("self_reported_accuracy", 0.90))
            self_acc = max(0.01, min(0.999, self_acc))
            conf_reason = str(parsed.get("confidence_reasoning", "Self-estimated by language model.")).strip()

            raw_q_ents = parsed.get("query_entities", [])
            raw_r_ents = parsed.get("response_entities", [])
            raw_triples = parsed.get("response_triples", [])

            # Canonicalize entities and track aliases across the response graph G'
            entity_aliases: Dict[str, set] = {}
            norm_query_entities: List[str] = []

            def register_entity(raw_n: Any, raw_a: Any = "") -> Optional[str]:
                c_name, c_alias = self._normalize_entity(raw_n, raw_a)
                if not c_name or c_name.lower() in _GENERIC_MENTIONS:
                    return None
                # Check if matches an already registered entity by case or acronym
                for existing, al_set in entity_aliases.items():
                    if c_name.lower() == existing.lower() or c_name.lower() in {a.lower() for a in al_set}:
                        if c_alias:
                            al_set.add(c_alias)
                        return existing
                    if KGBuilder._is_acronym_match(c_name, existing) or (
                        c_alias and (
                            c_alias.lower() == existing.lower()
                            or KGBuilder._is_acronym_match(c_alias, existing)
                        )
                    ):
                        al_set.add(c_name)
                        if c_alias:
                            al_set.add(c_alias)
                        return existing
                als = {c_name}
                if c_alias:
                    als.add(c_alias)
                init = KGBuilder._initials(c_name)
                if 2 <= len(init) <= 6:
                    als.add(init)
                entity_aliases[c_name] = als
                return c_name

            for qe in raw_q_ents:
                cn = register_entity(qe)
                if cn and cn not in norm_query_entities:
                    norm_query_entities.append(cn)

            for re_ent in raw_r_ents:
                register_entity(re_ent)

            normalized_triples: List[Dict[str, Any]] = []
            seen_edges = set()
            speculative_count = 0
            refuted_count = 0
            asserted_count = 0

            for t in raw_triples:
                if not isinstance(t, dict):
                    continue
                src_name = register_entity(t.get("source"), t.get("source_alias", ""))
                tgt_name = register_entity(t.get("target"), t.get("target_alias", ""))
                if not src_name or not tgt_name or src_name == tgt_name:
                    continue

                rel_type = self._normalize_relation(t.get("type", "USES"))
                claim_type = str(t.get("claim_type", "ASSERTED")).strip().upper()
                if claim_type not in {"ASSERTED", "SPECULATIVE", "REFUTED"}:
                    claim_type = "ASSERTED"

                try:
                    w = float(t.get("weight", 1.0))
                except (TypeError, ValueError):
                    w = 1.0

                edge_key = (src_name.lower(), rel_type, tgt_name.lower())
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                if claim_type == "SPECULATIVE":
                    speculative_count += 1
                elif claim_type == "REFUTED":
                    refuted_count += 1
                else:
                    asserted_count += 1

                normalized_triples.append({
                    "source": src_name,
                    "source_aliases": sorted(entity_aliases.get(src_name, {src_name})),
                    "target": tgt_name,
                    "target_aliases": sorted(entity_aliases.get(tgt_name, {tgt_name})),
                    "type": rel_type,
                    "claim_type": claim_type,
                    "weight": w
                })

            combined_entities = list(entity_aliases.keys())
            response_nodes = [
                {
                    "id": idx,
                    "name": ent,
                    "aliases": sorted(entity_aliases[ent])
                }
                for idx, ent in enumerate(combined_entities)
            ]
            node_map = {ent: idx for idx, ent in enumerate(combined_entities)}

            response_edges = []
            for triple in normalized_triples:
                src_id = node_map.get(triple["source"])
                tgt_id = node_map.get(triple["target"])
                if src_id is not None and tgt_id is not None:
                    response_edges.append({
                        "source": src_id,
                        "target": tgt_id,
                        "type": triple["type"],
                        "claim_type": triple["claim_type"],
                        "weight": triple["weight"]
                    })

            total_triples = max(1, len(normalized_triples))
            
            # THE FIX: Added user_query to metadata so truth_evaluator.py can read it
            metadata = {
                "user_query": user_query,
                "llm_extraction_confidence": self_acc,
                "llm_confidence_reasoning": conf_reason,
                "llm_model_name": self.config.llm_model_name,
                "response_latency_ms": latency_ms,
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
                "asserted_claim_count": asserted_count,
                "speculative_claim_count": speculative_count,
                "refuted_claim_count": refuted_count,
                "speculative_ratio": round(speculative_count / total_triples, 4),
                "query_entities": norm_query_entities,
            }

            interception_payload = {
                "user_query": user_query,
                "llm_response": llm_response,
                "llm_self_reported_accuracy": self_acc,
                "llm_confidence_reasoning": conf_reason,
                "extracted_query_entities": norm_query_entities,
                "extracted_response_entities": combined_entities,
                "extracted_response_triples": normalized_triples,
                "telemetry_metadata": metadata,
                "source_documents": source_documents
            }

            export_path = self.config.captured_payload_file_path
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(interception_payload, f, indent=4)

            artifact = LLMInterceptorArtifact(
                user_query=user_query,
                llm_response=llm_response,
                llm_self_reported_accuracy=self_acc,
                llm_confidence_reasoning=conf_reason,
                source_documents=source_documents,
                captured_payload_path=export_path
            )

            graph_payload_resp = {
                "entities": combined_entities,
                "query_entities": norm_query_entities,
                "nodes": response_nodes,
                "edges": response_edges,
                "llm_self_reported_accuracy": self_acc,
                "metadata": metadata
            }

            return artifact, graph_payload_resp

        except Exception as e:
            raise KGAuditException(e, sys)
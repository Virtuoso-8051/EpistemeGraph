# Phase 3: Intercept query and features LLM responses
import os
import sys
import json
from typing import List, Dict, Any, Tuple
from openai import OpenAI

from kg_audit.entity.config_entity import LLMInterceptorConfig
from kg_audit.entity.artifact_entity import LLMInterceptorArtifact
from kg_audit.exceptions.exception import KGAuditException
from kg_audit.logging.logger import logging


class LLMInterceptor:
    def __init__(self, config: LLMInterceptorConfig):
        try:
            self.config = config
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            raise KGAuditException(e, sys)

    def intercept_and_generate(
        self,
        user_query: str,
        source_documents: List[str] = None
    ) -> Tuple[LLMInterceptorArtifact, Dict[str, Any]]:
        """
        Phase 2 Single-Pass Interceptor:
        1. Sends user query (+ optional context snippets) to OpenAI ChatGPT.
        2. Requires ChatGPT to output its natural language response AND its own
           self-reported accuracy score (0.0 to 1.0) + confidence reasoning.
        3. Extracts query & response entities and directed triples for Graph G'.
        """
        try:
            logging.info(f"Intercepting user query: '{user_query}'")
            if source_documents is None:
                source_documents = []

            context_str = "\n".join(source_documents)
            prompt = f"""
            Answer the user's query accurately based on the uploaded research context (if provided) or your training knowledge.
            In addition to your answer, you MUST provide:
            1. "self_reported_accuracy": Your honest estimated probability (between 0.00 and 1.00) that your answer is 100% factually accurate.
            2. "confidence_reasoning": A one-sentence explanation of why you assigned that accuracy score.
            3. "query_entities": Key entities mentioned in the user's question.
            4. "response_entities": All key entities mentioned in your answer and the question.
            5. "response_triples": Directed factual relationships asserted in your answer.

            Return ONLY valid JSON matching this exact structure:
            {{
                "answer": "Your detailed natural language response here...",
                "self_reported_accuracy": 0.94,
                "confidence_reasoning": "Why you believe this answer is accurate.",
                "query_entities": ["Entity1", "Entity2"],
                "response_entities": ["Entity1", "Entity2", "Entity3"],
                "response_triples": [
                    {{"source": "Entity1", "target": "Entity2", "type": "RELATION_TYPE", "weight": 1.0}}
                ]
            }}

            CONTEXT:
            \"\"\"{context_str}\"\"\"

            USER QUERY:
            \"\"\"{user_query}\"\"\"
            """

            response = self.client.chat.completions.create(
                model=self.config.llm_model_name,
                messages=[
                    {"role": "system", "content": "You output strictly raw JSON with no markdown wrappers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature
            )

            raw_content = response.choices[0].message.content.strip()
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]

            parsed = json.loads(raw_content.strip())

            llm_response = str(parsed.get("answer", ""))
            self_acc = float(parsed.get("self_reported_accuracy", 0.90))
            conf_reason = str(parsed.get("confidence_reasoning", "Self-estimated by language model."))
            query_entities = parsed.get("query_entities", [])
            resp_entities = parsed.get("response_entities", [])
            resp_triples = parsed.get("response_triples", [])

            combined_entities = list(dict.fromkeys(query_entities + resp_entities))
            for t in resp_triples:
                if t.get("source") and t["source"] not in combined_entities:
                    combined_entities.append(t["source"])
                if t.get("target") and t["target"] not in combined_entities:
                    combined_entities.append(t["target"])

            response_nodes = [{"id": idx, "name": ent} for idx, ent in enumerate(combined_entities)]
            node_map = {ent: idx for idx, ent in enumerate(combined_entities)}

            response_edges = []
            for triple in resp_triples:
                src_id = node_map.get(triple.get("source"))
                tgt_id = node_map.get(triple.get("target"))
                if src_id is not None and tgt_id is not None:
                    response_edges.append({
                        "source": src_id,
                        "target": tgt_id,
                        "type": str(triple.get("type", "RELATED_TO")).upper().replace(" ", "_"),
                        "weight": float(triple.get("weight", 1.0))
                    })

            interception_payload = {
                "user_query": user_query,
                "llm_response": llm_response,
                "llm_self_reported_accuracy": self_acc,
                "llm_confidence_reasoning": conf_reason,
                "extracted_query_entities": query_entities,
                "extracted_response_entities": combined_entities,
                "extracted_response_triples": resp_triples,
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
                "nodes": response_nodes,
                "edges": response_edges,
                "llm_self_reported_accuracy": self_acc
            }

            return artifact, graph_payload_resp

        except Exception as e:
            raise KGAuditException(e, sys)
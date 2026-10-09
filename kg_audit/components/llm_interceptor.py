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

    def extract_entities_and_relations(self, text: str) -> Dict[str, Any]:
        """
        Uses LLM structured extraction to parse entities and triple relations 
        from the generated response to build the induced response graph G'.
        """
        try:
            extraction_prompt = f"""
            You are a Knowledge Graph extraction engine. Extract all primary entities and directed triple relations from the following text.
            
            Return ONLY a valid JSON object matching this schema:
            {{
                "entities": ["entity1", "entity2"],
                "triples": [
                    {{"source": "entity1", "target": "entity2", "type": "RELATION_TYPE", "weight": 1.0}}
                ]
            }}

            Text:
            "{text}"
            """

            response = self.client.chat.completions.create(
                model=self.config.llm_model_name,
                messages=[
                    {"role": "system", "content": "You output strictly raw JSON with no markdown formatting."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.0
            )

            raw_content = response.choices[0].message.content.strip()
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
            
            parsed_payload = json.loads(raw_content.strip())
            return parsed_payload

        except Exception as e:
            logging.warning(f"Entity/relation extraction failed: {str(e)}. Returning empty graph structures.")
            return {"entities": [], "triples": []}

    def intercept_and_generate(
        self, 
        user_query: str, 
        source_documents: List[str] = None
    ) -> Tuple[LLMInterceptorArtifact, Dict[str, Any]]:
        try:
            logging.info(f"Intercepting user query: '{user_query}'")
            if source_documents is None:
                source_documents = []

            context_str = "\n".join(source_documents)
            system_prompt = "You are an AI assistant answering questions based on the provided source documents."
            user_prompt = f"Context:\n{context_str}\n\nQuery: {user_query}" if context_str else user_query

            response = self.client.chat.completions.create(
                model=self.config.llm_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature
            )

            llm_response = response.choices[0].message.content
            logging.info("LLM response intercepted successfully.")

            # Extract entities and triples from query + response
            query_extraction = self.extract_entities_and_relations(user_query)
            response_extraction = self.extract_entities_and_relations(llm_response)

            combined_entities = list(set(query_extraction.get("entities", []) + response_extraction.get("entities", [])))

            # Transform extracted triples into formatted nodes and edges for G'
            response_nodes = [{"id": idx, "name": ent} for idx, ent in enumerate(combined_entities)]
            node_map = {ent: idx for idx, ent in enumerate(combined_entities)}

            response_edges = []
            for triple in response_extraction.get("triples", []):
                src_id = node_map.get(triple.get("source"))
                tgt_id = node_map.get(triple.get("target"))
                if src_id is not None and tgt_id is not None:
                    response_edges.append({
                        "source": src_id,
                        "target": tgt_id,
                        "type": triple.get("type", "RELATED_TO"),
                        "weight": float(triple.get("weight", 1.0))
                    })

            interception_payload = {
                "user_query": user_query,
                "llm_response": llm_response,
                "extracted_query_entities": query_extraction.get("entities", []),
                "extracted_response_entities": response_extraction.get("entities", []),
                "extracted_response_triples": response_extraction.get("triples", []),
                "source_documents": source_documents
            }

            export_path = self.config.captured_payload_file_path
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, "w") as f:
                json.dump(interception_payload, f, indent=4)

            artifact = LLMInterceptorArtifact(
                user_query=user_query,
                llm_response=llm_response,
                source_documents=source_documents,
                captured_payload_path=export_path
            )

            graph_payload_resp = {
                "entities": combined_entities,
                "nodes": response_nodes,
                "edges": response_edges
            }

            return artifact, graph_payload_resp

        except Exception as e:
            raise KGAuditException(e, sys)
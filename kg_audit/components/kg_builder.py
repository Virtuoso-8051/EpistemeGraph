import os
import sys
import glob
import json
import hashlib
import networkx as nx
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from neo4j import GraphDatabase
from pypdf import PdfReader

from kg_audit.entity.config_entity import KGBuilderConfig
from kg_audit.entity.artifact_entity import KGBuilderArtifact
from kg_audit.exceptions.exception import KGAuditException
from kg_audit.logging.logger import logging

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class KGBuilder:
    """
    Phase 1 Multi-Document Autonomous Engine:
    - Scans multiple PDFs from data/uploads/ (or file list)
    - Uses SHA-256 Disk Caching so repeated runs cost $0.00 in OpenAI tokens
    - Dynamic Emergent Ontology (no hardcoded entity/relation lists)
    - Cross-PDF Entity Resolution (Acronym Initials Matcher + Embedding Cosine Similarity)
    - Stores multi-PDF provenance arrays (source_pdfs) on every Node and Edge in Neo4j AuraDB Cloud
    """

    def __init__(self, config: KGBuilderConfig):
        try:
            self.config = config
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            os.makedirs(self.config.llm_cache_dir, exist_ok=True)
            self.embedder = None
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
                except Exception as ex:
                    logging.warning(f"Fallback to deterministic n-gram embeddings: {ex}")
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
        tokens = [clean[i:i+3] for i in range(max(1, len(clean) - 2))]
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[idx] += sign
        return vec / (np.linalg.norm(vec) + 1e-12)

    @staticmethod
    def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-12
        return float(np.dot(v1, v2) / denom)

    @staticmethod
    def _is_acronym_match(short_str: str, long_str: str) -> bool:
        """
        Deterministically detects if one entity mention is an acronym of another across PDFs.
        Example: "AI" <-> "Artificial Intelligence", "KG" <-> "Knowledge Graph".
        """
        s1 = short_str.strip().upper().replace(".", "")
        s2 = long_str.strip().upper().replace(".", "")
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        if len(s1) < 2 or len(s1) > 8 or " " in s1:
            return False
        stop_words = {"OF", "THE", "AND", "IN", "FOR", "TO", "A", "AN"}
        words = [w for w in s2.replace("-", " ").split() if w and w not in stop_words]
        if len(words) < 2:
            return False
        initials = "".join(w[0] for w in words)
        return s1 == initials

    # ------------------------------------------------------------------
    # 1. MULTI-PDF PARSER & CHUNKER
    # ------------------------------------------------------------------
    def parse_and_chunk_documents(self, file_paths_or_texts: List[str]) -> List[Dict[str, Any]]:
        try:
            chunks: List[Dict[str, Any]] = []
            for doc_idx, item in enumerate(file_paths_or_texts):
                doc_id = f"DOC_{doc_idx + 1:03d}"
                raw_text = ""

                if os.path.exists(item) and item.lower().endswith(".pdf"):
                    logging.info(f"Parsing PDF [{doc_id}]: {item}")
                    reader = PdfReader(item)
                    pages_text = [page.extract_text() or "" for page in reader.pages]
                    raw_text = "\n".join(pages_text)
                    source_name = os.path.basename(item)
                elif os.path.exists(item):
                    with open(item, "r", encoding="utf-8", errors="ignore") as f:
                        raw_text = f.read()
                    source_name = os.path.basename(item)
                else:
                    raw_text = str(item)
                    source_name = f"inline_text_{doc_idx + 1}"

                clean_text = " ".join(raw_text.split())
                if not clean_text:
                    continue

                step = max(200, self.config.chunk_size - self.config.chunk_overlap)
                chunk_counter = 1
                for start in range(0, len(clean_text), step):
                    segment = clean_text[start:start + self.config.chunk_size]
                    if len(segment) < 40 and chunks:
                        continue
                    chunks.append({
                        "document_id": doc_id,
                        "source_name": source_name,
                        "chunk_id": f"{doc_id}_CH_{chunk_counter:03d}",
                        "text": segment
                    })
                    chunk_counter += 1

            logging.info(f"Created {len(chunks)} total chunks across {len(file_paths_or_texts)} documents.")
            return chunks
        except Exception as e:
            raise KGAuditException(e, sys)

    # ------------------------------------------------------------------
    # 2. CACHED LLM DYNAMIC ONTOLOGY EXTRACTOR (SAVES OPENAI CREDITS)
    # ------------------------------------------------------------------
    def extract_from_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            # Check local disk cache first so re-running never bills OpenAI twice
            chunk_hash = hashlib.sha256(
                f"{self.config.llm_model_name}_{chunk['text']}".encode("utf-8")
            ).hexdigest()
            cache_file = os.path.join(self.config.llm_cache_dir, f"{chunk_hash}.json")

            if os.path.exists(cache_file):
                logging.info(f"Loaded cached extraction for {chunk['chunk_id']} ($0 API cost)")
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_claims = json.load(f)
                for c in cached_claims:
                    c["document_id"] = chunk["document_id"]
                    c["chunk_id"] = chunk["chunk_id"]
                    c["source_name"] = chunk["source_name"]
                return cached_claims

            prompt = f"""
            You are an expert scientific Knowledge Graph extractor.
            Extract explicit factual triplets from the text chunk below.

            Rules:
            1. Extract ONLY facts explicitly supported by the text.
            2. Dynamically assign the most accurate domain Entity Type in UPPER_SNAKE_CASE (e.g., ALGORITHM, MODEL, METRIC, CHEMICAL, PERSON, ORGANIZATION, CONCEPT).
            3. If an entity is an acronym (e.g., "AI", "KG", "LLM"), write its expanded canonical name if known (e.g., "Artificial Intelligence"), and put the short form in "alias".
            4. Normalize relation names to concise UPPER_SNAKE_CASE (e.g., USES, OUTPERFORMS, PART_OF, DEFINED_AS).
            5. Set "is_functional": true ONLY if the relation can logically have only ONE unique object at a time (e.g., CEO_OF, BORN_IN, DEVELOPED_IN_YEAR), else false.
            6. Provide the EXACT verbatim supporting sentence from the chunk in "evidence_quote".

            Return ONLY valid JSON matching this schema:
            {{
                "claims": [
                    {{
                        "subject": "Artificial Intelligence",
                        "subject_alias": "AI",
                        "subject_type": "CONCEPT",
                        "relation": "USES",
                        "is_functional": false,
                        "object": "Knowledge Graph",
                        "object_alias": "KG",
                        "object_type": "DATA_STRUCTURE",
                        "evidence_quote": "Exact verbatim sentence from the chunk.",
                        "extraction_confidence": 0.95
                    }}
                ]
            }}

            CHUNK TEXT:
            \"\"\"{chunk['text']}\"\"\"
            """

            response = self.openai_client.chat.completions.create(
                model=self.config.llm_model_name,
                messages=[
                    {"role": "system", "content": "You output strictly raw JSON without markdown formatting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )

            raw_output = response.choices[0].message.content.strip()
            if raw_output.startswith("```json"):
                raw_output = raw_output[7:]
            if raw_output.endswith("```"):
                raw_output = raw_output[:-3]

            parsed = json.loads(raw_output.strip())
            claims = parsed.get("claims", [])

            # Save to local cache immediately
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(claims, f, indent=2)

            enriched_claims = []
            for c in claims:
                c["document_id"] = chunk["document_id"]
                c["chunk_id"] = chunk["chunk_id"]
                c["source_name"] = chunk["source_name"]
                c["extracted_at"] = datetime.utcnow().isoformat()
                enriched_claims.append(c)

            return enriched_claims

        except Exception as e:
            logging.warning(f"Extraction error on chunk {chunk['chunk_id']}: {e}")
            return []

    # ------------------------------------------------------------------
    # 3. CROSS-PDF ENTITY RESOLUTION (ACRONYMS + EMBEDDINGS)
    # ------------------------------------------------------------------
    def validate_and_resolve_entities(
        self, raw_claims: List[Dict[str, Any]], chunks_by_id: Dict[str, str]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[float]]:
        try:
            grounding_scores: List[float] = []
            verified_claims: List[Dict[str, Any]] = []

            for claim in raw_claims:
                subj = str(claim.get("subject", "")).strip()
                obj = str(claim.get("object", "")).strip()
                rel = str(claim.get("relation", "RELATED_TO")).strip().upper().replace(" ", "_")
                quote = str(claim.get("evidence_quote", "")).strip()
                chunk_text = chunks_by_id.get(claim["chunk_id"], "")

                if not subj or not obj or not chunk_text:
                    grounding_scores.append(0.0)
                    continue

                quote_tokens = set(quote.lower().split())
                chunk_tokens = set(chunk_text.lower().split())
                if quote and quote.lower() in chunk_text.lower():
                    quote_match = 1.0
                elif quote_tokens:
                    overlap = len(quote_tokens.intersection(chunk_tokens)) / len(quote_tokens)
                    quote_match = 1.0 if overlap >= 0.75 else (overlap * 0.5)
                else:
                    quote_match = 0.0

                triplet_vec = self._encode_text(f"{subj} {rel} {obj}")
                quote_vec = self._encode_text(quote if quote else chunk_text)
                sem_sim = max(0.0, self._cosine_sim(triplet_vec, quote_vec))

                claim_grounding = float(quote_match * sem_sim)
                grounding_scores.append(claim_grounding)

                if quote_match >= 0.65:
                    claim["relation"] = rel
                    claim["grounding_score"] = claim_grounding
                    verified_claims.append(claim)

            # Cross-PDF Entity Resolution Registry
            canonical_entities: Dict[str, Dict[str, Any]] = {}
            alias_lookup: Dict[str, str] = {}

            for claim in verified_claims:
                doc_source = claim["source_name"]
                for role, alias_key, type_key in [
                    ("subject", "subject_alias", "subject_type"),
                    ("object", "object_alias", "object_type")
                ]:
                    mention = str(claim.get(role, "")).strip()
                    alias = str(claim.get(alias_key, "")).strip()
                    ent_type = str(claim.get(type_key, "CONCEPT")).strip().upper().replace(" ", "_")

                    low_mention = mention.lower()
                    low_alias = alias.lower() if alias else ""

                    # 1. Exact or registered alias match
                    if low_mention in alias_lookup:
                        canon_name = alias_lookup[low_mention]
                        canonical_entities[canon_name]["source_pdfs"].add(doc_source)
                        claim[role] = canon_name
                        continue
                    if low_alias and low_alias in alias_lookup:
                        canon_name = alias_lookup[low_alias]
                        alias_lookup[low_mention] = canon_name
                        canonical_entities[canon_name]["source_pdfs"].add(doc_source)
                        claim[role] = canon_name
                        continue

                    # 2. Acronym Initials Match OR High Embedding Similarity Match across PDFs
                    m_vec = self._encode_text(mention)
                    matched_canon = None
                    for c_name, c_data in canonical_entities.items():
                        if (
                            self._is_acronym_match(mention, c_name)
                            or (alias and self._is_acronym_match(alias, c_name))
                            or any(self._is_acronym_match(mention, al) for al in c_data["aliases"])
                            or self._cosine_sim(m_vec, c_data["embedding"]) >= self.config.entity_sim_threshold
                        ):
                            matched_canon = c_name
                            break

                    if matched_canon:
                        # Prefer the longer descriptive name as the canonical label (e.g., "Artificial Intelligence" over "AI")
                        alias_lookup[low_mention] = matched_canon
                        if low_alias:
                            alias_lookup[low_alias] = matched_canon
                        canonical_entities[matched_canon]["aliases"].add(mention)
                        canonical_entities[matched_canon]["source_pdfs"].add(doc_source)
                        claim[role] = matched_canon
                    else:
                        canonical_entities[mention] = {
                            "name": mention,
                            "type": ent_type,
                            "embedding": m_vec,
                            "aliases": {mention, alias} if alias else {mention},
                            "source_pdfs": {doc_source}
                        }
                        alias_lookup[low_mention] = mention
                        if low_alias:
                            alias_lookup[low_alias] = mention
                        claim[role] = mention

            return verified_claims, canonical_entities, grounding_scores

        except Exception as e:
            raise KGAuditException(e, sys)

    # ------------------------------------------------------------------
    # 4. PHASE 1 MATHEMATICAL GRAPH FIDELITY ENGINE (S_KG)
    # ------------------------------------------------------------------
    def compute_kg_fidelity_metrics(
        self,
        chunks: List[Dict[str, Any]],
        verified_claims: List[Dict[str, Any]],
        grounding_scores: List[float]
    ) -> Tuple[float, float, float, float, float]:
        try:
            phi_prov = float(np.mean(grounding_scores)) if grounding_scores else 0.0

            claims_by_chunk: Dict[str, List[str]] = {}
            for c in verified_claims:
                cid = c["chunk_id"]
                claims_by_chunk.setdefault(cid, []).append(
                    f"{c['subject']} {c['relation']} {c['object']}"
                )

            weighted_cov_sum = 0.0
            total_weight = 0.0
            for ch in chunks:
                cid = ch["chunk_id"]
                w_j = float(np.log1p(len(ch["text"])))
                total_weight += w_j
                extracted_strs = claims_by_chunk.get(cid, [])
                if not extracted_strs:
                    continue
                chunk_vec = self._encode_text(ch["text"])
                triplets_vec = self._encode_text(" . ".join(extracted_strs))
                cov_sim = max(0.0, self._cosine_sim(chunk_vec, triplets_vec))
                weighted_cov_sum += w_j * cov_sim

            phi_cov = float(weighted_cov_sum / total_weight) if total_weight > 0 else 0.0

            G = nx.Graph()
            self_loops = 0
            for c in verified_claims:
                u, v = c["subject"], c["object"]
                G.add_node(u)
                G.add_node(v)
                if u == v:
                    self_loops += 1
                else:
                    G.add_edge(u, v)

            if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
                largest_cc = max(nx.connected_components(G), key=len)
                lcc_ratio = len(largest_cc) / G.number_of_nodes()
                loop_penalty = self_loops / max(1, len(verified_claims))
                phi_topo = float((0.5 + 0.5 * lcc_ratio) * (1.0 - loop_penalty))
            else:
                phi_topo = 0.0

            # Dynamic Functional Consistency check using LLM-inferred "is_functional" flag
            functional_map: Dict[Tuple[str, str], set] = {}
            conflict_count = 0
            for c in verified_claims:
                if bool(c.get("is_functional", False)):
                    key = (c["subject"], c["relation"])
                    functional_map.setdefault(key, set()).add(c["object"])
                    if len(functional_map[key]) > 1:
                        conflict_count += 1

            phi_cons = float(1.0 - (conflict_count / max(1, len(verified_claims))))

            harmonic_pr = (2.0 * phi_prov * phi_cov) / (phi_prov + phi_cov + 1e-8)
            structural_reg = max(1e-8, phi_topo * phi_cons)
            s_kg = float(np.clip((harmonic_pr ** 0.7) * (structural_reg ** 0.3), 0.0, 1.0))

            return phi_prov, phi_cov, phi_topo, phi_cons, s_kg

        except Exception as e:
            raise KGAuditException(e, sys)

    # ------------------------------------------------------------------
    # 5. PUSH MULTI-PDF GRAPH TO NEO4J AURADB CLOUD
    # ------------------------------------------------------------------
    def push_to_neo4j_cloud(
        self,
        canonical_entities: Dict[str, Dict[str, Any]],
        verified_claims: List[Dict[str, Any]],
        s_kg_score: float
    ) -> bool:
        try:
            logging.info(f"Connecting to Neo4j AuraDB Cloud at {self.config.neo4j_uri}...")
            driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            driver.verify_connectivity()

            with driver.session(database=self.config.neo4j_database) as session:
                # 1. Upsert Canonical Entities with Cross-PDF Source Lists
                for ent_name, ent_data in canonical_entities.items():
                    session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type,
                            e.aliases = $aliases,
                            e.source_pdfs = $source_pdfs,
                            e.pdf_mention_count = size($source_pdfs),
                            e.updated_at = $ts
                        """,
                        name=ent_name,
                        type=ent_data["type"],
                        aliases=list(ent_data["aliases"]),
                        source_pdfs=list(ent_data["source_pdfs"]),
                        ts=datetime.utcnow().isoformat()
                    )

                # 2. Upsert Verified Relationships (Accumulates multiple PDFs supporting the same edge!)
                for claim in verified_claims:
                    session.run(
                        """
                        MATCH (u:Entity {name: $subj})
                        MATCH (v:Entity {name: $obj})
                        MERGE (u)-[r:RELATED_TO {type: $rel}]->(v)
                        SET r.weight = $conf,
                            r.evidence_quote = $quote,
                            r.source_pdfs = CASE
                                WHEN r.source_pdfs IS NULL THEN [$source_name]
                                WHEN NOT $source_name IN r.source_pdfs THEN r.source_pdfs + $source_name
                                ELSE r.source_pdfs
                            END,
                            r.document_ids = CASE
                                WHEN r.document_ids IS NULL THEN [$doc_id]
                                WHEN NOT $doc_id IN r.document_ids THEN r.document_ids + $doc_id
                                ELSE r.document_ids
                            END,
                            r.grounding_score = $grounding,
                            r.kg_fidelity_score = $s_kg
                        """,
                        subj=claim["subject"],
                        obj=claim["object"],
                        rel=claim["relation"],
                        conf=float(claim.get("extraction_confidence", 0.9)),
                        quote=claim.get("evidence_quote", ""),
                        doc_id=claim["document_id"],
                        source_name=claim["source_name"],
                        grounding=float(claim.get("grounding_score", 1.0)),
                        s_kg=s_kg_score
                    )

                # 3. Store Global Phase 1 KG Fidelity Meta Node
                session.run(
                    """
                    MERGE (m:GraphMeta {id: 'GLOBAL_KG_FIDELITY'})
                    SET m.overall_kg_accuracy_score = $s_kg,
                        m.verified_triplets = $count,
                        m.updated_at = $ts
                    """,
                    s_kg=s_kg_score,
                    count=len(verified_claims),
                    ts=datetime.utcnow().isoformat()
                )

            driver.close()
            logging.info("Successfully pushed Multi-PDF Knowledge Graph to Neo4j AuraDB Cloud.")
            return True

        except Exception as e:
            logging.error(f"Neo4j Cloud push failed: {e}")
            return False

    # ------------------------------------------------------------------
    # MASTER EXECUTION ENTRYPOINTS
    # ------------------------------------------------------------------
    def initiate_kg_from_directory(self, directory_path: str = None) -> KGBuilderArtifact:
        """
        Automatically discovers all .pdf and .txt files inside data/uploads/
        and builds a unified, cross-document Knowledge Graph in Neo4j Cloud.
        """
        target_dir = directory_path or self.config.data_uploads_dir
        os.makedirs(target_dir, exist_ok=True)
        pdf_files = sorted(glob.glob(os.path.join(target_dir, "*.pdf")))
        txt_files = sorted(glob.glob(os.path.join(target_dir, "*.txt")))
        all_files = pdf_files + txt_files

        if not all_files:
            raise ValueError(f"No .pdf or .txt files found in {target_dir}. Place your PDFs there first!")

        logging.info(f"Found {len(all_files)} documents in {target_dir}: {all_files}")
        return self.initiate_kg_construction(all_files)

    def initiate_kg_construction(self, file_paths_or_texts: List[str]) -> KGBuilderArtifact:
        try:
            logging.info("=== Starting Phase 1: Multi-PDF KG Construction & Fidelity Audit ===")
            chunks = self.parse_and_chunk_documents(file_paths_or_texts)
            chunks_by_id = {ch["chunk_id"]: ch["text"] for ch in chunks}

            raw_claims: List[Dict[str, Any]] = []
            for ch in chunks:
                extracted = self.extract_from_chunk(ch)
                raw_claims.extend(extracted)

            verified_claims, canonical_entities, grounding_scores = (
                self.validate_and_resolve_entities(raw_claims, chunks_by_id)
            )

            phi_prov, phi_cov, phi_topo, phi_cons, s_kg = self.compute_kg_fidelity_metrics(
                chunks, verified_claims, grounding_scores
            )

            os.makedirs(os.path.dirname(self.config.kg_fidelity_state_path), exist_ok=True)
            # Count how many entities are shared across 2 or more PDFs!
            shared_entities = [
                name for name, data in canonical_entities.items() if len(data["source_pdfs"]) > 1
            ]

            fidelity_state = {
                "overall_kg_accuracy_score": s_kg,
                "provenance_grounding_score": phi_prov,
                "semantic_coverage_score": phi_cov,
                "topological_coherence_score": phi_topo,
                "internal_consistency_score": phi_cons,
                "total_chunks_processed": len(chunks),
                "raw_triplets_extracted": len(raw_claims),
                "verified_triplets_pushed": len(verified_claims),
                "resolved_entity_count": len(canonical_entities),
                "cross_pdf_shared_entities_count": len(shared_entities),
                "cross_pdf_shared_entities": shared_entities,
                "timestamp": datetime.utcnow().isoformat()
            }
            with open(self.config.kg_fidelity_state_path, "w") as f:
                json.dump(fidelity_state, f, indent=4)

            os.makedirs(os.path.dirname(self.config.extracted_kg_file_path), exist_ok=True)
            kg_payload = {
                "fidelity_metrics": fidelity_state,
                "nodes": [
                    {
                        "id": idx,
                        "name": name,
                        "type": data["type"],
                        "aliases": list(data["aliases"]),
                        "source_pdfs": list(data["source_pdfs"])
                    }
                    for idx, (name, data) in enumerate(canonical_entities.items())
                ],
                "edges": verified_claims
            }
            with open(self.config.extracted_kg_file_path, "w") as f:
                json.dump(kg_payload, f, indent=4)

            pushed_ok = self.push_to_neo4j_cloud(canonical_entities, verified_claims, s_kg)

            logging.info(
                f"Phase 1 Complete | Overall KG Accuracy (S_KG): {s_kg:.4f} | "
                f"Cross-PDF Shared Entities: {len(shared_entities)}"
            )

            return KGBuilderArtifact(
                total_chunks_processed=len(chunks),
                raw_triplets_extracted=len(raw_claims),
                verified_triplets_pushed=len(verified_claims),
                resolved_entity_count=len(canonical_entities),
                provenance_grounding_score=phi_prov,
                semantic_coverage_score=phi_cov,
                topological_coherence_score=phi_topo,
                internal_consistency_score=phi_cons,
                overall_kg_accuracy_score=s_kg,
                kg_fidelity_state_path=self.config.kg_fidelity_state_path,
                extracted_kg_file_path=self.config.extracted_kg_file_path,
                neo4j_cloud_pushed=pushed_ok
            )

        except Exception as e:
            raise KGAuditException(e, sys)
import os
import sys
import json
import pickle
import hashlib
import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance
from typing import Dict, Any, List, Tuple, Optional

from kg_audit.entity.config_entity import TruthEvaluatorConfig
from kg_audit.entity.artifact_entity import GraphRetrieverArtifact, TruthEvaluatorArtifact
from kg_audit.exceptions.exception import KGAuditException
from kg_audit.logging.logger import logging

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class TruthEvaluator:
    def __init__(self, config: TruthEvaluatorConfig):
        try:
            self.config = config
            self.embedder = None
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self.embedder = SentenceTransformer(self.config.embedding_model_name)
                except Exception as ex:
                    logging.warning(f"Could not load SentenceTransformer, using deterministic spectral hash embeddings: {ex}")
        except Exception as e:
            raise KGAuditException(e, sys)

    def _encode_text(self, text: str, dim: int = 384) -> np.ndarray:
        """
        Generates a normalized dense vector for an entity or relation string.
        Uses SentenceTransformer if available; otherwise uses a deterministic
        orthogonal SHA-256 projection (zero random noise).
        """
        clean_text = text.strip().lower()
        if self.embedder is not None:
            vec = self.embedder.encode(clean_text, show_progress_bar=False)
            norm = np.linalg.norm(vec)
            return vec / (norm + 1e-12)

        # Deterministic algebraic projection from text n-grams (strictly reproducible)
        vec = np.zeros(dim, dtype=np.float64)
        tokens = [clean_text[i:i+3] for i in range(max(1, len(clean_text) - 2))]
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-12)

    def _build_aligned_graphs(
        self,
        gt_nodes: List[Dict[str, Any]],
        gt_edges: List[Dict[str, Any]],
        resp_nodes: List[Dict[str, Any]],
        resp_edges: List[Dict[str, Any]]
    ) -> Tuple[nx.Graph, nx.Graph, Dict[int, str]]:
        """
        Aligns nodes between Ground-Truth (G_gt) and LLM Response (G_resp)
        by canonical entity name so node IDs correspond to identical entities.
        """
        try:
            canonical_name_to_id: Dict[str, int] = {}
            canonical_id_to_name: Dict[int, str] = {}

            def register_entity(name_str: str) -> int:
                key = str(name_str).strip().lower()
                if key not in canonical_name_to_id:
                    new_id = len(canonical_name_to_id)
                    canonical_name_to_id[key] = new_id
                    canonical_id_to_name[new_id] = str(name_str).strip()
                return canonical_name_to_id[key]

            gt_raw_id_to_canon: Dict[Any, int] = {}
            for node in gt_nodes:
                c_id = register_entity(node.get("name", node.get("id")))
                gt_raw_id_to_canon[node["id"]] = c_id

            resp_raw_id_to_canon: Dict[Any, int] = {}
            for node in resp_nodes:
                c_id = register_entity(node.get("name", node.get("id")))
                resp_raw_id_to_canon[node["id"]] = c_id

            G_gt = nx.Graph()
            G_resp = nx.Graph()

            for c_id, c_name in canonical_id_to_name.items():
                G_gt.add_node(c_id, name=c_name)
                G_resp.add_node(c_id, name=c_name)

            for edge in gt_edges:
                u = gt_raw_id_to_canon.get(edge["source"])
                v = gt_raw_id_to_canon.get(edge["target"])
                if u is not None and v is not None:
                    G_gt.add_edge(
                        u, v,
                        weight=float(edge.get("weight", 1.0)),
                        type=str(edge.get("type", "RELATED_TO")).strip().upper()
                    )

            for edge in resp_edges:
                u = resp_raw_id_to_canon.get(edge["source"])
                v = resp_raw_id_to_canon.get(edge["target"])
                if u is not None and v is not None:
                    G_resp.add_edge(
                        u, v,
                        weight=float(edge.get("weight", 1.0)),
                        type=str(edge.get("type", "RELATED_TO")).strip().upper()
                    )

            return G_gt, G_resp, canonical_id_to_name

        except Exception as e:
            raise KGAuditException(e, sys)

    def compute_sheaf_laplacian_energy(
        self, G_gt: nx.Graph, G_resp: nx.Graph, id_to_name: Dict[int, str]
    ) -> float:
        """
        Computes authentic Cellular Sheaf Coboundary Residual E_sheaf.
        Assigns semantic embeddings x_u, x_v to stalks F(u), F(v) and evaluates
        relational restriction maps F_{u->e}(x_u) - F_{v->e}(x_v) against G_gt.
        """
        try:
            if G_resp.number_of_edges() == 0:
                return 1.0

            node_stalks: Dict[int, np.ndarray] = {
                n: self._encode_text(id_to_name[n]) for n in G_resp.nodes()
            }

            total_coboundary_energy = 0.0
            total_stalk_norm = sum(float(np.sum(vec ** 2)) for vec in node_stalks.values())

            for u, v, resp_data in G_resp.edges(data=True):
                weight = float(resp_data.get("weight", 1.0))
                rel_resp = resp_data.get("type", "RELATED_TO")
                rel_resp_vec = self._encode_text(rel_resp)

                if G_gt.has_edge(u, v):
                    rel_gt = G_gt[u][v].get("type", "RELATED_TO")
                    rel_gt_vec = self._encode_text(rel_gt)
                    # Coboundary residual measures semantic divergence between claimed relation and GT relation
                    residual = rel_resp_vec - rel_gt_vec
                    edge_energy = float(np.sum(residual ** 2))
                else:
                    # Edge is missing in G_gt: compute shortest-path compositional residual in G_gt
                    if nx.has_path(G_gt, u, v):
                        path = nx.shortest_path(G_gt, source=u, target=v)
                        composed_rel_vec = np.zeros_like(rel_resp_vec)
                        for i in range(len(path) - 1):
                            edge_type = G_gt[path[i]][path[i + 1]].get("type", "RELATED_TO")
                            composed_rel_vec += self._encode_text(edge_type)
                        hop_penalty = float(len(path) - 1)
                        residual = (node_stalks[u] + rel_resp_vec - node_stalks[v]) - composed_rel_vec
                        edge_energy = hop_penalty * float(np.sum(residual ** 2))
                    else:
                        # Completely disconnected components: maximum stalk inconsistency
                        residual = node_stalks[u] + rel_resp_vec - node_stalks[v]
                        edge_energy = 2.0 * float(np.sum(residual ** 2))

                total_coboundary_energy += weight * edge_energy

            if total_stalk_norm == 0.0:
                return 1.0

            return float(total_coboundary_energy / total_stalk_norm)

        except Exception as e:
            raise KGAuditException(e, sys)

    def compute_forman_ricci_curvature(self, G: nx.Graph) -> Dict[Tuple[int, int], float]:
        try:
            curvatures = {}
            node_weights = {
                n: float(G.degree(n, weight="weight")) if G.degree(n, weight="weight") > 0 else 1.0
                for n in G.nodes()
            }

            for u, v, data in G.edges(data=True):
                w_e = float(data.get("weight", 1.0))
                w_u, w_v = node_weights[u], node_weights[v]

                sum_u = sum(
                    w_u / np.sqrt(w_e * float(e_u[2].get("weight", 1.0)))
                    for e_u in G.edges(u, data=True)
                    if (e_u[1] if e_u[0] == u else e_u[0]) != v
                )
                sum_v = sum(
                    w_v / np.sqrt(w_e * float(e_v[2].get("weight", 1.0)))
                    for e_v in G.edges(v, data=True)
                    if (e_v[1] if e_v[0] == v else e_v[0]) != u
                )

                ric = w_e * ((w_u / w_e) + (w_v / w_e) - sum_u - sum_v)
                curvatures[(u, v)] = float(ric)
                curvatures[(v, u)] = float(ric)

            return curvatures
        except Exception as e:
            raise KGAuditException(e, sys)

    def compute_curvature_distortion(self, G_gt: nx.Graph, G_resp: nx.Graph, gamma: float = 1.0) -> float:
        try:
            ric_gt = self.compute_forman_ricci_curvature(G_gt)
            ric_resp = self.compute_forman_ricci_curvature(G_resp)

            intersect_diff = 0.0
            intersect_count = 0
            hallucinated_penalty = 0.0

            for (u, v), ric_r in ric_resp.items():
                if u < v:
                    if G_gt.has_edge(u, v):
                        ric_g = ric_gt.get((u, v), 0.0)
                        intersect_diff += abs(ric_g - ric_r)
                        intersect_count += 1
                    else:
                        hallucinated_penalty += np.exp(-ric_r)

            avg_distortion = (intersect_diff / intersect_count) if intersect_count > 0 else 1.0
            return float(avg_distortion + (gamma * hallucinated_penalty))
        except Exception as e:
            raise KGAuditException(e, sys)

    def compute_spectral_laplacian_divergence(self, G_gt: nx.Graph, G_resp: nx.Graph, alpha: float = 0.5) -> float:
        try:
            if G_gt.number_of_nodes() == 0 or G_resp.number_of_nodes() == 0:
                return 1.0

            L_gt = nx.normalized_laplacian_matrix(G_gt).toarray()
            L_resp = nx.normalized_laplacian_matrix(G_resp).toarray()

            evals_gt = np.sort(np.linalg.eigvalsh(L_gt))
            evals_resp = np.sort(np.linalg.eigvalsh(L_resp))

            w1_dist = float(wasserstein_distance(evals_gt, evals_resp))
            frobenius_diff = float(np.linalg.norm(L_gt - L_resp, ord="fro") ** 2)

            return float(frobenius_diff + (alpha * w1_dist))
        except Exception as e:
            raise KGAuditException(e, sys)

    def compute_provenance_and_conflict_features(
        self,
        G_gt: nx.Graph,
        G_resp: nx.Graph,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, float, float, float, float]:
        """
        Computes the 5 statistical provenance features from the research proposal:
        1. Bayesian Source Trust: E[Beta(alpha, beta)] = (1 + pos) / (2 + pos + neg)
        2. Temporal Freshness: exp(-lambda * age_in_years)
        3. Source Independence Score: ratio of independent clusters to total sources
        4. Active Conflict Ratio: fraction of response edges violating GT functional relations
        5. Raw LLM Extraction Confidence
        """
        if metadata is None:
            metadata = {}

        # 1. Bayesian Source Trust (Beta(1,1) uniform prior)
        pos_claims = float(metadata.get("source_historical_correct", 15))
        neg_claims = float(metadata.get("source_historical_incorrect", 2))
        bayesian_trust = (1.0 + pos_claims) / (2.0 + pos_claims + neg_claims)

        # 2. Temporal Freshness Decay: F(t) = exp(-lambda * delta_t)
        doc_age_years = float(metadata.get("document_age_years", 0.5))
        decay_lambda = float(metadata.get("relation_decay_lambda", 0.15))
        freshness = float(np.exp(-decay_lambda * doc_age_years))

        # 3. Source Independence Score
        source_independence = float(metadata.get("source_independence_score", 0.85))

        # 4. Active Conflict Ratio (checks for unverified or conflicting edges in G_resp)
        total_resp_edges = max(1, G_resp.number_of_edges())
        unverified_edges = sum(1 for u, v in G_resp.edges() if not G_gt.has_edge(u, v))
        conflict_ratio = float(unverified_edges / total_resp_edges)

        # 5. LLM Extraction Confidence
        llm_confidence = float(metadata.get("llm_extraction_confidence", 0.90))

        return bayesian_trust, freshness, source_independence, conflict_ratio, llm_confidence

    def evaluate_truth(
        self,
        retriever_artifact: GraphRetrieverArtifact,
        response_nodes: List[Dict[str, Any]],
        response_edges: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> TruthEvaluatorArtifact:
        try:
            logging.info("Initiating aligned CASSE + Provenance mathematical evaluation...")

            G_gt, G_resp, id_to_name = self._build_aligned_graphs(
                retriever_artifact.extracted_nodes,
                retriever_artifact.extracted_edges,
                response_nodes,
                response_edges
            )

            e_sheaf = self.compute_sheaf_laplacian_energy(G_gt, G_resp, id_to_name)
            delta_k = self.compute_curvature_distortion(G_gt, G_resp)
            d_spec = self.compute_spectral_laplacian_divergence(G_gt, G_resp)

            bayesian_trust, freshness, independence, conflict_ratio, llm_conf = (
                self.compute_provenance_and_conflict_features(G_gt, G_resp, metadata)
            )

            # 8-Dimensional Feature Vector X for ModelTrainer & Calibrated Inference
            feature_vector = [
                e_sheaf,
                delta_k,
                d_spec,
                bayesian_trust,
                freshness,
                independence,
                conflict_ratio,
                llm_conf
            ]

            # If a trained & calibrated ML model exists in final_model/model.pkl, use it!
            if os.path.exists(self.config.trained_model_path):
                logging.info(f"Loading calibrated belief model from {self.config.trained_model_path}")
                with open(self.config.trained_model_path, "rb") as f:
                    calibrated_model = pickle.load(f)
                X_arr = np.array([feature_vector], dtype=np.float64)
                final_accuracy_score = float(calibrated_model.predict_proba(X_arr)[0, 1])
            else:
                # Analytical CASSE exponential baseline before model training
                exponent = (
                    (self.config.entailment_weight * e_sheaf)
                    + (self.config.triple_match_weight * delta_k)
                    + (self.config.topological_weight * d_spec)
                    + (0.5 * conflict_ratio)
                )
                trust_multiplier = 0.5 * (bayesian_trust + freshness * independence)
                final_accuracy_score = float(np.clip(trust_multiplier * np.exp(-exponent), 0.0, 1.0))

            # Three-way decision rule: VERIFIED vs. ABSTAIN vs. HALLUCINATED
            if final_accuracy_score >= self.config.accuracy_pass_threshold:
                decision_status = "VERIFIED"
                is_hallucinated = False
            elif final_accuracy_score >= self.config.abstention_threshold:
                decision_status = "ABSTAIN"
                is_hallucinated = False
            else:
                decision_status = "HALLUCINATED"
                is_hallucinated = True

            report = {
                "metrics": {
                    "sheaf_laplacian_energy": e_sheaf,
                    "forman_ricci_curvature_distortion": delta_k,
                    "spectral_laplacian_divergence": d_spec,
                    "bayesian_source_trust": bayesian_trust,
                    "temporal_freshness": freshness,
                    "source_independence": independence,
                    "conflict_ratio": conflict_ratio,
                    "llm_extraction_confidence": llm_conf,
                    "final_accuracy_score": final_accuracy_score
                },
                "feature_vector_X": feature_vector,
                "verdict": {
                    "decision_status": decision_status,
                    "is_hallucinated": is_hallucinated,
                    "pass_threshold": self.config.accuracy_pass_threshold,
                    "abstention_threshold": self.config.abstention_threshold
                }
            }

            export_path = self.config.evaluation_report_file_path
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, "w") as f:
                json.dump(report, f, indent=4)

            logging.info(
                f"Evaluation complete. Score: {final_accuracy_score:.4f} | Status: {decision_status}"
            )

            return TruthEvaluatorArtifact(
                semantic_entailment_score=e_sheaf,
                triple_match_score=delta_k,
                topological_similarity_score=d_spec,
                bayesian_source_trust=bayesian_trust,
                temporal_freshness=freshness,
                source_independence=independence,
                conflict_ratio=conflict_ratio,
                feature_vector=feature_vector,
                final_accuracy_score=final_accuracy_score,
                decision_status=decision_status,
                is_hallucinated=is_hallucinated,
                evaluation_report_path=export_path
            )

        except Exception as e:
            raise KGAuditException(e, sys)
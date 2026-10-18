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

_EVAL_EMBEDDER = None


class TruthEvaluator:
    def __init__(self, config: TruthEvaluatorConfig):
        try:
            global _EVAL_EMBEDDER
            self.config = config
            self.embedder = None
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                if _EVAL_EMBEDDER is None:
                    _EVAL_EMBEDDER = SentenceTransformer(self.config.embedding_model_name)
                self.embedder = _EVAL_EMBEDDER
        except Exception as e:
            raise KGAuditException(e, sys)

    def _encode_text(self, text: str, dim: int = 384) -> np.ndarray:
        clean_text = str(text).strip().lower()
        if not clean_text:
            return np.zeros(dim, dtype=np.float64)
        if self.embedder is not None:
            vec = self.embedder.encode(clean_text, show_progress_bar=False)
            return vec / (np.linalg.norm(vec) + 1e-12)

        vec = np.zeros(dim, dtype=np.float64)
        tokens = [clean_text[i:i+3] for i in range(max(1, len(clean_text) - 2))]
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[idx] += sign
        return vec / (np.linalg.norm(vec) + 1e-12)

    @staticmethod
    def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1, v2) / ((np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-12))

    @staticmethod
    def _is_acronym_or_substring(s1: str, s2: str) -> bool:
        a = s1.strip().lower()
        b = s2.strip().lower()
        if not a or not b:
            return False
        if a == b or (len(a) >= 4 and a in b) or (len(b) >= 4 and b in a):
            return True
        # Check acronym initials (e.g., "line" <-> "large-scale information network embedding", "bfs" <-> "breadth-first search")
        short, long_s = (a, b) if len(a) < len(b) else (b, a)
        short_clean = short.replace(".", "").replace("-", "")
        stop = {"of", "the", "and", "in", "for", "to", "a", "an", "strategy"}
        words = [w for w in long_s.replace("-", " ").split() if w and w not in stop]
        if len(words) >= 2:
            initials = "".join(w[0] for w in words)
            if short_clean == initials or initials.startswith(short_clean):
                return True
        return False

    def _build_aligned_graphs(
        self,
        gt_nodes: List[Dict[str, Any]],
        gt_edges: List[Dict[str, Any]],
        resp_nodes: List[Dict[str, Any]],
        resp_edges: List[Dict[str, Any]]
    ) -> Tuple[nx.Graph, nx.Graph, Dict[int, str]]:
        """
        Aligns nodes between Ground-Truth (G_gt) and LLM Response (G_resp)
        using exact match, alias match, acronym expansion, and semantic similarity >= 0.68.
        """
        try:
            canonical_id_to_name: Dict[int, str] = {}
            canonical_id_to_vec: Dict[int, np.ndarray] = {}
            canonical_id_to_aliases: Dict[int, List[str]] = {}

            gt_raw_id_to_canon: Dict[Any, int] = {}
            for node in gt_nodes:
                name_str = str(node.get("name", node.get("id"))).strip()
                aliases = [str(a) for a in node.get("aliases", [])] + [name_str]
                c_id = len(canonical_id_to_name)
                canonical_id_to_name[c_id] = name_str
                canonical_id_to_vec[c_id] = self._encode_text(name_str)
                canonical_id_to_aliases[c_id] = aliases
                gt_raw_id_to_canon[node["id"]] = c_id

            resp_raw_id_to_canon: Dict[Any, int] = {}
            for node in resp_nodes:
                r_name = str(node.get("name", node.get("id"))).strip()
                r_vec = self._encode_text(r_name)
                matched_cid = None
                best_sim = 0.0

                for cid, gt_name in canonical_id_to_name.items():
                    if any(self._is_acronym_or_substring(r_name, al) for al in canonical_id_to_aliases.get(cid, [gt_name])):
                        matched_cid = cid
                        break
                    sim = self._cosine_sim(r_vec, canonical_id_to_vec[cid])
                    if sim >= 0.68 and sim > best_sim:
                        best_sim = sim
                        matched_cid = cid

                if matched_cid is None:
                    matched_cid = len(canonical_id_to_name)
                    canonical_id_to_name[matched_cid] = r_name
                    canonical_id_to_vec[matched_cid] = r_vec
                    canonical_id_to_aliases[matched_cid] = [r_name]

                resp_raw_id_to_canon[node["id"]] = matched_cid

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
        try:
            if G_resp.number_of_edges() == 0:
                return 0.25

            node_stalks: Dict[int, np.ndarray] = {
                n: self._encode_text(id_to_name[n]) for n in G_resp.nodes()
            }

            gt_active_nodes = {n for n, d in G_gt.degree() if d > 0}
            gt_concept_vecs = [node_stalks[n] for n in gt_active_nodes] if gt_active_nodes else []

            edge_energies = []
            for u, v, resp_data in G_resp.edges(data=True):
                rel_resp = resp_data.get("type", "RELATED_TO")
                rel_resp_vec = self._encode_text(rel_resp)

                if G_gt.has_edge(u, v):
                    rel_gt = G_gt[u][v].get("type", "RELATED_TO")
                    rel_gt_vec = self._encode_text(rel_gt)
                    sim = max(0.0, self._cosine_sim(rel_resp_vec, rel_gt_vec))
                    edge_energies.append(0.15 * (1.0 - sim))
                elif nx.has_path(G_gt, u, v):
                    path = nx.shortest_path(G_gt, source=u, target=v)
                    hops = len(path) - 1
                    edge_energies.append(min(0.35, 0.08 * hops))
                elif gt_concept_vecs:
                    # Semantic stalk alignment against the active Neo4j subgraph concepts
                    u_max_sim = max(self._cosine_sim(node_stalks[u], gv) for gv in gt_concept_vecs)
                    v_max_sim = max(self._cosine_sim(node_stalks[v], gv) for gv in gt_concept_vecs)
                    mean_grounding = 0.5 * (max(0.0, u_max_sim) + max(0.0, v_max_sim))
                    # Low energy if concepts are semantically grounded in the KG; high energy if alien/hallucinated
                    if mean_grounding >= 0.45:
                        edge_energies.append(0.30 * (1.0 - mean_grounding))
                    else:
                        edge_energies.append(1.40 * (1.0 - mean_grounding))
                else:
                    edge_energies.append(1.50)

            return float(np.mean(edge_energies))

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

    def compute_curvature_distortion(
        self, G_gt: nx.Graph, G_resp: nx.Graph, id_to_name: Dict[int, str]
    ) -> float:
        """
        Bounded Forman-Ricci Curvature Distortion over the active query subgraph.
        """
        try:
            active_resp_nodes = [n for n, d in G_resp.degree() if d > 0]
            if not active_resp_nodes:
                return 0.15

            G_gt_sub = G_gt.subgraph(active_resp_nodes).copy()
            G_resp_sub = G_resp.subgraph(active_resp_nodes).copy()

            ric_gt = self.compute_forman_ricci_curvature(G_gt_sub)
            ric_resp = self.compute_forman_ricci_curvature(G_resp_sub)
            gt_active_nodes = {n for n, d in G_gt.degree() if d > 0}
            gt_vecs = [self._encode_text(id_to_name[n]) for n in gt_active_nodes] if gt_active_nodes else []

            distortions = []
            for (u, v), ric_r in ric_resp.items():
                if u < v:
                    if G_gt.has_edge(u, v):
                        ric_g = ric_gt.get((u, v), 0.0)
                        distortions.append(float(np.tanh(abs(ric_g - ric_r) / 8.0) * 0.15))
                    elif nx.has_path(G_gt, u, v):
                        distortions.append(0.12)
                    elif u in gt_active_nodes and v in gt_active_nodes:
                        distortions.append(0.18)
                    elif gt_vecs:
                        u_sim = max(self._cosine_sim(self._encode_text(id_to_name[u]), gv) for gv in gt_vecs)
                        v_sim = max(self._cosine_sim(self._encode_text(id_to_name[v]), gv) for gv in gt_vecs)
                        if min(u_sim, v_sim) >= 0.45:
                            distortions.append(0.22)
                        else:
                            bounded_pen = 1.0 / (1.0 + float(np.exp(np.clip(ric_r, -10.0, 10.0))))
                            distortions.append(0.75 + 0.45 * bounded_pen)
                    else:
                        distortions.append(0.90)

            return float(np.mean(distortions)) if distortions else 0.15
        except Exception as e:
            raise KGAuditException(e, sys)

    def compute_spectral_laplacian_divergence(self, G_gt: nx.Graph, G_resp: nx.Graph) -> float:
        """
        Computes Normalized Laplacian eigenvalue divergence over the active response entity set
        so unmentioned background nodes in Neo4j do not distort the spectrum.
        """
        try:
            active_nodes = [n for n, d in G_resp.degree() if d > 0]
            if len(active_nodes) < 2 or G_gt.number_of_edges() == 0:
                return 0.20

            G_resp_sub = G_resp.subgraph(active_nodes).copy()
            # Build induced ground-truth connectivity among the active response entities (1-hop or 2-hop reachability)
            G_gt_induced = nx.Graph()
            for n in active_nodes:
                G_gt_induced.add_node(n)
            for i in range(len(active_nodes)):
                for j in range(i + 1, len(active_nodes)):
                    u, v = active_nodes[i], active_nodes[j]
                    if G_gt.has_edge(u, v):
                        G_gt_induced.add_edge(u, v, weight=1.0)
                    elif nx.has_path(G_gt, u, v):
                        plen = len(nx.shortest_path(G_gt, u, v)) - 1
                        if plen <= 3:
                            G_gt_induced.add_edge(u, v, weight=1.0 / plen)

            L_gt = nx.normalized_laplacian_matrix(G_gt_induced).toarray()
            L_resp = nx.normalized_laplacian_matrix(G_resp_sub).toarray()

            evals_gt = np.sort(np.linalg.eigvalsh(L_gt))
            evals_resp = np.sort(np.linalg.eigvalsh(L_resp))

            w1_dist = float(wasserstein_distance(evals_gt, evals_resp))
            norm_frob = float(np.linalg.norm(L_gt - L_resp, ord="fro") / max(1.0, len(active_nodes)))

            return float(0.4 * w1_dist + 0.25 * norm_frob)
        except Exception as e:
            raise KGAuditException(e, sys)

    def compute_provenance_and_conflict_features(
        self,
        G_gt: nx.Graph,
        G_resp: nx.Graph,
        id_to_name: Dict[int, str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, float, float, float, float]:
        if metadata is None:
            metadata = {}

        pos_claims = float(metadata.get("source_historical_correct", 18))
        neg_claims = float(metadata.get("source_historical_incorrect", 2))
        bayesian_trust = (1.0 + pos_claims) / (2.0 + pos_claims + neg_claims)

        doc_age_years = float(metadata.get("document_age_years", 0.5))
        decay_lambda = float(metadata.get("relation_decay_lambda", 0.12))
        freshness = float(np.exp(-decay_lambda * doc_age_years))

        source_independence = float(metadata.get("source_independence_score", 0.88))

        # Semantic + Topological Grounding check for Conflict Ratio
        gt_active_nodes = {n for n, d in G_gt.degree() if d > 0}
        gt_vecs = [self._encode_text(id_to_name[n]) for n in gt_active_nodes] if gt_active_nodes else []

        total_resp_edges = max(1, G_resp.number_of_edges())
        alien_edges = 0
        for u, v in G_resp.edges():
            if G_gt.has_edge(u, v) or nx.has_path(G_gt, u, v):
                continue
            if u in gt_active_nodes or v in gt_active_nodes:
                continue
            if gt_vecs:
                u_sim = max(self._cosine_sim(self._encode_text(id_to_name[u]), gv) for gv in gt_vecs)
                v_sim = max(self._cosine_sim(self._encode_text(id_to_name[v]), gv) for gv in gt_vecs)
                if max(u_sim, v_sim) >= 0.48:
                    continue
            alien_edges += 1

        conflict_ratio = float(alien_edges / total_resp_edges)
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
            delta_k = self.compute_curvature_distortion(G_gt, G_resp, id_to_name)
            d_spec = self.compute_spectral_laplacian_divergence(G_gt, G_resp)

            bayesian_trust, freshness, independence, conflict_ratio, llm_conf = (
                self.compute_provenance_and_conflict_features(G_gt, G_resp, id_to_name, metadata)
            )

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

            # Step A: Compute Query-Level Belief Score (S_Query)
            if os.path.exists(self.config.trained_model_path):
                with open(self.config.trained_model_path, "rb") as f:
                    calibrated_model = pickle.load(f)
                X_arr = np.array([feature_vector], dtype=np.float64)
                query_belief_score = float(calibrated_model.predict_proba(X_arr)[0, 1])
            else:
                exponent = (
                    (self.config.entailment_weight * e_sheaf)
                    + (self.config.triple_match_weight * delta_k)
                    + (self.config.topological_weight * d_spec)
                    + (0.5 * conflict_ratio)
                )
                trust_multiplier = 0.5 * (bayesian_trust + freshness * independence)
                query_belief_score = float(np.clip(trust_multiplier * np.exp(-exponent), 0.0, 1.0))

            # Step B: Load Phase 1 Knowledge Graph Fidelity Score (S_KG)
            if os.path.exists(self.config.kg_fidelity_state_path):
                with open(self.config.kg_fidelity_state_path, "r", encoding="utf-8") as f:
                    kg_state = json.load(f)
                s_kg = float(kg_state.get("overall_kg_accuracy_score", 1.0))
            else:
                s_kg = 1.0

            # Step C: Confidence-Weighted Geometric Fusion: S_Final = (S_Query ^ 0.80) * (S_KG ^ 0.20)
            final_accuracy_score = float(np.clip((query_belief_score ** 0.80) * (max(0.1, s_kg) ** 0.20), 0.0, 1.0))

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
                    "query_belief_score_S_Query": query_belief_score,
                    "phase1_kg_fidelity_score_S_KG": s_kg,
                    "final_compound_accuracy_score": final_accuracy_score
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
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4)

            return TruthEvaluatorArtifact(
                semantic_entailment_score=e_sheaf,
                triple_match_score=delta_k,
                topological_similarity_score=d_spec,
                bayesian_source_trust=bayesian_trust,
                temporal_freshness=freshness,
                source_independence=independence,
                conflict_ratio=conflict_ratio,
                llm_self_reported_accuracy=llm_conf,
                query_belief_score=query_belief_score,
                phase1_kg_fidelity_score=s_kg,
                feature_vector=feature_vector,
                final_accuracy_score=final_accuracy_score,
                decision_status=decision_status,
                is_hallucinated=is_hallucinated,
                evaluation_report_path=export_path
            )

        except Exception as e:
            raise KGAuditException(e, sys)
# Phase 4: Calculates the accuracy score
import os
import sys
import json
import networkx as nx
import numpy as np
from scipy import sparse
from scipy.stats import wasserstein_distance
from typing import Dict, Any, List, Tuple

from kg_audit.entity.config_entity import TruthEvaluatorConfig
from kg_audit.entity.artifact_entity import GraphRetrieverArtifact, TruthEvaluatorArtifact
from kg_audit.exceptions.exception import KGAuditException
from kg_audit.logging.logger import logging

class TruthEvaluator:
    def __init__(self, config: TruthEvaluatorConfig):
        try:
            self.config = config
        except Exception as e:
            raise KGAuditException(e, sys)

    def _build_graph_from_payload(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> nx.Graph:
        try:
            G = nx.Graph()
            node_id_map = {}
            for idx, node in enumerate(nodes):
                G.add_node(idx, original_id=node["id"], name=node.get("name", str(node["id"])))
                node_id_map[node["id"]] = idx

            for edge in edges:
                u = node_id_map.get(edge["source"])
                v = node_id_map.get(edge["target"])
                if u is not None and v is not None:
                    weight = float(edge.get("weight", 1.0))
                    G.add_edge(u, v, weight=weight, type=edge.get("type", "RELATED_TO"))

            return G
        except Exception as e:
            raise KGAuditException(e, sys)

    def compute_sheaf_laplacian_energy(self, G_gt: nx.Graph, G_resp: nx.Graph, embedding_dim: int = 8) -> float:
        try:
            if G_resp.number_of_nodes() == 0 or G_resp.number_of_edges() == 0:
                return 1.0

            np.random.seed(42)
            node_cochains = {}
            for node in G_resp.nodes():
                node_cochains[node] = np.random.randn(embedding_dim)

            total_coboundary_energy = 0.0
            total_node_norm = 0.0

            for node, vec in node_cochains.items():
                total_node_norm += float(np.sum(vec ** 2))

            for u, v, data in G_resp.edges(data=True):
                weight = float(data.get("weight", 1.0))
                if G_gt.has_edge(u, v):
                    F_u_e = np.eye(embedding_dim)
                    F_v_e = np.eye(embedding_dim)
                else:
                    F_u_e = np.random.randn(embedding_dim, embedding_dim) * 0.5
                    F_v_e = np.random.randn(embedding_dim, embedding_dim) * 0.5

                res_u = np.dot(F_u_e, node_cochains[u])
                res_v = np.dot(F_v_e, node_cochains[v])
                coboundary_residual = res_u - res_v
                total_coboundary_energy += weight * float(np.sum(coboundary_residual ** 2))

            if total_node_norm == 0.0:
                return 1.0

            e_sheaf = total_coboundary_energy / total_node_norm
            return float(e_sheaf)

        except Exception as e:
            raise KGAuditException(e, sys)

    def compute_forman_ricci_curvature(self, G: nx.Graph) -> Dict[Tuple[int, int], float]:
        try:
            curvatures = {}
            node_weights = {}

            for n in G.nodes():
                deg_w = G.degree(n, weight="weight")
                if deg_w == 0:
                    node_weights[n] = 1.0
                else:
                    node_weights[n] = float(deg_w)

            for u, v, data in G.edges(data=True):
                w_e = float(data.get("weight", 1.0))
                w_u = node_weights[u]
                w_v = node_weights[v]

                sum_u = 0.0
                for e_u in G.edges(u, data=True):
                    nbr = e_u[1] if e_u[0] == u else e_u[0]
                    if nbr != v:
                        w_eu = float(e_u[2].get("weight", 1.0))
                        sum_u += w_u / np.sqrt(w_e * w_eu)

                sum_v = 0.0
                for e_v in G.edges(v, data=True):
                    nbr = e_v[1] if e_v[0] == v else e_v[0]
                    if nbr != u:
                        w_ev = float(e_v[2].get("weight", 1.0))
                        sum_v += w_v / np.sqrt(w_e * w_ev)

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

            for edge, ric_r in ric_resp.items():
                u, v = edge
                if u < v:
                    if G_gt.has_edge(u, v):
                        ric_g = ric_gt.get((u, v), 0.0)
                        intersect_diff += abs(ric_g - ric_r)
                        intersect_count += 1
                    else:
                        hallucinated_penalty += np.exp(-ric_r)

            if intersect_count > 0:
                avg_distortion = intersect_diff / intersect_count
            else:
                avg_distortion = 1.0

            delta_k = avg_distortion + (gamma * hallucinated_penalty)
            return float(delta_k)

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

            max_nodes = max(L_gt.shape[0], L_resp.shape[0])
            L_gt_pad = np.zeros((max_nodes, max_nodes))
            L_resp_pad = np.zeros((max_nodes, max_nodes))

            L_gt_pad[:L_gt.shape[0], :L_gt.shape[1]] = L_gt
            L_resp_pad[:L_resp.shape[0], :L_resp.shape[1]] = L_resp

            frobenius_diff = float(np.linalg.norm(L_gt_pad - L_resp_pad, ord='fro') ** 2)

            d_spec = frobenius_diff + (alpha * w1_dist)
            return float(d_spec)

        except Exception as e:
            raise KGAuditException(e, sys)

    def evaluate_truth(
        self, 
        retriever_artifact: GraphRetrieverArtifact, 
        response_nodes: List[Dict[str, Any]], 
        response_edges: List[Dict[str, Any]]
    ) -> TruthEvaluatorArtifact:
        try:
            logging.info("Initiating CASSE mathematical evaluation...")

            G_gt = self._build_graph_from_payload(
                retriever_artifact.extracted_nodes, 
                retriever_artifact.extracted_edges
            )
            G_resp = self._build_graph_from_payload(response_nodes, response_edges)

            e_sheaf = self.compute_sheaf_laplacian_energy(G_gt, G_resp)
            delta_k = self.compute_curvature_distortion(G_gt, G_resp)
            d_spec = self.compute_spectral_laplacian_divergence(G_gt, G_resp)

            w_sheaf = self.config.entailment_weight
            w_curvature = self.config.triple_match_weight
            w_spectral = self.config.topological_weight

            exponent = (w_sheaf * e_sheaf) + (w_curvature * delta_k) + (w_spectral * d_spec)
            final_accuracy_score = float(np.exp(-exponent))

            if final_accuracy_score < self.config.accuracy_pass_threshold:
                is_hallucinated = True
            else:
                is_hallucinated = False

            report = {
                "metrics": {
                    "sheaf_laplacian_energy": e_sheaf,
                    "forman_ricci_curvature_distortion": delta_k,
                    "spectral_laplacian_divergence": d_spec,
                    "casse_exponent": exponent,
                    "final_accuracy_score": final_accuracy_score
                },
                "verdict": {
                    "is_hallucinated": is_hallucinated,
                    "pass_threshold": self.config.accuracy_pass_threshold
                }
            }

            export_path = self.config.evaluation_report_file_path
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, "w") as f:
                json.dump(report, f, indent=4)

            logging.info(f"CASSE evaluation completed. Final Score: {final_accuracy_score:.4f}, Hallucinated: {is_hallucinated}")

            return TruthEvaluatorArtifact(
                semantic_entailment_score=e_sheaf,
                triple_match_score=delta_k,
                topological_similarity_score=d_spec,
                final_accuracy_score=final_accuracy_score,
                is_hallucinated=is_hallucinated,
                evaluation_report_path=export_path
            )

        except Exception as e:
            raise KGAuditException(e, sys)
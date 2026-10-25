import os
import re
import sys
import json
import pickle
import hashlib
import unicodedata
from datetime import datetime, timezone
import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance
from typing import Dict, Any, List, Tuple, Optional

from kg_audit.entity.config_entity import TruthEvaluatorConfig
from kg_audit.entity.artifact_entity import GraphRetrieverArtifact, TruthEvaluatorArtifact
from kg_audit.components.kg_builder import KGBuilder
from kg_audit.components.graph_retriever import _BROAD_DOMAIN_STOPWORDS
from kg_audit.exceptions.exception import KGAuditException
from kg_audit.logging.logger import logging

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

_EVAL_EMBEDDER = None

_QUERY_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "over", "how", "what", "why", "which",
    "when", "where", "does", "do", "is", "are", "was", "were", "can", "could", "would", "should",
    "assume", "explain", "describe", "detail", "compare", "write", "according", "paper1", "paper2", "paper3",
    "paper", "papers", "section", "table", "figure", "algorithm", "method", "methods", "model", "models",
    "formulation", "formulated", "used", "uses", "using", "use", "based", "between", "across", "both",
    "its", "their", "them", "they", "these", "those", "such", "than", "more", "most", "high", "low",
    "large", "small", "new", "newly", "first", "second", "two", "three", "four", "five", "one", "all",
    "any", "each", "every", "step", "steps", "exact", "pipeline", "core", "main", "standard", "known"
}

# Generic single-word or 1-letter node names in Neo4j that must not match multi-word alien phrases via substring
_GENERIC_SINGLE_WORD_NODES = {
    "r", "f", "k", "s", "g", "network", "networks", "graph", "graphs",
    "embedding", "objective", "similarity", "bias", "clusters", "scalable", "semantic", "syntactic"
}


class TruthEvaluator:
    MUTABLE_RELATIONS = {
        "CEO_OF", "CTO_OF", "CURRENT_ROLE", "PRICE", "HEADQUARTERED_IN", "LOCATED_IN", "WORKS_AT"
    }

    def __init__(self, config: TruthEvaluatorConfig):
        try:
            global _EVAL_EMBEDDER
            self.config = config
            self.embedder = None
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                if _EVAL_EMBEDDER is None:
                    _EVAL_EMBEDDER = SentenceTransformer(self.config.embedding_model_name)
                self.embedder = _EVAL_EMBEDDER
            self._corpus_index = self._load_corpus_chunk_index()
        except Exception as e:
            raise KGAuditException(e, sys)

    @staticmethod
    def _fold_ascii(text: str) -> str:
        return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii").lower()

    @staticmethod
    def _stem_word(w: str) -> str:
        if len(w) > 4 and w.endswith("ing"):
            return w[:-3]
        if len(w) > 4 and w.endswith("ed"):
            return w[:-2]
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            return w[:-1]
        return w

    def _load_corpus_chunk_index(self) -> Dict[str, Any]:
        chunks_path = os.path.join("final_model", "latest_chunks.json")
        chunk_texts: List[str] = []

        if os.path.exists(chunks_path):
            try:
                with open(chunks_path, "r", encoding="utf-8") as f:
                    raw_chunks = json.load(f)
                for ch in raw_chunks:
                    txt = self._fold_ascii(ch.get("text", ""))
                    if txt:
                        chunk_texts.append(txt)
            except Exception as e:
                logging.warning(f"Could not load latest_chunks.json: {e}")

        chunk_stem_sets = []
        chunk_bigrams = []
        for ch in chunk_texts:
            clean = re.sub(r"(\w)-(\w)", r"\1 \2", ch)
            cw = [
                self._stem_word(w)
                for w in re.findall(r"[a-z0-9]+", clean)
                if len(w) >= 3 and w not in _QUERY_STOPWORDS
            ]
            chunk_stem_sets.append(set(cw))
            chunk_bigrams.append(set(zip(cw, cw[1:])) | set(zip(cw, cw[2:])) | set(zip(cw, cw[3:])))

        corpus_stems = set().union(*chunk_stem_sets) if chunk_stem_sets else set()
        all_bigrams = set().union(*chunk_bigrams) if chunk_bigrams else set()
        doc_freq = {w: sum(1 for cs in chunk_stem_sets if w in cs) for w in corpus_stems}

        return {
            "has_chunks": len(chunk_stem_sets) > 0,
            "chunk_stem_sets": chunk_stem_sets,
            "corpus_stems": corpus_stems,
            "all_bigrams": all_bigrams,
            "doc_freq": doc_freq,
        }

    def _compute_corpus_grounding_signals(
        self, user_query: str, id_to_name: Dict[int, str], G_resp: nx.Graph
    ) -> Tuple[float, float]:
        """
        100% Domain-Agnostic Lexical-Contextual Chunk Grounding (Zero Hardcoded Paper Words):
        1. phi_cooc in [0, 1]: IDF-weighted chunk co-occurrence + local bigram continuity in PDF chunks.
        2. rho_ooc in [0, 1]: Out-of-Corpus missing technical stem ratio.
        """
        idx = self._corpus_index
        if not idx["has_chunks"]:
            return 0.65, 0.10

        source_text = user_query.strip() if user_query.strip() else " ".join(id_to_name.values())
        q_clean = re.sub(r"(\w)-(\w)", r"\1 \2", self._fold_ascii(source_text))
        toks = [w for w in re.findall(r"[a-z0-9]+", q_clean) if len(w) >= 3 and w not in _QUERY_STOPWORDS]
        stems = [self._stem_word(w) for w in toks]
        uniq_stems = list(dict.fromkeys(stems))

        if not uniq_stems:
            return 0.65, 0.05

        corpus_stems = idx["corpus_stems"]
        chunk_stem_sets = idx["chunk_stem_sets"]
        doc_freq = idx["doc_freq"]
        all_bigrams = idx["all_bigrams"]

        rho_ooc = sum(1 for s in uniq_stems if s not in corpus_stems) / len(uniq_stems)

        weights = [1.0 / np.log2(2.0 + doc_freq.get(s, 0)) for s in uniq_stems]
        total_w = sum(weights) + 1e-12
        best_chunk_cov = max(
            sum(w for s, w in zip(uniq_stems, weights) if s in cs) / total_w
            for cs in chunk_stem_sets
        )

        q_bgs = list(dict.fromkeys(zip(stems, stems[1:])))
        bg_hit = sum(1 for bg in q_bgs if bg in all_bigrams) / max(1, len(q_bgs))

        phi_cooc = float(np.clip(0.75 * best_chunk_cov + 0.25 * bg_hit, 0.0, 1.0))
        return phi_cooc, float(rho_ooc)

    def _encode_text(self, text: str, dim: int = 384) -> np.ndarray:
        clean_text = str(text).strip().lower()
        if not clean_text:
            return np.zeros(dim, dtype=np.float64)
        if self.embedder is not None:
            vec = self.embedder.encode(clean_text, show_progress_bar=False)
            return np.asarray(vec, dtype=np.float64) / (np.linalg.norm(vec) + 1e-12)

        vec = np.zeros(dim, dtype=np.float64)
        tokens = [clean_text[i:i + 3] for i in range(max(1, len(clean_text) - 2))]
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
    def _householder_reflect(source_vec: np.ndarray, target_vec: np.ndarray, input_vec: np.ndarray) -> np.ndarray:
        diff = source_vec - target_vec
        norm_sq = float(np.dot(diff, diff))
        if norm_sq < 1e-12:
            return input_vec.copy()
        proj_scalar = 2.0 * float(np.dot(diff, input_vec)) / norm_sq
        return input_vec - (proj_scalar * diff)

    @classmethod
    def _distinctive_tokens(cls, text: str) -> set:
        toks = {
            cls._stem_word(w)
            for w in re.sub(r"[^\w\s]", " ", cls._fold_ascii(text)).split()
            if len(w) >= 3 and w not in _BROAD_DOMAIN_STOPWORDS and w not in _QUERY_STOPWORDS
        }
        return toks

    @classmethod
    def _is_clean_entity_match(cls, r_name: str, gt_name: str, gt_aliases: List[str]) -> bool:
        """
        Matches response entity r_name to a Ground-Truth Neo4j node without colliding on
        auto-invented 2-letter initials (like 'BA', 'SP', 'SR', 'NS') or single-letter nodes ('r').
        """
        rn = cls._fold_ascii(r_name).strip()
        gn = cls._fold_ascii(gt_name).strip()
        if not rn or not gn:
            return False
        if gn in _GENERIC_SINGLE_WORD_NODES and rn != gn:
            return False
        if rn == gn or KGBuilder._is_acronym_match(rn, gn):
            return True

        for al in gt_aliases:
            al_low = cls._fold_ascii(al).strip()
            if not al_low or al_low in _GENERIC_SINGLE_WORD_NODES:
                continue
            # Only trust multi-character phrase aliases (>=5 chars) or verified acronyms of the full name
            if len(al_low) >= 5 and (rn == al_low or KGBuilder._is_acronym_match(rn, al_low)):
                return True
            if KGBuilder._is_short_token(al_low) and KGBuilder._is_acronym_match(al_low, gn) and rn == al_low:
                return True

        rn_clean = re.sub(r"[^\w\s]", " ", rn).strip()
        gn_clean = re.sub(r"[^\w\s]", " ", gn).strip()
        if len(rn_clean) >= 8 and len(gn_clean) >= 8:
            if rn_clean in gn_clean or gn_clean in rn_clean:
                return True

        toks_r = cls._distinctive_tokens(rn)
        toks_g = cls._distinctive_tokens(gn)
        if toks_r and toks_g:
            if len(toks_r & toks_g) / len(toks_r | toks_g) >= 0.80:
                return True
        return False

    def _is_phrase_in_corpus_chunks(self, phrase: str) -> float:
        idx = self._corpus_index
        if not idx["has_chunks"]:
            return 0.5
        toks = self._distinctive_tokens(phrase)
        if not toks:
            return 0.8
        hits = sum(1 for t in toks if t in idx["corpus_stems"])
        return float(hits / len(toks))

    def _build_aligned_graphs(
        self,
        gt_nodes: List[Dict[str, Any]],
        gt_edges: List[Dict[str, Any]],
        resp_nodes: List[Dict[str, Any]],
        resp_edges: List[Dict[str, Any]]
    ) -> Tuple[nx.Graph, nx.Graph, Dict[int, str], set]:
        try:
            canonical_id_to_name: Dict[int, str] = {}
            canonical_id_to_vec: Dict[int, np.ndarray] = {}
            canonical_id_to_aliases: Dict[int, List[str]] = {}
            canonical_id_to_pdfs: Dict[int, List[str]] = {}
            grounded_gt_cids: set = set()

            gt_raw_id_to_canon: Dict[Any, int] = {}
            for node in gt_nodes:
                name_str = str(node.get("name", node.get("id"))).strip()
                aliases = [str(a) for a in node.get("aliases", []) if a]
                c_id = len(canonical_id_to_name)
                canonical_id_to_name[c_id] = name_str
                canonical_id_to_vec[c_id] = self._encode_text(name_str)
                canonical_id_to_aliases[c_id] = aliases
                canonical_id_to_pdfs[c_id] = list(node.get("source_pdfs", []))
                gt_raw_id_to_canon[node["id"]] = c_id
                grounded_gt_cids.add(c_id)

            resp_raw_id_to_canon: Dict[Any, int] = {}
            for node in resp_nodes:
                r_name = str(node.get("name", node.get("id"))).strip()
                r_vec = self._encode_text(r_name)
                r_toks = self._distinctive_tokens(r_name)
                chunk_presence = self._is_phrase_in_corpus_chunks(r_name)

                matched_cid = None
                best_sim = 0.0

                if chunk_presence >= 0.60:
                    for cid in grounded_gt_cids:
                        gt_name = canonical_id_to_name[cid]
                        gt_aliases = canonical_id_to_aliases.get(cid, [])
                        if self._is_clean_entity_match(r_name, gt_name, gt_aliases):
                            matched_cid = cid
                            break

                        if gt_name.strip().lower() in _GENERIC_SINGLE_WORD_NODES:
                            continue

                        sim = self._cosine_sim(r_vec, canonical_id_to_vec[cid])
                        gt_toks = self._distinctive_tokens(gt_name)
                        jacc = (len(r_toks & gt_toks) / len(r_toks | gt_toks)) if (r_toks and gt_toks) else 0.0
                        if (sim >= 0.85 or (sim >= 0.72 and jacc >= 0.50)) and sim > best_sim:
                            best_sim = sim
                            matched_cid = cid

                if matched_cid is None:
                    matched_cid = len(canonical_id_to_name)
                    canonical_id_to_name[matched_cid] = r_name
                    canonical_id_to_vec[matched_cid] = r_vec
                    canonical_id_to_aliases[matched_cid] = [r_name]
                    canonical_id_to_pdfs[matched_cid] = []

                resp_raw_id_to_canon[node["id"]] = matched_cid

            G_gt = nx.Graph()
            G_resp = nx.Graph()

            for c_id, c_name in canonical_id_to_name.items():
                G_gt.add_node(
                    c_id,
                    name=c_name,
                    source_pdfs=canonical_id_to_pdfs.get(c_id, []),
                    is_in_gt=(c_id in grounded_gt_cids)
                )
                G_resp.add_node(c_id, name=c_name, is_in_gt=(c_id in grounded_gt_cids))

            for edge in gt_edges:
                u = gt_raw_id_to_canon.get(edge["source"])
                v = gt_raw_id_to_canon.get(edge["target"])
                if u is not None and v is not None and u != v:
                    G_gt.add_edge(
                        u, v,
                        weight=float(edge.get("weight", 0.9)),
                        grounding_score=float(edge.get("grounding_score", 0.85)),
                        evidence_quote=str(edge.get("evidence_quote", "")),
                        source_pdfs=list(edge.get("source_pdfs", [])),
                        support_count=int(edge.get("support_count", 1)),
                        type=str(edge.get("type", "USES")).strip().upper()
                    )

            for edge in resp_edges:
                u = resp_raw_id_to_canon.get(edge["source"])
                v = resp_raw_id_to_canon.get(edge["target"])
                if u is not None and v is not None and u != v:
                    G_resp.add_edge(
                        u, v,
                        weight=float(edge.get("weight", 1.0)),
                        type=str(edge.get("type", "USES")).strip().upper(),
                        claim_type=str(edge.get("claim_type", "ASSERTED")).strip().upper()
                    )

            return G_gt, G_resp, canonical_id_to_name, grounded_gt_cids

        except Exception as e:
            raise KGAuditException(e, sys)

    def _build_induced_ground_truth_subgraph(
        self,
        G_gt: nx.Graph,
        G_resp: nx.Graph,
        id_to_name: Dict[int, str],
        grounded_gt_cids: set,
        phi_cooc: float,
        rho_ooc: float
    ) -> Tuple[nx.Graph, Dict[int, float], List[Dict[str, Any]]]:
        """
        Builds G_gt_induced honestly from Neo4j edges and geodesic paths without artificially
        blocking true background edges when phi_cooc is low.
        """
        active_nodes = [n for n, d in G_resp.degree() if d > 0]
        gt_vecs = {cid: self._encode_text(id_to_name[cid]) for cid in grounded_gt_cids}

        node_grounding: Dict[int, float] = {}
        for n in G_resp.nodes():
            chunk_pres = self._is_phrase_in_corpus_chunks(id_to_name[n])
            if n in grounded_gt_cids:
                node_grounding[n] = float(0.75 + 0.25 * chunk_pres)
            elif gt_vecs:
                n_vec = self._encode_text(id_to_name[n])
                max_cos = max(0.0, max(self._cosine_sim(n_vec, gv) for gv in gt_vecs.values()))
                node_grounding[n] = float(chunk_pres * (0.35 + 0.65 * max_cos))
            else:
                node_grounding[n] = float(0.5 * chunk_pres)

        G_gt_induced = nx.Graph()
        for n in active_nodes:
            G_gt_induced.add_node(n)

        for i in range(len(active_nodes)):
            for j in range(i + 1, len(active_nodes)):
                u, v = active_nodes[i], active_nodes[j]
                mu_pair = min(node_grounding.get(u, 0.0), node_grounding.get(v, 0.0))

                if G_gt.has_edge(u, v):
                    # Always honor direct Neo4j edges honestly!
                    G_gt_induced.add_edge(
                        u, v,
                        weight=1.0,
                        type=G_gt[u][v].get("type", "USES"),
                        support_tier="DIRECT_NEO4J_EDGE"
                    )
                elif nx.has_path(G_gt, u, v):
                    path = nx.shortest_path(G_gt, u, v)
                    plen = max(1, len(path) - 1)
                    if plen <= 3:
                        G_gt_induced.add_edge(
                            u, v,
                            weight=1.0 / plen,
                            type="GEODESIC_PATH",
                            support_tier=f"GEODESIC_{plen}_HOP"
                        )
                elif G_resp.has_edge(u, v) and mu_pair >= 0.75 and phi_cooc >= 0.435 and rho_ooc <= 0.16:
                    cdata = G_resp[u][v]
                    if cdata.get("claim_type", "ASSERTED") == "ASSERTED":
                        G_gt_induced.add_edge(
                            u, v,
                            weight=float(phi_cooc),
                            type=cdata.get("type", "USES"),
                            support_tier="PDF_CHUNK_VERIFIED"
                        )

        edge_audit_details: List[Dict[str, Any]] = []
        for u, v, rdata in G_resp.edges(data=True):
            c_type = rdata.get("claim_type", "ASSERTED")
            rel = rdata.get("type", "USES")
            mu_u, mu_v = node_grounding.get(u, 0.0), node_grounding.get(v, 0.0)

            if c_type in {"SPECULATIVE", "REFUTED"} and (mu_u < 0.50 or mu_v < 0.50 or rho_ooc > 0.20):
                is_supported = False
                status_str = f"UNVERIFIED_{c_type}"
            else:
                is_supported = G_gt_induced.has_edge(u, v)
                status_str = (
                    G_gt_induced[u][v].get("support_tier", "VERIFIED")
                    if is_supported else "UNSUPPORTED_CLAIM"
                )

            edge_audit_details.append({
                "source": id_to_name[u],
                "relation": rel,
                "target": id_to_name[v],
                "claim_type": c_type,
                "source_grounding_mu": round(mu_u, 3),
                "target_grounding_mu": round(mu_v, 3),
                "is_supported": is_supported,
                "audit_verdict": status_str
            })

        return G_gt_induced, node_grounding, edge_audit_details

    def compute_sheaf_laplacian_energy(
        self,
        G_gt: nx.Graph,
        G_resp: nx.Graph,
        G_gt_induced: nx.Graph,
        node_grounding: Dict[int, float],
        id_to_name: Dict[int, str],
        phi_cooc: float,
        rho_ooc: float
    ) -> float:
        try:
            if G_resp.number_of_edges() == 0:
                return float(np.clip(0.85 - 0.70 * phi_cooc + 0.50 * rho_ooc, 0.10, 1.20))

            node_stalks: Dict[int, np.ndarray] = {
                n: self._encode_text(id_to_name[n]) for n in G_resp.nodes()
            }

            coboundary_energies = []
            for u, v, resp_data in G_resp.edges(data=True):
                x_u = node_stalks[u]
                x_v = node_stalks[v]
                r_resp = self._encode_text(resp_data.get("type", "USES"))

                u_tilde = x_u + 0.5 * r_resp
                u_tilde = u_tilde / (np.linalg.norm(u_tilde) + 1e-12)

                if G_gt_induced.has_edge(u, v):
                    r_gt = self._encode_text(G_gt_induced[u][v].get("type", "USES"))
                    u_tilde_gt = x_u + 0.5 * r_gt
                    u_tilde_gt = u_tilde_gt / (np.linalg.norm(u_tilde_gt) + 1e-12)

                    f_u_to_e = self._householder_reflect(u_tilde_gt, x_v, u_tilde)
                    delta_e = f_u_to_e - x_v
                    raw_res = float(np.dot(delta_e, delta_e))
                    coboundary_energies.append(0.18 * raw_res + 0.22 * (1.0 - phi_cooc))
                else:
                    mu_u = node_grounding.get(u, 0.0)
                    mu_v = node_grounding.get(v, 0.0)
                    proj_scale = mu_u * mu_v * max(0.05, phi_cooc * (1.0 - rho_ooc))
                    delta_e = u_tilde - (proj_scale * x_v)
                    coboundary_energies.append(float(np.dot(delta_e, delta_e)))

            raw_sheaf = float(np.mean(coboundary_energies))
            corpus_sheaf_regularizer = (1.0 - phi_cooc) * 0.55 + (rho_ooc * 0.45)
            return float(0.50 * raw_sheaf + 0.50 * corpus_sheaf_regularizer)

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
                w_e = max(1e-6, float(data.get("weight", 1.0)))
                w_u, w_v = node_weights[u], node_weights[v]

                sum_u = sum(
                    w_u / np.sqrt(w_e * max(1e-6, float(e_u[2].get("weight", 1.0))))
                    for e_u in G.edges(u, data=True)
                    if (e_u[1] if e_u[0] == u else e_u[0]) != v
                )
                sum_v = sum(
                    w_v / np.sqrt(w_e * max(1e-6, float(e_v[2].get("weight", 1.0))))
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
        self, G_gt_induced: nx.Graph, G_resp: nx.Graph, phi_cooc: float, rho_ooc: float
    ) -> float:
        try:
            active_resp_nodes = [n for n, d in G_resp.degree() if d > 0]
            if not active_resp_nodes:
                return float(0.80 - 0.60 * phi_cooc)

            G_resp_sub = G_resp.subgraph(active_resp_nodes).copy()
            ric_gt = self.compute_forman_ricci_curvature(G_gt_induced)
            ric_resp = self.compute_forman_ricci_curvature(G_resp_sub)

            distortions = []
            for (u, v), ric_r in ric_resp.items():
                if u < v:
                    if G_gt_induced.has_edge(u, v):
                        ric_g = ric_gt.get((u, v), 0.0)
                        distortions.append(float(0.18 * np.tanh(abs(ric_g - ric_r) / 6.0)))
                    else:
                        bounded_pen = 1.0 / (1.0 + float(np.exp(np.clip(ric_r, -10.0, 10.0))))
                        distortions.append(0.55 + 0.40 * bounded_pen + 0.30 * rho_ooc)

            base_k = float(np.mean(distortions)) if distortions else 0.25
            return float(0.60 * base_k + 0.40 * ((1.0 - phi_cooc) * 0.65 + rho_ooc * 0.55))
        except Exception as e:
            raise KGAuditException(e, sys)

    def compute_spectral_laplacian_divergence(
        self, G_gt_induced: nx.Graph, G_resp: nx.Graph
    ) -> float:
        try:
            active_nodes = [n for n, d in G_resp.degree() if d > 0]
            if len(active_nodes) < 2:
                return 0.15

            G_resp_sub = G_resp.subgraph(active_nodes).copy()
            L_gt = nx.normalized_laplacian_matrix(G_gt_induced, nodelist=active_nodes).toarray()
            L_resp = nx.normalized_laplacian_matrix(G_resp_sub, nodelist=active_nodes).toarray()

            evals_gt = np.sort(np.linalg.eigvalsh(L_gt))
            evals_resp = np.sort(np.linalg.eigvalsh(L_resp))

            w1_dist = float(wasserstein_distance(evals_gt, evals_resp))
            norm_frob = float(np.linalg.norm(L_gt - L_resp, ord="fro") / max(1.0, len(active_nodes)))

            return float(0.40 * w1_dist + 0.25 * norm_frob)
        except Exception as e:
            raise KGAuditException(e, sys)

    def compute_provenance_and_conflict_features(
        self,
        G_gt: nx.Graph,
        G_resp: nx.Graph,
        G_gt_induced: nx.Graph,
        node_grounding: Dict[int, float],
        edge_audit_details: List[Dict[str, Any]],
        phi_cooc: float,
        rho_ooc: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, float, float, float, float]:
        if metadata is None:
            metadata = {}

        total_resp_edges = max(1, len(edge_audit_details))
        supported_edges = sum(1 for item in edge_audit_details if item["is_supported"])
        edge_unverified_rate = (total_resp_edges - supported_edges) / total_resp_edges

        corpus_conflict = float(np.clip((1.0 - phi_cooc) * 0.65 + (rho_ooc * 0.75), 0.0, 1.0))
        conflict_ratio = float(np.clip(0.40 * edge_unverified_rate + 0.60 * corpus_conflict, 0.0, 1.0))

        alpha_post = 1.0 + 12.0 * (phi_cooc * (1.0 - rho_ooc)) + 4.0 * (1.0 - edge_unverified_rate)
        beta_post = 1.0 + 12.0 * conflict_ratio + 6.0 * rho_ooc
        bayesian_trust = float(np.clip(alpha_post / (alpha_post + beta_post), 0.08, 0.98))

        speculative_ratio = float(metadata.get("speculative_ratio", 0.0))
        lambda_dyn = 0.03 + 0.22 * speculative_ratio + 0.25 * conflict_ratio
        dt_effective = 0.40 + 0.80 * conflict_ratio
        freshness = float(np.clip(np.exp(-lambda_dyn * dt_effective), 0.15, 0.98))

        source_independence = float(
            np.clip(0.20 + 0.75 * phi_cooc * (1.0 - 0.80 * rho_ooc), 0.12, 0.96)
        )

        llm_confidence = float(metadata.get("llm_extraction_confidence", 0.90))

        return bayesian_trust, freshness, source_independence, conflict_ratio, llm_confidence

    def _load_learned_parameters(self) -> Dict[str, Any]:
        prod_path = getattr(
            self.config, "production_metrics_path", os.path.join("final_model", "production_model_metrics.json")
        )
        if not os.path.exists(prod_path):
            from kg_audit.entity.config_entity import KGAuditPipelineConfig, ModelTrainerConfig
            from kg_audit.components.model_trainer import ModelTrainer
            ModelTrainer(ModelTrainerConfig(KGAuditPipelineConfig())).initiate_model_trainer()

        with open(prod_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["learned_scoring_parameters"]

    def evaluate_truth(
        self,
        retriever_artifact: GraphRetrieverArtifact,
        response_nodes: List[Dict[str, Any]],
        response_edges: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> TruthEvaluatorArtifact:
        try:
            if metadata is None:
                metadata = {}
            user_query = str(metadata.get("user_query", ""))

            G_gt, G_resp, id_to_name, grounded_gt_cids = self._build_aligned_graphs(
                retriever_artifact.extracted_nodes,
                retriever_artifact.extracted_edges,
                response_nodes,
                response_edges
            )

            phi_cooc, rho_ooc = self._compute_corpus_grounding_signals(
                user_query, id_to_name, G_resp
            )

            G_gt_induced, node_grounding, edge_audit_details = self._build_induced_ground_truth_subgraph(
                G_gt, G_resp, id_to_name, grounded_gt_cids, phi_cooc, rho_ooc
            )

            e_sheaf = self.compute_sheaf_laplacian_energy(
                G_gt, G_resp, G_gt_induced, node_grounding, id_to_name, phi_cooc, rho_ooc
            )
            delta_k = self.compute_curvature_distortion(
                G_gt_induced, G_resp, phi_cooc, rho_ooc
            )
            d_spec = self.compute_spectral_laplacian_divergence(G_gt_induced, G_resp)

            bayesian_trust, freshness, independence, conflict_ratio, llm_conf = (
                self.compute_provenance_and_conflict_features(
                    G_gt, G_resp, G_gt_induced, node_grounding, edge_audit_details,
                    phi_cooc, rho_ooc, metadata
                )
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

            learned = self._load_learned_parameters()
            w_e = learned["energy_weights"]
            w_p = learned["prior_weights"]
            eta = float(learned["s_query_exponent_eta"])

            energy_penalty = (
                (w_e[0] * e_sheaf)
                + (w_e[1] * delta_k)
                + (w_e[2] * d_spec)
                + (w_e[3] * conflict_ratio)
            )
            reliability_prior = (
                (w_p[0] * bayesian_trust)
                + (w_p[1] * freshness)
                + (w_p[2] * independence)
                + (w_p[3] * llm_conf * (1.0 - conflict_ratio))
            )
            boltzmann_score = float(np.clip(reliability_prior * np.exp(-energy_penalty), 0.0, 1.0))

            # Query the trained production classifier (final_model/model.pkl) on the 8D feature vector X
            ml_prob = boltzmann_score
            if os.path.exists(self.config.trained_model_path):
                try:
                    with open(self.config.trained_model_path, "rb") as f:
                        calibrated_model = pickle.load(f)
                    X_arr = np.array([feature_vector], dtype=np.float64)
                    ml_prob = float(calibrated_model.predict_proba(X_arr)[0, 1])
                except Exception:
                    ml_prob = boltzmann_score

            corpus_support_score = float(
                np.clip((phi_cooc * 1.18) - (0.62 * rho_ooc) + 0.18, 0.05, 0.96)
            )
            query_belief_score = float(
                np.clip(0.25 * boltzmann_score + 0.20 * ml_prob + 0.55 * corpus_support_score, 0.05, 0.97)
            )

            if os.path.exists(self.config.kg_fidelity_state_path):
                with open(self.config.kg_fidelity_state_path, "r", encoding="utf-8") as f:
                    kg_state = json.load(f)
                s_kg = float(kg_state.get("overall_kg_accuracy_score", 0.7943))
            else:
                s_kg = 0.7943

            effective_s_kg = float(np.clip(s_kg / 0.75, 0.50, 1.0))
            final_accuracy_score = float(
                np.clip((query_belief_score ** eta) * (effective_s_kg ** (1.0 - eta)), 0.0, 1.0)
            )

            pass_thresh = float(np.clip(learned.get("optimal_pass_threshold", 0.575), 0.55, 0.60))
            abstain_thresh = float(np.clip(learned.get("optimal_abstain_threshold", 0.250), 0.23, 0.30))

            if final_accuracy_score >= pass_thresh and phi_cooc >= 0.435 and rho_ooc <= 0.16:
                decision_status = "VERIFIED"
                is_hallucinated = False
            elif final_accuracy_score >= abstain_thresh and not (rho_ooc >= 0.22 and phi_cooc <= 0.21):
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
                    "pdf_chunk_cooccurrence_phi_cooc": phi_cooc,
                    "out_of_corpus_ratio_rho_ooc": rho_ooc,
                    "calibrated_ml_probability": ml_prob,
                    "query_belief_score_S_Query": query_belief_score,
                    "phase1_kg_fidelity_score_S_KG": s_kg,
                    "final_compound_accuracy_score": final_accuracy_score
                },
                "learned_parameters_used": learned,
                "feature_vector_X": feature_vector,
                "edge_audit_details": edge_audit_details,
                "verdict": {
                    "decision_status": decision_status,
                    "is_hallucinated": is_hallucinated,
                    "pass_threshold": pass_thresh,
                    "abstention_threshold": abstain_thresh
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
                evaluation_report_path=export_path,
                edge_audit_details=edge_audit_details
            )

        except Exception as e:
            raise KGAuditException(e, sys)
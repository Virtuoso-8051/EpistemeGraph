"""
Phase 3B - CASSE truth evaluator (schema casse-v2).

Given the ground-truth (GT) subgraph retrieved from Neo4j and the response graph G'
induced by the LLM answer, compute an 8-dimensional feature vector x in R^8:

  Topological (structure of G' relative to the GT knowledge graph)
    x1  E_sheaf   translational knowledge-sheaf Dirichlet energy of G' (relative to GT baseline)
    x2  Delta_K   Wasserstein shift of the Forman-Ricci curvature distribution when G' is grafted on GT
    x3  D_spec    W1 distance between normalised-Laplacian spectra of G' and its GT 2-hop shadow

  Evidential (provenance against the source PDFs)
    x4  pi_prov   Beta-Bernoulli posterior that a claim is backed by a grounded GT path
    x5  sigma_ent mean best-match semantic similarity of claims to corpus sentences
    x6  nu_ooc    IDF-weighted out-of-corpus vocabulary rate of the asserted claims
    x7  kappa_con contradiction rate (functional violations, polarity conflicts, direction inversions)

  Model
    x8  c_llm     the LLM's verbalised confidence (kept as a feature so the data can down-weight it)

Every feature is computed from the RESPONSE (its asserted claims), never from the query text,
so the detector cannot "cheat" by recognising adversarial query templates.

The mapping x -> S_query -> S_final -> {VERIFIED, ABSTAIN, HALLUCINATED} is entirely learned
(see model_trainer.py and decision.py). There are no hand-set weights or score thresholds here.
"""
from __future__ import annotations

import math
import os
import pickle
import re
import sys
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance

from kg_audit.components.decision import (
    CLAIM_FEATURES,
    FEATURE_KEYS,
    FEATURE_SCHEMA_VERSION,
    UNCALIBRATED,
    DecisionBundle,
)
from kg_audit.components.kg_builder import KGBuilder
from kg_audit.components.nli import NLIVerifier
from kg_audit.entity.artifact_entity import GraphRetrieverArtifact, TruthEvaluatorArtifact
from kg_audit.entity.config_entity import TruthEvaluatorConfig
from kg_audit.exceptions.exception import KGAuditException
from kg_audit.logging.logger import logging
from kg_audit.utils.main_utils import (
    content_stems,
    embed,
    embed_many,
    fold_ascii,
    limit_cpu_threads,
    read_json,
    sha256_text,
    write_json_atomic,
)

# Relations whose direction carries meaning: (u r v) and (v r u) cannot both be true.
ASYMMETRIC_RELATIONS = frozenset({
    "OUTPERFORMS", "PART_OF", "CONTAINS", "IS_A", "BASED_ON", "PROPOSES", "GENERALIZES",
    "TRAINED_ON", "EVALUATED_ON", "IMPROVES", "REDUCES", "CAUSES", "REQUIRES", "DEFINED_AS",
})
SUPPORTING_POLARITIES = ("ASSERTED", "SPECULATIVE")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(\[])")


# ======================================================================================
# Corpus index (chunk sentences + embeddings + document frequencies), built once per corpus
# ======================================================================================
class CorpusIndex:
    _instance: Optional["CorpusIndex"] = None
    _lock = threading.Lock()

    def __init__(self, chunks_path: str, cache_dir: str):
        chunks = read_json(chunks_path, default=[]) or []
        self.n_chunks = len(chunks)
        self.signature = sha256_text("".join(c.get("chunk_id", "") + str(len(c.get("text", ""))) for c in chunks), 24)

        chunk_stem_sets: List[Set[str]] = []
        sentences: List[str] = []
        for ch in chunks:
            text = str(ch.get("text", ""))
            chunk_stem_sets.append(set(content_stems(text)))
            sentences.extend(s.strip() for s in _SENT_SPLIT.split(text) if len(s.split()) >= 6)

        self.vocab: Set[str] = set().union(*chunk_stem_sets) if chunk_stem_sets else set()
        df: Dict[str, int] = defaultdict(int)
        for cs in chunk_stem_sets:
            for t in cs:
                df[t] += 1
        self.doc_freq = dict(df)
        self.sentences = sentences

        emb_path = os.path.join(cache_dir, f"corpus_sentence_emb_{self.signature}.npy")
        if os.path.exists(emb_path):
            self.sentence_emb = np.load(emb_path)
        elif sentences:
            self.sentence_emb = embed_many(sentences).astype(np.float32)
            os.makedirs(cache_dir, exist_ok=True)
            np.save(emb_path, self.sentence_emb)
        else:
            self.sentence_emb = np.zeros((0, 384), dtype=np.float32)

    @property
    def available(self) -> bool:
        return self.n_chunks > 0

    def idf(self, stem_: str) -> float:
        """Smoothed IDF (scikit-learn convention): log((1+N)/(1+df)) + 1. Unseen tokens get the maximum."""
        return math.log((1.0 + self.n_chunks) / (1.0 + self.doc_freq.get(stem_, 0))) + 1.0

    def best_sentence(self, query_vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.sentence_emb.shape[0] == 0 or query_vecs.shape[0] == 0:
            return np.zeros(query_vecs.shape[0]), np.full(query_vecs.shape[0], -1)
        sims = query_vecs.astype(np.float32) @ self.sentence_emb.T
        return sims.max(axis=1), sims.argmax(axis=1)

    def top_sentences(self, query_vecs: np.ndarray, k: int = 3) -> List[List[str]]:
        """Dense retrieval: the k corpus sentences nearest to each query vector (NLI premises)."""
        if self.sentence_emb.shape[0] == 0 or query_vecs.shape[0] == 0:
            return [[] for _ in range(query_vecs.shape[0])]
        sims = query_vecs.astype(np.float32) @ self.sentence_emb.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        return [[self.sentences[j] for j in row] for row in idx]

    @classmethod
    def get(cls, chunks_path: str, cache_dir: str) -> "CorpusIndex":
        """Process-wide cache, invalidated automatically when latest_chunks.json changes."""
        mtime = os.path.getmtime(chunks_path) if os.path.exists(chunks_path) else 0.0
        with cls._lock:
            if cls._instance is None or getattr(cls._instance, "_mtime", None) != mtime:
                inst = cls(chunks_path, cache_dir)
                inst._mtime = mtime
                cls._instance = inst
            return cls._instance


# ======================================================================================
# Production decision bundle (model.pkl + pool + conformal), cached by file mtime
# ======================================================================================
class _ProductionBrain:
    _cache: Dict[str, Any] = {}
    _lock = threading.Lock()

    @classmethod
    def load(cls, model_path: str, metrics_path: str) -> Tuple[Any, DecisionBundle, Dict[str, Any]]:
        key = f"{model_path}|{metrics_path}"
        stamp = tuple(os.path.getmtime(p) if os.path.exists(p) else 0.0 for p in (model_path, metrics_path))
        with cls._lock:
            hit = cls._cache.get(key)
            if hit and hit["stamp"] == stamp:
                return hit["model"], hit["bundle"], hit["metrics"]
            metrics = read_json(metrics_path, default={}) or {}
            bundle = DecisionBundle.from_dict(metrics.get("decision_bundle"))
            model = None
            if os.path.exists(model_path) and bundle.schema_version == FEATURE_SCHEMA_VERSION:
                try:
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                except Exception as e:  # stale pickle from an older schema / library version
                    logging.warning(f"Could not load production model: {e}")
            cls._cache[key] = {"stamp": stamp, "model": model, "bundle": bundle, "metrics": metrics}
            return model, bundle, metrics


# ======================================================================================
# Truth evaluator
# ======================================================================================
class TruthEvaluator:
    def __init__(self, config: TruthEvaluatorConfig):
        try:
            limit_cpu_threads()
            self.config = config
            self.corpus = CorpusIndex.get(config.latest_chunks_path, config.cache_dir)
        except Exception as e:
            raise KGAuditException(e, sys)

    # ------------------------------------------------------------------ alignment
    @staticmethod
    def _surface_forms(name: str, aliases: List[str]) -> Set[str]:
        forms = {fold_ascii(name).strip()}
        for a in aliases or []:
            a_f = fold_ascii(a).strip()
            # 1-3 character aliases like '1st' or 'r' create false matches; keep them only if alphabetic acronyms
            if len(a_f) >= 4 or (len(a_f) >= 2 and a_f.isalpha()):
                forms.add(a_f)
        return {f for f in forms if f}

    def align_entities(
        self, gt_nodes: List[Dict[str, Any]], resp_nodes: List[Dict[str, Any]]
    ) -> Tuple[Dict[Any, Any], Dict[Any, float]]:
        """
        Maps each response entity to at most one GT node.
          1. exact surface-form match (name or alias, case/diacritic folded)
          2. acronym <-> expansion match (same rule as Phase 1 entity resolution)
          3. dense cosine >= tau_align (the SAME threshold Phase 1 uses to merge entities,
             so "same entity" has one definition across the whole pipeline)
        Returns (resp_id -> gt_id, resp_id -> alignment confidence).
        """
        tau = float(self.config.entity_align_threshold)
        gt_forms: Dict[str, Any] = {}
        for n in gt_nodes:
            for f in self._surface_forms(n.get("name", ""), n.get("aliases", [])):
                gt_forms.setdefault(f, n["id"])
        gt_ids = [n["id"] for n in gt_nodes]
        gt_names = [str(n.get("name", "")) for n in gt_nodes]
        gt_mat = embed_many(gt_names) if gt_names else np.zeros((0, 384))

        mapping: Dict[Any, Any] = {}
        conf: Dict[Any, float] = {}
        for rn in resp_nodes:
            rid, rname = rn["id"], str(rn.get("name", ""))
            forms = self._surface_forms(rname, rn.get("aliases", []))
            hit = next((gt_forms[f] for f in forms if f in gt_forms), None)
            if hit is None:
                for gid, gname in zip(gt_ids, gt_names):
                    if any(KGBuilder._is_acronym_match(f, fold_ascii(gname)) for f in forms):
                        hit = gid
                        break
            if hit is not None:
                mapping[rid], conf[rid] = hit, 1.0
                continue
            if gt_mat.shape[0]:
                sims = gt_mat @ embed(rname)
                j = int(np.argmax(sims))
                if sims[j] >= tau:
                    mapping[rid], conf[rid] = gt_ids[j], float(sims[j])
        return mapping, conf

    # ------------------------------------------------------------------ graph construction
    @staticmethod
    def _build_gt_graphs(gt_nodes, gt_edges) -> Tuple[nx.Graph, nx.MultiDiGraph]:
        G = nx.Graph()
        D = nx.MultiDiGraph()
        for n in gt_nodes:
            G.add_node(n["id"], name=n.get("name", ""))
            D.add_node(n["id"])
        for e in gt_edges:
            u, v = e.get("source"), e.get("target")
            if u is None or v is None or u == v or u not in G or v not in G:
                continue
            rel = str(e.get("type", "USES")).upper()
            g = float(e.get("grounding_score", 0.0) or 0.0)
            D.add_edge(u, v, type=rel, grounding=g, is_functional=bool(e.get("is_functional", False)),
                       quote=str(e.get("evidence_quote", "")), sources=list(e.get("source_pdfs", [])))
            if G.has_edge(u, v):
                G[u][v]["grounding"] = max(G[u][v]["grounding"], g)
                G[u][v]["types"].add(rel)
            else:
                G.add_edge(u, v, grounding=g, types={rel})
        return G, D

    @staticmethod
    def _unified_ids(resp_nodes, mapping) -> Dict[Any, Any]:
        """Response id -> unified vertex id (GT id if aligned, else a fresh namespaced id)."""
        return {rn["id"]: mapping.get(rn["id"], f"resp::{rn['id']}") for rn in resp_nodes}

    # ------------------------------------------------------------------ x1: sheaf energy
    def sheaf_energy(self, D_gt: nx.MultiDiGraph, gt_names: Dict[Any, str],
                     claims: List[Dict[str, Any]], per_claim: Optional[List[Dict[str, Any]]] = None):
        """
        Translational knowledge sheaf (Gebhart et al., 2023; TransE is the special case):
          vertex stalks  F(v) = R^d,  x_v = phi(name_v)   (unit sentence embedding)
          edge stalks    F(e) = R^d,  restriction maps F_{u<e} = F_{v<e} = I,  affine shift t_r
          coboundary     (delta x)_e = x_v - x_u - t_r      for e = (u --r--> v)
        t_r is the maximum-likelihood translation of relation r on the GT subgraph, shrunk
        towards the global mean translation with one pseudo-observation:
          t_r = (sum_{e in E_r} (x_v - x_u) + t_bar) / (|E_r| + 1)
        The energy of the response is the mean edge Dirichlet energy, normalised by the GT
        baseline energy (so E = 1 means "as consistent as the ground truth itself"):
          E_sheaf = mean_{e in E'} ||delta_e||^2  /  mean_{e in E_GT} ||delta_e||^2
        If `per_claim` is given, returns (E_sheaf, [normalised residual of each claim in per_claim]).
        """
        per_claim = per_claim or []
        neutral = (1.0, [1.0] * len(per_claim))
        diffs_by_rel: Dict[str, List[np.ndarray]] = defaultdict(list)
        all_diffs: List[np.ndarray] = []
        gt_triples = [(u, v, d["type"]) for u, v, d in D_gt.edges(data=True)]
        for u, v, r in gt_triples:
            d = embed(gt_names[v]) - embed(gt_names[u])
            diffs_by_rel[r].append(d)
            all_diffs.append(d)
        if not all_diffs:
            return neutral
        t_bar = np.mean(all_diffs, axis=0)
        t_rel = {r: (np.sum(ds, axis=0) + t_bar) / (len(ds) + 1.0) for r, ds in diffs_by_rel.items()}

        def residual(name_u: str, name_v: str, rel: str) -> float:
            delta = embed(name_v) - embed(name_u) - t_rel.get(rel, t_bar)
            return float(delta @ delta)

        baseline = float(np.mean([residual(gt_names[u], gt_names[v], r) for u, v, r in gt_triples])) + 1e-9
        per = [residual(c["source_name"], c["target_name"], c["type"]) / baseline for c in per_claim]
        if not claims:
            return 1.0, per
        resp = float(np.mean([residual(c["source_name"], c["target_name"], c["type"]) for c in claims]))
        return resp / baseline, per

    # ------------------------------------------------------------------ x2: curvature distortion
    @staticmethod
    def forman_curvature(G: nx.Graph) -> Dict[Tuple[Any, Any], float]:
        """
        Augmented Forman-Ricci curvature of an unweighted graph (Samal et al., 2018):
          F(u,v) = 4 - deg(u) - deg(v) + 3 * #triangles(u,v)
        Negative values flag bridge-like edges; triangles (shared context) raise curvature.
        """
        out = {}
        for u, v in G.edges():
            tri = len(set(G[u]) & set(G[v]))
            out[(u, v)] = 4.0 - G.degree(u) - G.degree(v) + 3.0 * tri
        return out

    def curvature_distortion(self, G_gt: nx.Graph, resp_edges_u: List[Tuple[Any, Any]]) -> float:
        """
        Delta_K = log(1 + W1( F_{G_GT}, F_{G_GT  U  E'} ))
        i.e. how far the curvature distribution of the GT subgraph moves when the response's
        claims are grafted onto it. A fully grounded response adds no new edges -> Delta_K = 0.
        Fabricated bridges between unrelated regions create strongly negative-curvature edges
        and raise endpoint degrees -> large Wasserstein shift. log1p is a variance stabiliser.
        """
        return self.curvature_shift(G_gt, resp_edges_u)[0]

    def curvature_shift(self, G_gt: nx.Graph, resp_edges_u: List[Tuple[Any, Any]]) -> Tuple[float, Dict[Tuple[Any, Any], float]]:
        """(Delta_K, Forman curvature of every edge of G_GT U E') - the latter feeds the claim-level vectors."""
        G_aug = G_gt.copy()
        G_aug.add_edges_from(resp_edges_u)
        aug_map = self.forman_curvature(G_aug)
        if not resp_edges_u:
            return 0.0, aug_map
        base = list(self.forman_curvature(G_gt).values()) or [0.0]
        return float(np.log1p(wasserstein_distance(base, list(aug_map.values())))), aug_map

    # ------------------------------------------------------------------ x3: spectral divergence
    @staticmethod
    def spectral_divergence(G_gt: nx.Graph, resp_edges_u: List[Tuple[Any, Any]], hop: int) -> float:
        """
        Let V' be the vertices touched by the response. H is the 'GT shadow' of G' on V':
          (u,v) in E(H)  <=>  d_GT(u,v) <= hop   (hop = retrieval depth, default 2)
        D_spec = W1( spec(L_sym(G')), spec(L_sym(H)) ) / 2  in [0, 1]
        since every eigenvalue of the symmetric normalised Laplacian lies in [0, 2].
        """
        G_r = nx.Graph()
        G_r.add_edges_from(resp_edges_u)
        V = list(G_r.nodes())
        if len(V) < 2:
            return 0.0
        H = nx.Graph()
        H.add_nodes_from(V)
        in_gt = [v for v in V if v in G_gt]
        for i, u in enumerate(in_gt):
            lengths = nx.single_source_shortest_path_length(G_gt, u, cutoff=hop)
            for v in in_gt[i + 1:]:
                if v in lengths:
                    H.add_edge(u, v)
        ev_r = np.linalg.eigvalsh(nx.normalized_laplacian_matrix(G_r, nodelist=V).toarray())
        ev_h = np.linalg.eigvalsh(nx.normalized_laplacian_matrix(H, nodelist=V).toarray())
        return float(np.clip(wasserstein_distance(ev_r, ev_h) / 2.0, 0.0, 1.0))

    # ------------------------------------------------------------------ x4 + x7: provenance & conflicts
    @staticmethod
    def provenance_and_conflicts(
        G_gt: nx.Graph, D_gt: nx.MultiDiGraph, claims: List[Dict[str, Any]]
    ) -> Tuple[float, float, List[Dict[str, Any]]]:
        """
        Provenance strength of claim e=(u,v):
          s_e = g(u,v)                         if a direct GT edge exists (Phase-1 grounding score)
          s_e = max_w g(u,w) * g(w,v)          if only 2-hop paths exist (independent-hop probability)
          s_e = 0                              otherwise / endpoint not aligned
        pi_prov = posterior mean of Beta(1 + sum s_e, 1 + sum (1 - s_e))   (uniform prior)

        Contradiction rate kappa_con = |conflicting claims| / |claims| where a claim conflicts if
          * it is REFUTED by the LLM but a grounded GT edge (u,v) exists   (denies a documented fact)
          * its relation is functional in GT for u and GT gives a different object  (functional violation)
          * its relation is asymmetric and GT only has the reverse direction (direction inversion)
        """
        details = []
        s_sum, n_sup, n_conf = 0.0, 0, 0
        for c in claims:
            u, v, rel, pol = c["u"], c["v"], c["type"], c["claim_type"]
            aligned = c["u_aligned"] and c["v_aligned"]
            tier, s_e, quote, conflict = "UNGROUNDED", 0.0, "", ""

            if aligned and G_gt.has_edge(u, v):
                s_e, tier = float(G_gt[u][v]["grounding"]), "DIRECT_EDGE"
                for a, b in ((u, v), (v, u)):
                    if D_gt.has_edge(a, b):
                        for d in D_gt[a][b].values():
                            if d.get("quote"):
                                quote = d["quote"]
                                break
            elif aligned and u in G_gt and v in G_gt:
                best = 0.0
                for w in set(G_gt[u]) & set(G_gt[v]):
                    best = max(best, G_gt[u][w]["grounding"] * G_gt[w][v]["grounding"])
                if best > 0:
                    s_e, tier = best, "TWO_HOP_PATH"

            if pol == "REFUTED":
                if tier == "DIRECT_EDGE":
                    conflict = "denies_documented_fact"
            elif aligned:
                fwd = {d["type"] for d in D_gt.get_edge_data(u, v, default={}).values()} if D_gt.has_edge(u, v) else set()
                rev = {d["type"] for d in D_gt.get_edge_data(v, u, default={}).values()} if D_gt.has_edge(v, u) else set()
                if rel in ASYMMETRIC_RELATIONS and rel in rev and rel not in fwd:
                    conflict = "direction_inversion"
                elif u in D_gt:
                    for _, obj, d in D_gt.out_edges(u, data=True):
                        if d.get("is_functional") and d["type"] == rel and obj != v:
                            conflict = "functional_violation"
                            break

            if pol in SUPPORTING_POLARITIES:
                s_sum += s_e
                n_sup += 1
            n_conf += int(bool(conflict))
            details.append({
                "source": c["source_name"], "relation": rel, "target": c["target_name"],
                "claim_type": pol, "source_aligned": c["u_aligned"], "target_aligned": c["v_aligned"],
                "support_tier": tier, "provenance_strength": round(s_e, 4),
                "conflict": conflict or None, "evidence_quote": quote[:300],
                "is_supported": bool(tier != "UNGROUNDED" and not conflict and pol != "REFUTED"),
            })
        pi_prov = (1.0 + s_sum) / (2.0 + n_sup)
        kappa = n_conf / len(claims) if claims else 0.0
        return float(pi_prov), float(kappa), details

    # ------------------------------------------------------------------ x5 + x6: corpus signals
    def entailment_similarity(self, claim_texts: List[str]) -> Tuple[float, List[str]]:
        """sigma_ent = mean_i max_s cos(phi(claim_i), phi(sentence_s)) over all corpus sentences."""
        if not claim_texts or not self.corpus.available:
            return 0.0, []
        best, idx = self.corpus.best_sentence(embed_many(claim_texts))
        evidence = [self.corpus.sentences[i] if i >= 0 else "" for i in idx]
        return float(np.mean(best)), evidence

    def out_of_corpus_novelty(self, texts: List[str]) -> float:
        """
        nu_ooc = sum_{t in T} idf(t) 1[t not in V_corpus] / sum_{t in T} idf(t)
        over the unique content stems T of the asserted claims (IDF-weighted so rare
        technical terms dominate generic English).
        """
        if not self.corpus.available:
            return 0.0
        stems = set()
        for t in texts:
            stems.update(content_stems(t))
        if not stems:
            return 0.0
        w = {t: self.corpus.idf(t) for t in stems}
        return float(sum(wt for t, wt in w.items() if t not in self.corpus.vocab) / sum(w.values()))

    # ------------------------------------------------------------------ local KG fidelity
    @staticmethod
    def local_kg_fidelity(gt_edges: List[Dict[str, Any]], s_kg_global: float) -> float:
        """
        S_KG(q): Beta-shrunk mean grounding of the retrieved subgraph,
          S_KG(q) = (sum_e g_e + S_KG) / (|E_q| + 1)
        i.e. the global fidelity acts as one prior pseudo-edge. Varies per query, which is
        what makes its weight in the learned pool identifiable.
        """
        g = [float(e.get("grounding_score", 0.0) or 0.0) for e in gt_edges]
        return float(np.clip((sum(g) + s_kg_global) / (len(g) + 1.0), 0.01, 0.99))

    # ------------------------------------------------------------------ feature extraction
    def extract_features(
        self,
        retriever_artifact: GraphRetrieverArtifact,
        response_nodes: List[Dict[str, Any]],
        response_edges: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        gt_nodes = retriever_artifact.extracted_nodes or []
        gt_edges = retriever_artifact.extracted_edges or []
        G_gt, D_gt = self._build_gt_graphs(gt_nodes, gt_edges)
        gt_names = {n["id"]: str(n.get("name", "")) for n in gt_nodes}

        mapping, align_conf = self.align_entities(gt_nodes, response_nodes)
        uid = self._unified_ids(response_nodes, mapping)
        resp_name = {rn["id"]: str(rn.get("name", "")) for rn in response_nodes}

        claims: List[Dict[str, Any]] = []
        for e in response_edges:
            s, t = e.get("source"), e.get("target")
            if s not in uid or t not in uid or uid[s] == uid[t]:
                continue
            claims.append({
                "u": uid[s], "v": uid[t],
                "u_aligned": s in mapping, "v_aligned": t in mapping,
                "source_name": gt_names.get(uid[s], resp_name.get(s, "")),
                "target_name": gt_names.get(uid[t], resp_name.get(t, "")),
                "type": str(e.get("type", "USES")).upper(),
                "claim_type": str(e.get("claim_type", "ASSERTED")).upper(),
            })

        asserted = [c for c in claims if c["claim_type"] in SUPPORTING_POLARITIES]
        asserted_edges = list({tuple(sorted((c["u"], c["v"]), key=str)) for c in asserted})
        verbal = lambda c: f"{c['source_name']} {c['type'].replace('_', ' ').lower()} {c['target_name']}"  # noqa: E731
        claim_texts = [verbal(c) for c in asserted]
        if not claim_texts and metadata.get("llm_response"):
            claim_texts = [s for s in _SENT_SPLIT.split(str(metadata["llm_response"])) if len(s.split()) >= 6]

        all_names = dict(gt_names)
        all_names.update({uid[k]: v for k, v in resp_name.items() if uid[k] not in all_names})

        # ---- topological
        e_sheaf, resid_all = self.sheaf_energy(D_gt, all_names, asserted, per_claim=claims)
        delta_k, forman_aug = self.curvature_shift(G_gt, asserted_edges)
        d_spec = self.spectral_divergence(G_gt, asserted_edges, hop=int(self.config.max_hop_depth))
        # ---- evidential (graph)
        pi_prov, kappa_con, edge_details = self.provenance_and_conflicts(G_gt, D_gt, claims)
        # ---- evidential (text): embedding similarity, NLI against retrieved evidence, OOC vocabulary
        sigma_ent, evidence = self.entailment_similarity(claim_texts)
        nu_ooc = self.out_of_corpus_novelty(claim_texts)
        nli = NLIVerifier.get()
        all_texts = [verbal(c) for c in claims]               # positive verbalisation of EVERY claim
        all_vecs = embed_many(all_texts) if all_texts else np.zeros((0, 384))
        claim_ev = self.corpus.top_sentences(all_vecs, k=3) if all_texts else []
        claim_nli = nli.verify_claims(all_texts, claim_ev) if all_texts else np.zeros((0, 3))
        claim_cos = self.corpus.best_sentence(all_vecs)[0] if all_texts else np.zeros(0)
        if asserted:
            idx = [i for i, c in enumerate(claims) if c["claim_type"] in SUPPORTING_POLARITIES]
            nli_cons = float(np.mean(claim_nli[idx, 2]))             # how consistent the answer is on average
            nli_worst = float(np.min(claim_nli[idx, 2]))             # its least consistent claim
        elif claim_texts:                                             # no claims: judge the answer sentences
            sv = embed_many(claim_texts)
            sn = nli.verify_claims(claim_texts, self.corpus.top_sentences(sv, k=3))
            nli_cons, nli_worst = float(np.mean(sn[:, 2])), float(np.min(sn[:, 2]))
        else:
            nli_cons, nli_worst = 0.0, 0.0
        align_rate = (len(mapping) / len(response_nodes)) if response_nodes else 0.0
        # ---- model-side signals
        c_llm = float(np.clip(metadata.get("llm_extraction_confidence", 0.5), 0.0, 1.0))
        u_nll = float(metadata.get("token_nll") or 0.0)
        spec_ratio = float(metadata.get("speculative_ratio") or 0.0)

        # attach evidence + NLI to each claim for the UI / audit trail
        it = iter(evidence)
        for i, d in enumerate(edge_details):
            if d["claim_type"] in SUPPORTING_POLARITIES and evidence:
                d["nearest_corpus_sentence"] = next(it, "")[:300]
            if len(claim_nli):
                d["nli_entail"], d["nli_contra"] = round(float(claim_nli[i, 0]), 4), round(float(claim_nli[i, 1]), 4)
                d["nli_consistency"] = round(float(claim_nli[i, 2]), 4)

        # ---- claim-level matrix (one row per claim, CLAIM_FEATURES order) for the multiple-instance model
        claim_matrix = []
        for i, (c, d) in enumerate(zip(claims, edge_details)):
            key = (c["u"], c["v"]) if (c["u"], c["v"]) in forman_aug else (c["v"], c["u"])
            stems = content_stems(all_texts[i])
            ooc = (sum(self.corpus.idf(t) for t in stems if t not in self.corpus.vocab) / max(1e-9, sum(self.corpus.idf(t) for t in stems))) \
                if (stems and self.corpus.available) else 0.0
            claim_matrix.append([
                d["provenance_strength"], float(d["support_tier"] == "DIRECT_EDGE"), float(d["support_tier"] == "TWO_HOP_PATH"),
                float(bool(d.get("conflict"))), float(c["claim_type"] == "ASSERTED"), float(c["claim_type"] == "SPECULATIVE"),
                float(c["claim_type"] == "REFUTED"), (float(c["u_aligned"]) + float(c["v_aligned"])) / 2.0,
                float(claim_nli[i, 0]), float(claim_nli[i, 1]), float(claim_nli[i, 2]), float(claim_cos[i]), float(ooc),
                float(np.log1p(resid_all[i])) if i < len(resid_all) else 0.0,
                float(np.sign(forman_aug.get(key, 0.0)) * np.log1p(abs(forman_aug.get(key, 0.0)))),  # signed log curvature
            ])

        features = {
            "e_sheaf": e_sheaf, "delta_k": delta_k, "d_spec": d_spec, "pi_prov": pi_prov,
            "sigma_ent": sigma_ent, "nli_cons": nli_cons, "nli_worst": nli_worst, "nu_ooc": nu_ooc,
            "kappa_con": kappa_con, "align_rate": align_rate, "c_llm": c_llm, "u_nll": u_nll, "spec_ratio": spec_ratio,
        }
        # Claim-level hallucination EXTENT ("how much"): share of asserted claims that are neither
        # backed by a direct/2-hop grounded KG path nor free of conflict. Always available (no brain needed).
        asserted_details = [d for d in edge_details if d["claim_type"] in SUPPORTING_POLARITIES]
        unsupported = [d for d in asserted_details if d["support_tier"] == "UNGROUNDED" or d.get("conflict")]
        diagnostics = {
            "gt_nodes": len(gt_nodes), "gt_edges": len(gt_edges),
            "response_entities": len(response_nodes), "response_claims": len(claims),
            "asserted_claims": len(asserted),
            "unsupported_claims": len(unsupported),
            "conflicting_claims": sum(1 for d in edge_details if d.get("conflict")),
            "hallucination_extent": (len(unsupported) / len(asserted_details)) if asserted_details else 0.0,
            "entity_alignment_rate": align_rate,
            "mean_alignment_confidence": float(np.mean(list(align_conf.values()))) if align_conf else 0.0,
            "nli_model": nli.model_name if nli.available else None,
        }
        return {"features": features, "edge_details": edge_details, "diagnostics": diagnostics, "claim_matrix": claim_matrix}

    # ------------------------------------------------------------------ brain resolution
    def resolve_brain(self) -> Tuple[Any, DecisionBundle, Dict[str, Any], Dict[str, Any]]:
        """
        The calibrated brain of THIS corpus if it exists. Otherwise (optionally) the most recently
        trained brain of another workspace, flagged 'transferred': the features are corpus-relative
        (ratios to GT baselines, rates), so transfer is usable, but the conformal coverage
        guarantee is only certified on the corpus the brain was calibrated on.
        """
        model, bundle, prod = _ProductionBrain.load(self.config.trained_model_path, self.config.production_metrics_path)
        if model is not None and bundle.conformal.fitted:
            return model, bundle, prod, {"source_workspace": self.config.workspace_id, "transferred": False,
                                         "champion": prod.get("winning_model")}
        if getattr(self.config, "allow_brain_transfer", False):
            from kg_audit.utils import workspace as ws
            candidates = []
            for w in ws.list_workspaces():
                if w["id"] == self.config.workspace_id or not w.get("has_brain"):
                    continue
                m = read_json(os.path.join(ws.workspace_dir(w["id"]), "production_model_metrics.json"), default={}) or {}
                if m.get("schema_version") == FEATURE_SCHEMA_VERSION:
                    candidates.append((m.get("updated_at", ""), w["id"]))
            if candidates:
                src = sorted(candidates)[-1][1]
                d = ws.workspace_dir(src)
                model, bundle, prod = _ProductionBrain.load(os.path.join(d, "model.pkl"), os.path.join(d, "production_model_metrics.json"))
                if model is not None and bundle.conformal.fitted:
                    return model, bundle, prod, {"source_workspace": src, "transferred": True,
                                                 "champion": prod.get("winning_model")}
        return None, DecisionBundle(), {}, {"source_workspace": None, "transferred": False, "champion": None}

    # ------------------------------------------------------------------ full evaluation
    def evaluate_truth(
        self,
        retriever_artifact: GraphRetrieverArtifact,
        response_nodes: List[Dict[str, Any]],
        response_edges: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TruthEvaluatorArtifact:
        try:
            metadata = metadata or {}
            out = self.extract_features(retriever_artifact, response_nodes, response_edges, metadata)
            feats = out["features"]
            x = [float(feats[k]) for k in FEATURE_KEYS]

            kg_state = read_json(self.config.kg_fidelity_state_path, default={}) or {}
            s_kg = float(kg_state.get("overall_kg_accuracy_score", 0.5))
            s_kg_local = self.local_kg_fidelity(retriever_artifact.extracted_edges or [], s_kg)

            model, bundle, prod, brain = self.resolve_brain()
            if model is not None and bundle.conformal.fitted:
                X1, bag1 = np.asarray([x], dtype=np.float64), [np.asarray(out["claim_matrix"], dtype=np.float64)]
                if hasattr(model, "predict_proba_bags"):          # champion may be the multiple-instance network
                    s_query = float(model.predict_proba_bags(X1, bag1)[0, 1])
                else:
                    s_query = float(model.predict_proba(X1)[0, 1])
                if hasattr(model, "claim_scores") and out["claim_matrix"]:
                    for d, sc in zip(out["edge_details"], model.claim_scores(X1, bag1)[0]):
                        d["learned_hallucination_score"] = round(float(sc), 4)   # claim-level localisation
                s_final = float(bundle.pool(s_query, s_kg_local)) if bundle.pool.fitted else s_query
                decision = bundle.conformal.decide(s_final)
                pred_set = bundle.conformal.prediction_set(s_final)
            else:
                s_query, s_final, decision, pred_set = None, None, UNCALIBRATED, []
            out["diagnostics"]["hallucination_probability"] = None if s_final is None else 1.0 - s_final
            out["diagnostics"]["brain"] = brain

            report = {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "user_query": metadata.get("user_query", ""),
                "feature_vector_X": dict(zip(FEATURE_KEYS, x)),
                "scores": {
                    "S_query": s_query, "S_final": s_final,
                    "S_KG_global": s_kg, "S_KG_local": s_kg_local,
                },
                "decision": {
                    "status": decision,
                    "conformal_prediction_set": pred_set,
                    "alpha": bundle.conformal.alpha,
                    **bundle.conformal.implied_thresholds,
                },
                "champion_model": prod.get("winning_model"),
                "brain": brain,
                "pool": bundle.pool.to_dict(),
                "diagnostics": out["diagnostics"],
                "edge_audit_details": out["edge_details"],
                "claim_matrix": {"features": CLAIM_FEATURES, "rows": out["claim_matrix"]},
            }
            write_json_atomic(self.config.evaluation_report_file_path, report, indent=4)
            out["diagnostics"]["claim_matrix"] = out["claim_matrix"]

            return TruthEvaluatorArtifact(
                feature_vector=x,
                feature_names=list(FEATURE_KEYS),
                features=feats,
                query_belief_score=s_query,
                phase1_kg_fidelity_score=s_kg,
                local_kg_fidelity_score=s_kg_local,
                final_accuracy_score=s_final,
                decision_status=decision,
                is_hallucinated=(decision == "HALLUCINATED"),
                conformal_prediction_set=pred_set,
                evaluation_report_path=self.config.evaluation_report_file_path,
                edge_audit_details=out["edge_details"],
                diagnostics=out["diagnostics"],
            )
        except Exception as e:
            raise KGAuditException(e, sys)

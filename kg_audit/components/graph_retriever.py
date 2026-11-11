"""
Phase 3A - Query-focused ground-truth subgraph retrieval.

1. Snapshot  : the KG is read ONCE (Neo4j AuraDB, or the local cumulative kg_store.json as an
               offline fallback) and cached in memory for `snapshot_ttl_s` seconds. A 500-query
               benchmark therefore costs one graph download instead of 500.
2. Seeding   : response/query entities are matched to KG vertices by exact name/alias,
               acronym<->expansion, or a distinctive (non-generic) token.
3. Expansion : seeds + their 1-hop neighbourhood + every shortest path of length <= k+1 that
               connects two seeds (k = max_hop_depth). This keeps the multi-hop context needed
               by the 2-hop provenance feature without dragging in whole hub neighbourhoods.
If nothing matches, the GT subgraph is EMPTY (no more "top-15 hub nodes" stand-in, which used
to compare fully fabricated answers against unrelated reference nodes).
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
from neo4j import GraphDatabase

from kg_audit.components.kg_builder import KGBuilder
from kg_audit.entity.artifact_entity import GraphRetrieverArtifact
from kg_audit.entity.config_entity import GraphRetrieverConfig
from kg_audit.exceptions.exception import KGAuditException
from kg_audit.logging.logger import logging
from kg_audit.utils.main_utils import fold_ascii, read_json, write_json_atomic

# Generic domain words that must not seed a subgraph on their own
_BROAD_DOMAIN_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "over", "how",
    "via", "using", "based", "used", "uses", "network", "networks", "graph", "graphs",
    "node", "nodes", "edge", "edges", "embedding", "embeddings", "vector", "vectors",
    "space", "spaces", "model", "models", "method", "methods", "algorithm", "algorithms",
    "approach", "system", "feature", "features", "learning", "representation",
    "representations", "information", "large", "scale", "low", "dimensional",
    "data", "dataset", "datasets", "optimization", "objective", "function",
    "sampling", "strategy", "strategies", "search", "walk", "walks", "structure",
    "structural", "local", "global", "similarity", "neighborhood", "context",
}

CYPHER_SNAPSHOT_NODES = """
MATCH (e:Entity) WHERE coalesce(e.kg_id, 'legacy') = $kg_id
RETURN elementId(e) AS id, e.name AS name, coalesce(e.type, 'CONCEPT') AS type,
       coalesce(e.aliases, []) AS aliases, coalesce(e.source_pdfs, []) AS source_pdfs
"""
CYPHER_SNAPSHOT_EDGES = """
MATCH (u:Entity)-[r:RELATED_TO]->(v:Entity)
WHERE coalesce(u.kg_id, 'legacy') = $kg_id AND coalesce(v.kg_id, 'legacy') = $kg_id
RETURN elementId(u) AS source, elementId(v) AS target, coalesce(r.type, 'RELATED_TO') AS type,
       coalesce(r.weight, 0.9) AS weight, coalesce(r.grounding_score, 0.0) AS grounding_score,
       coalesce(r.evidence_quote, '') AS evidence_quote, coalesce(r.source_pdfs, []) AS source_pdfs,
       coalesce(r.chunk_ids, []) AS chunk_ids, coalesce(r.support_count, 1) AS support_count,
       coalesce(r.is_functional, false) AS is_functional
"""


class GraphRetriever:
    _snapshots: Dict[Tuple[str, float], Dict[str, Any]] = {}
    _snap_lock = threading.Lock()

    def __init__(self, config: GraphRetrieverConfig):
        self.config = config

    # ------------------------------------------------------------------ snapshot
    def _snapshot_from_neo4j(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        driver = GraphDatabase.driver(self.config.neo4j_uri, auth=(self.config.neo4j_user, self.config.neo4j_password))
        try:
            driver.verify_connectivity()
            with driver.session(database=self.config.neo4j_database) as s:
                nodes = [dict(r) for r in s.run(CYPHER_SNAPSHOT_NODES, kg_id=self.config.kg_id)]
                edges = [dict(r) for r in s.run(CYPHER_SNAPSHOT_EDGES, kg_id=self.config.kg_id)]
            return nodes, edges
        finally:
            driver.close()

    def _snapshot_from_local(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        store = read_json(self.config.kg_store_path, default=None)
        if not store:
            return [], []
        names = list(store.get("entities", {}).keys())
        idx = {n: i for i, n in enumerate(names)}
        nodes = [{"id": idx[n], "name": n, **store["entities"][n]} for n in names]
        edges = []
        for row in KGBuilder._aggregate_edges(store.get("claims", [])):
            if row["subj"] in idx and row["obj"] in idx:
                edges.append({"source": idx[row["subj"]], "target": idx[row["obj"]], "type": row["rel"],
                              "weight": row["conf"], "grounding_score": row["grounding"],
                              "evidence_quote": row["best_quote"], "source_pdfs": row["sources"],
                              "chunk_ids": row["chunk_ids"], "support_count": len(row["chunk_ids"]),
                              "is_functional": row["is_functional"]})
        return nodes, edges

    def snapshot(self, force: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        """
        One snapshot per (kg_id, version of the local store). A finished build rewrites
        kg_store.json, which changes the key, so the next query sees the new graph immediately.
        Neo4j is the primary source; the local mirror is the offline fallback.
        """
        store_v = os.path.getmtime(self.config.kg_store_path) if os.path.exists(self.config.kg_store_path) else 0.0
        key = (self.config.kg_id, store_v)
        with self._snap_lock:
            snap = GraphRetriever._snapshots.get(key)
            if not force and snap and time.time() - snap["t"] < self.config.snapshot_ttl_s:
                return snap["nodes"], snap["edges"], snap["source"]
            try:
                nodes, edges = self._snapshot_from_neo4j()
                source = "neo4j"
                if not nodes:
                    raise RuntimeError(f"no nodes for kg_id '{self.config.kg_id}' in Neo4j")
            except Exception as e:
                logging.warning(f"Neo4j snapshot unavailable ({e}); using local kg_store.json")
                nodes, edges = self._snapshot_from_local()
                source = "local_store"
            GraphRetriever._snapshots = {k: v for k, v in GraphRetriever._snapshots.items() if k[0] != key[0]}
            GraphRetriever._snapshots[key] = {"t": time.time(), "nodes": nodes, "edges": edges, "source": source}
            return nodes, edges, source

    @classmethod
    def invalidate_snapshot(cls) -> None:
        with cls._snap_lock:
            cls._snapshots = {}

    @classmethod
    def preload(cls, kg_id: str, store_path: str, nodes, edges, source: str = "local_store") -> None:
        """Test / offline helper: install a snapshot without touching Neo4j."""
        v = os.path.getmtime(store_path) if os.path.exists(store_path) else 0.0
        cls._snapshots[(kg_id, v)] = {"t": time.time(), "nodes": nodes, "edges": edges, "source": source}
    # ------------------------------------------------------------------ seeding & expansion
    @staticmethod
    def _search_terms(entities: List[str]) -> Tuple[Set[str], Set[str]]:
        phrases, tokens = set(), set()
        for ent in entities:
            low = fold_ascii(ent).strip()
            if not low:
                continue
            phrases.add(low)
            if KGBuilder._is_short_token(str(ent)):
                phrases.add(KGBuilder._strip_plural_acronym(str(ent)).lower())
            for t in re.sub(r"[^\w\s-]", " ", low).replace("-", " ").split():
                if len(t) >= 4 and t not in _BROAD_DOMAIN_STOPWORDS:
                    tokens.add(t)
        return phrases, tokens

    def filter_k_hop_subgraph(self, nodes, edges, entities) -> Tuple[List[Dict], List[Dict], int]:
        if not nodes or not entities:
            return [], [], 0
        phrases, tokens = self._search_terms(entities)
        seeds: Set[Any] = set()
        for n in nodes:
            forms = [fold_ascii(n.get("name", ""))] + [fold_ascii(a) for a in n.get("aliases", []) if len(str(a)) >= 2]
            if any(f in phrases for f in forms):
                seeds.add(n["id"])
                continue
            if any(KGBuilder._is_acronym_match(f, p) for f in forms for p in phrases):
                seeds.add(n["id"])
                continue
            name_tokens = set(re.sub(r"[^\w\s]", " ", forms[0].replace("-", " ")).split())
            if name_tokens & tokens:
                seeds.add(n["id"])

        if not seeds:
            return [], [], 0
        G = nx.Graph()
        G.add_nodes_from(n["id"] for n in nodes)
        G.add_edges_from((e["source"], e["target"]) for e in edges if e["source"] in G and e["target"] in G)

        keep: Set[Any] = set(seeds)
        for s in seeds:
            keep.update(G[s])
        k = max(1, int(self.config.max_hop_depth))
        seed_list = sorted(seeds, key=str)
        for i, u in enumerate(seed_list):
            _, paths = nx.single_source_dijkstra(G, u, cutoff=k + 1)
            for v in seed_list[i + 1:]:
                if v in paths:
                    keep.update(paths[v])
        if len(keep) > self.config.max_subgraph_nodes:  # keep seeds + highest-degree context
            extra = sorted(keep - seeds, key=lambda x: G.degree(x), reverse=True)
            keep = seeds | set(extra[: max(0, self.config.max_subgraph_nodes - len(seeds))])

        sub_nodes = [n for n in nodes if n["id"] in keep]
        sub_edges = [e for e in edges if e["source"] in keep and e["target"] in keep]
        return sub_nodes, sub_edges, len(seeds)

    # ------------------------------------------------------------------ entry point
    def initiate_graph_retrieval(self, query_entities: List[str]) -> GraphRetrieverArtifact:
        try:
            nodes, edges, source = self.snapshot()
            sub_nodes, sub_edges, n_seeds = self.filter_k_hop_subgraph(nodes, edges, query_entities)
            write_json_atomic(self.config.subgraph_export_file_path, {
                "query_entities": query_entities, "snapshot_source": source, "seed_count": n_seeds,
                "node_count": len(sub_nodes), "edge_count": len(sub_edges),
                "nodes": sub_nodes, "edges": sub_edges,
            }, indent=2)
            logging.info(f"GT subgraph [{source}]: {len(sub_nodes)}/{len(nodes)} nodes, {len(sub_edges)}/{len(edges)} edges, seeds={n_seeds}")
            return GraphRetrieverArtifact(
                query_entities=query_entities, extracted_nodes=sub_nodes, extracted_edges=sub_edges,
                subgraph_file_path=self.config.subgraph_export_file_path,
                retrieval_status=bool(sub_nodes), seed_count=n_seeds,
            )
        except Exception as e:
            raise KGAuditException(e, sys)

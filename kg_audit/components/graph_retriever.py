import os
import re
import sys
import glob
import json
import networkx as nx
from scipy import sparse
from typing import List, Dict, Any, Tuple, Set
from neo4j import GraphDatabase, Driver

from kg_audit.entity.config_entity import GraphRetrieverConfig
from kg_audit.entity.artifact_entity import GraphRetrieverArtifact
from kg_audit.components.kg_builder import KGBuilder
from kg_audit.exceptions.exception import KGAuditException
from kg_audit.logging.logger import logging


# Generic domain-wide words that should NOT trigger broad single-word CONTAINS matches across all nodes
_BROAD_DOMAIN_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "over", "how",
    "via", "using", "based", "used", "uses", "network", "networks", "graph", "graphs",
    "node", "nodes", "edge", "edges", "embedding", "embeddings", "vector", "vectors",
    "space", "spaces", "model", "models", "method", "methods", "algorithm", "algorithms",
    "approach", "system", "feature", "features", "learning", "representation",
    "representations", "information", "large", "scale", "low", "dimensional",
    "data", "dataset", "datasets", "optimization", "objective", "function",
    "sampling", "strategy", "strategies", "search", "walk", "walks", "structure",
    "structural", "local", "global", "similarity", "neighborhood", "context"
}


class GraphRetriever:
    def __init__(self, config: GraphRetrieverConfig):
        try:
            self.config = config
            self.driver: Driver = None
        except Exception as e:
            raise KGAuditException(e, sys)

    def connect_neo4j(self):
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            self.driver.verify_connectivity()
        except Exception as e:
            raise KGAuditException(e, sys)

    def close(self):
        if self.driver:
            try:
                self.driver.close()
            except Exception:
                pass
            self.driver = None

    def _extract_search_terms(self, entities: List[str]) -> Tuple[List[str], List[str]]:
        """
        Separates entity queries into:
        1. exact_phrases: full lowercase entity names and derived acronyms (e.g., 'large-scale information network embedding', 'line', 'node2vec', 'breadth-first search', 'bfs')
        2. specific_tokens: distinctive non-generic tokens (e.g., 'node2vec', 'skip-gram', 'proximity', 'homophily', 'attention', 'convolutional')
        """
        exact_phrases: Set[str] = set()
        specific_tokens: Set[str] = set()

        for ent in entities:
            clean = str(ent).strip()
            if not clean:
                continue
            low = clean.lower()
            exact_phrases.add(low)

            init = KGBuilder._initials(clean)
            if 2 <= len(init) <= 8:
                exact_phrases.add(init.lower())
                specific_tokens.add(init.lower())

            if KGBuilder._is_short_token(clean):
                acr = KGBuilder._strip_plural_acronym(clean).lower()
                exact_phrases.add(acr)
                specific_tokens.add(acr)

            for token in re.sub(r"[^\w\s-]", " ", low).replace("-", " ").split():
                if len(token) >= 3 and token not in _BROAD_DOMAIN_STOPWORDS:
                    specific_tokens.add(token)

        return sorted(exact_phrases), sorted(specific_tokens)

    def _filter_k_hop_subgraph(
        self,
        all_nodes: List[Dict[str, Any]],
        all_edges: List[Dict[str, Any]],
        query_entities: List[str]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Finds the seed nodes in the KG that specifically match the query/response entities
        (via exact name, alias, acronym, or specific technical token) and extracts the
        induced k-hop neighborhood (k = config.max_hop_depth, default 2) so each query
        gets its own focused Ground-Truth Subgraph instead of the entire database.
        """
        if not all_nodes:
            return [], []

        exact_phrases, specific_tokens = self._extract_search_terms(query_entities)
        phrase_set = set(exact_phrases)
        token_set = set(specific_tokens)

        seed_node_ids: Set[Any] = set()
        for node in all_nodes:
            nid = node["id"]
            n_name = str(node.get("name", "")).strip()
            n_low = n_name.lower()
            aliases = [str(a).strip() for a in node.get("aliases", []) if a]
            all_names = [n_name] + aliases
            all_low = [s.lower() for s in all_names]

            # 1. Exact or Acronym Match against any query entity
            matched = False
            for cand_low in all_low:
                if cand_low in phrase_set:
                    matched = True
                    break
                for q_ent in query_entities:
                    if KGBuilder._is_acronym_match(cand_low, str(q_ent)):
                        matched = True
                        break
                if matched:
                    break

            # 2. Phrase containment (for phrases >= 4 chars) or specific technical token match
            if not matched:
                for cand_low in all_low:
                    if any(len(p) >= 4 and (p in cand_low or cand_low in p) for p in phrase_set):
                        matched = True
                        break
                    cand_tokens = set(re.sub(r"[^\w\s]", " ", cand_low).split())
                    if cand_tokens & token_set:
                        matched = True
                        break

            if matched:
                seed_node_ids.add(nid)

        # Build undirected graph for hop neighborhood traversal
        G = nx.Graph()
        for n in all_nodes:
            G.add_node(n["id"])
        for e in all_edges:
            if e["source"] in G and e["target"] in G:
                G.add_edge(e["source"], e["target"])

        if not seed_node_ids:
            # If no specific seed matched (e.g., 100% alien entities), return top high-degree core nodes as reference
            deg_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
            keep_ids = {nid for nid, _ in deg_sorted[:min(15, len(deg_sorted))]}
        else:
            keep_ids: Set[Any] = set(seed_node_ids)
            max_hops = max(1, int(getattr(self.config, "max_hop_depth", 2)))
            # Include 1-hop neighbors around seed nodes, plus 2-hop shortest paths connecting pairs of seed nodes
            for sid in seed_node_ids:
                if sid in G:
                    keep_ids.update(nx.single_source_shortest_path_length(G, sid, cutoff=1).keys())

            seed_list = [s for s in seed_node_ids if s in G]
            if max_hops >= 2 and len(seed_list) > 1:
                for i in range(len(seed_list)):
                    for j in range(i + 1, len(seed_list)):
                        u, v = seed_list[i], seed_list[j]
                        if nx.has_path(G, u, v):
                            sp = nx.shortest_path(G, u, v)
                            if len(sp) - 1 <= max_hops + 1:
                                keep_ids.update(sp)

        filtered_nodes = [n for n in all_nodes if n["id"] in keep_ids]
        filtered_edges = [
            e for e in all_edges
            if e["source"] in keep_ids and e["target"] in keep_ids
        ]
        return filtered_nodes, filtered_edges

    def fetch_subgraph_cypher(self, entities: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        try:
            if not self.driver:
                self.connect_neo4j()

            cypher_all = """
            MATCH (u:Entity)-[r:RELATED_TO]->(v:Entity)
            RETURN
                collect(DISTINCT {
                    id: elementId(u),
                    name: u.name,
                    type: coalesce(u.type, 'CONCEPT'),
                    labels: labels(u),
                    aliases: coalesce(u.aliases, []),
                    source_pdfs: coalesce(u.source_pdfs, []),
                    updated_at: coalesce(u.updated_at, '')
                }) +
                collect(DISTINCT {
                    id: elementId(v),
                    name: v.name,
                    type: coalesce(v.type, 'CONCEPT'),
                    labels: labels(v),
                    aliases: coalesce(v.aliases, []),
                    source_pdfs: coalesce(v.source_pdfs, []),
                    updated_at: coalesce(v.updated_at, '')
                }) AS nodes,
                collect(DISTINCT {
                    source: elementId(startNode(r)),
                    target: elementId(endNode(r)),
                    type: coalesce(r.type, type(r)),
                    weight: coalesce(r.weight, 0.9),
                    grounding_score: coalesce(r.grounding_score, 0.85),
                    evidence_quote: coalesce(r.evidence_quote, ''),
                    source_pdfs: coalesce(r.source_pdfs, []),
                    document_ids: coalesce(r.document_ids, []),
                    chunk_ids: coalesce(r.chunk_ids, []),
                    support_count: coalesce(r.support_count, 1)
                }) AS edges
            """

            with self.driver.session(database=self.config.neo4j_database) as session:
                record = session.run(cypher_all).single()
                if record and record["nodes"]:
                    unique_nodes = list({n["id"]: n for n in record["nodes"]}.values())
                    unique_edges = list({
                        f"{e['source']}_{e['target']}_{e['type']}": e for e in record["edges"]
                    }.values())
                    sub_nodes, sub_edges = self._filter_k_hop_subgraph(
                        unique_nodes, unique_edges, entities
                    )
                    logging.info(
                        f"Retrieved query-focused Neo4j subgraph: {len(sub_nodes)}/{len(unique_nodes)} nodes, "
                        f"{len(sub_edges)}/{len(unique_edges)} edges."
                    )
                    return sub_nodes, sub_edges

            return self._load_local_fallback_kg(entities)

        except Exception as e:
            logging.warning(f"Neo4j query warning ({e}). Using local extracted KG fallback.")
            return self._load_local_fallback_kg(entities)

    def _load_local_fallback_kg(self, entities: List[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        try:
            pattern = os.path.join("Artifacts", "*", "kg_builder", "extracted_knowledge_graph.json")
            files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
            if not files:
                return [], []
            with open(files[0], "r", encoding="utf-8") as f:
                data = json.load(f)
            nodes = data.get("nodes", [])
            name_to_id = {n["name"]: n["id"] for n in nodes}
            edges = []
            for e in data.get("edges", []):
                u_id = name_to_id.get(e.get("subject"))
                v_id = name_to_id.get(e.get("object"))
                if u_id is not None and v_id is not None:
                    edges.append({
                        "source": u_id,
                        "target": v_id,
                        "type": e.get("relation", "RELATED_TO"),
                        "weight": float(e.get("extraction_confidence", 0.9)),
                        "grounding_score": float(e.get("grounding_score", 0.85)),
                        "evidence_quote": e.get("evidence_quote", ""),
                        "source_pdfs": [e.get("source_name", "local_pdf")],
                        "document_ids": [e.get("document_id", "DOC_001")],
                        "chunk_ids": [e.get("chunk_id", "CH_001")],
                        "support_count": 1
                    })
            if entities:
                return self._filter_k_hop_subgraph(nodes, edges, entities)
            return nodes, edges
        except Exception:
            return [], []

    def build_sparse_matrices(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Tuple[nx.Graph, sparse.csr_matrix]:
        try:
            G = nx.Graph()
            node_id_map = {}

            for idx, node in enumerate(nodes):
                G.add_node(idx, original_id=node["id"], name=node["name"], labels=node.get("labels", []))
                node_id_map[node["id"]] = idx

            for edge in edges:
                u = node_id_map.get(edge["source"])
                v = node_id_map.get(edge["target"])
                if u is not None and v is not None:
                    weight = float(edge.get("weight", 1.0))
                    G.add_edge(u, v, weight=weight, type=edge.get("type", "RELATED_TO"))

            if G.number_of_nodes() == 0:
                return G, sparse.csr_matrix((0, 0))

            adj_sparse = nx.adjacency_matrix(G, weight="weight")
            return G, adj_sparse

        except Exception as e:
            raise KGAuditException(e, sys)

    def initiate_graph_retrieval(self, query_entities: List[str]) -> GraphRetrieverArtifact:
        try:
            nodes, edges = self.fetch_subgraph_cypher(query_entities)
            G, _ = self.build_sparse_matrices(nodes, edges)

            subgraph_payload = {
                "query_entities": query_entities,
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "nodes": nodes,
                "edges": edges
            }

            export_path = self.config.subgraph_export_file_path
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(subgraph_payload, f, indent=4)

            self.close()

            return GraphRetrieverArtifact(
                query_entities=query_entities,
                extracted_nodes=nodes,
                extracted_edges=edges,
                subgraph_file_path=export_path,
                retrieval_status=G.number_of_nodes() > 0
            )

        except Exception as e:
            self.close()
            raise KGAuditException(e, sys)
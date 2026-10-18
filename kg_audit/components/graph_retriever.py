import os
import sys
import glob
import json
import networkx as nx
from scipy import sparse
from typing import List, Dict, Any, Tuple
from neo4j import GraphDatabase, Driver

from kg_audit.entity.config_entity import GraphRetrieverConfig
from kg_audit.entity.artifact_entity import GraphRetrieverArtifact
from kg_audit.exceptions.exception import KGAuditException
from kg_audit.logging.logger import logging


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
            self.driver.close()

    def _extract_search_keywords(self, entities: List[str]) -> List[str]:
        """Splits entity phrases into clean lowercase search keywords & acronyms."""
        stop_words = {"the", "and", "for", "with", "from", "that", "this", "into", "over", "how"}
        keywords = set()
        for ent in entities:
            clean = str(ent).strip().lower()
            if clean:
                keywords.add(clean)
                for token in clean.replace("-", " ").replace("(", " ").replace(")", " ").split():
                    if len(token) >= 3 and token not in stop_words:
                        keywords.add(token)
        return list(keywords)

    def fetch_subgraph_cypher(self, entities: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        try:
            if not self.driver:
                self.connect_neo4j()

            keywords = self._extract_search_keywords(entities)
            logging.info(f"Querying Neo4j Cloud with search keywords: {keywords}")

            # 1. Case-insensitive match on Entity name, aliases, and keywords
            cypher_match = """
            MATCH (u:Entity)-[r:RELATED_TO]-(v:Entity)
            WHERE any(kw IN $keywords WHERE toLower(u.name) CONTAINS kw OR kw CONTAINS toLower(u.name))
               OR any(kw IN $keywords WHERE toLower(v.name) CONTAINS kw OR kw CONTAINS toLower(v.name))
               OR (u.aliases IS NOT NULL AND any(al IN u.aliases WHERE any(kw IN $keywords WHERE toLower(al) CONTAINS kw)))
               OR (v.aliases IS NOT NULL AND any(al IN v.aliases WHERE any(kw IN $keywords WHERE toLower(al) CONTAINS kw)))
            RETURN
                collect(DISTINCT {id: elementId(u), name: u.name, labels: labels(u), aliases: coalesce(u.aliases, [])}) +
                collect(DISTINCT {id: elementId(v), name: v.name, labels: labels(v), aliases: coalesce(v.aliases, [])}) AS nodes,
                collect(DISTINCT {
                    source: elementId(startNode(r)),
                    target: elementId(endNode(r)),
                    type: coalesce(r.type, type(r)),
                    weight: coalesce(r.weight, 1.0)
                }) AS edges
            """

            with self.driver.session(database=self.config.neo4j_database) as session:
                if keywords:
                    record = session.run(cypher_match, keywords=keywords).single()
                    if record and record["nodes"] and len(record["nodes"]) > 0:
                        unique_nodes = list({n["id"]: n for n in record["nodes"]}.values())
                        unique_edges = list({f"{e['source']}_{e['target']}_{e['type']}": e for e in record["edges"]}.values())
                        return unique_nodes, unique_edges

                # 2. Explicit Fallback: If specific keyword match returned 0 nodes, pull the core KG neighborhood
                logging.info("Keyword match returned 0 nodes; pulling core Neo4j KG subgraph...")
                broad_query = """
                MATCH (u:Entity)-[r:RELATED_TO]->(v:Entity)
                LIMIT 50
                RETURN
                    collect(DISTINCT {id: elementId(u), name: u.name, labels: labels(u), aliases: coalesce(u.aliases, [])}) +
                    collect(DISTINCT {id: elementId(v), name: v.name, labels: labels(v), aliases: coalesce(v.aliases, [])}) AS nodes,
                    collect(DISTINCT {
                        source: elementId(startNode(r)),
                        target: elementId(endNode(r)),
                        type: coalesce(r.type, type(r)),
                        weight: coalesce(r.weight, 1.0)
                    }) AS edges
                """
                res_broad = session.run(broad_query).single()
                if res_broad and res_broad["nodes"]:
                    unique_nodes = list({n["id"]: n for n in res_broad["nodes"]}.values())
                    unique_edges = list({f"{e['source']}_{e['target']}_{e['type']}": e for e in res_broad["edges"]}.values())
                    return unique_nodes, unique_edges

            return self._load_local_fallback_kg()

        except Exception as e:
            logging.warning(f"Neo4j query warning ({e}). Using local extracted KG fallback.")
            return self._load_local_fallback_kg()

    def _load_local_fallback_kg(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Reads the most recent local extracted_knowledge_graph.json if Neo4j returns empty."""
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
                        "weight": float(e.get("extraction_confidence", 0.9))
                    })
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
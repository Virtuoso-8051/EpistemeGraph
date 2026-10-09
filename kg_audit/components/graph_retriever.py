# Phase 2: Fetches subgraph from Neo4j
import os
import sys
import json
import networkx as nx
import numpy as np
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
            logging.info("Connecting to Neo4j instance...")
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            self.driver.verify_connectivity()
            logging.info("Neo4j connection established successfully.")
        except Exception as e:
            raise KGAuditException(e, sys)

    def close(self):
        if self.driver:
            self.driver.close()
            logging.info("Neo4j connection closed.")

    def fetch_subgraph_cypher(self, entities: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        try:
            if not self.driver:
                self.connect_neo4j()

            cypher_query = """
            MATCH (e) WHERE e.name IN $entities
            CALL apoc.path.subgraphAll(e, {
                maxLevel: $max_depth
            })
            YIELD nodes, relationships
            UNWIND nodes AS n
            UNWIND relationships AS r
            RETURN 
                collect(DISTINCT {id: id(n), name: coalesce(n.name, toString(id(n))), labels: labels(n)}) AS nodes,
                collect(DISTINCT {source: id(startNode(r)), target: id(endNode(r)), type: type(r), weight: coalesce(r.weight, 1.0)}) AS edges
            """

            with self.driver.session() as session:
                result = session.run(
                    cypher_query, 
                    entities=entities, 
                    max_depth=self.config.max_hop_depth
                )
                record = result.single()
                if record:
                    return record["nodes"], record["edges"]
                return [], []

        except Exception as e:
            logging.warning(f"Neo4j Cypher query failed or APOC missing. Falling back to basic match: {str(e)}")
            return self._fallback_match(entities)

    def _fallback_match(self, entities: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        try:
            cypher_fallback = """
            MATCH (u)-[r]-(v)
            WHERE u.name IN $entities OR v.name IN $entities
            RETURN 
                collect(DISTINCT {id: id(u), name: coalesce(u.name, toString(id(u))), labels: labels(u)}) +
                collect(DISTINCT {id: id(v), name: coalesce(v.name, toString(id(v))), labels: labels(v)}) AS nodes,
                collect(DISTINCT {source: id(startNode(r)), target: id(endNode(r)), type: type(r), weight: coalesce(r.weight, 1.0)}) AS edges
            """
            with self.driver.session() as session:
                result = session.run(cypher_fallback, entities=entities)
                record = result.single()
                if record:
                    unique_nodes = {n["id"]: n for n in record["nodes"]}.values()
                    unique_edges = {f"{e['source']}_{e['target']}_{e['type']}": e for e in record["edges"]}.values()
                    return list(unique_nodes), list(unique_edges)
                return [], []
        except Exception as e:
            raise KGAuditException(e, sys)

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

            adj_sparse = nx.adjacency_matrix(G, weight="weight")
            return G, adj_sparse

        except Exception as e:
            raise KGAuditException(e, sys)

    def initiate_graph_retrieval(self, query_entities: List[str]) -> GraphRetrieverArtifact:
        try:
            logging.info(f"Initiating graph retrieval for target entities: {query_entities}")
            
            nodes, edges = self.fetch_subgraph_cypher(query_entities)
            G, adj_matrix = self.build_sparse_matrices(nodes, edges)

            subgraph_payload = {
                "query_entities": query_entities,
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "nodes": nodes,
                "edges": edges
            }

            export_path = self.config.subgraph_export_file_path
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, "w") as f:
                json.dump(subgraph_payload, f, indent=4)

            logging.info(f"Subgraph exported successfully to {export_path}. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

            self.close()

            return GraphRetrieverArtifact(
                query_entities=query_entities,
                extracted_nodes=nodes,
                extracted_edges=edges,
                subgraph_file_path=export_path,
                retrieval_status=True
            )

        except Exception as e:
            self.close()
            raise KGAuditException(e, sys)
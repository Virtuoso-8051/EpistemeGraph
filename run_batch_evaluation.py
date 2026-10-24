import os
import sys
import csv
from typing import List

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from kg_audit.entity.config_entity import (
    KGAuditPipelineConfig,
    LLMInterceptorConfig,
    GraphRetrieverConfig,
    TruthEvaluatorConfig
)
from kg_audit.components.llm_interceptor import LLMInterceptor
from kg_audit.components.graph_retriever import GraphRetriever
from kg_audit.components.truth_evaluator import TruthEvaluator
from kg_audit.logging.logger import logging


def run_batch_evaluation(query_file_path: str = "query.txt", output_csv_path: str = "output.csv"):
    if not os.path.exists(query_file_path):
        print(f"❌ Error: Query file '{query_file_path}' not found. Please create it and add queries (one per line).")
        return

    # Read queries from query.txt (skipping empty lines)
    with open(query_file_path, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    if not queries:
        print(f"⚠️ Warning: No queries found in '{query_file_path}'.")
        return

    print(f"🚀 Loaded {len(queries)} queries from {query_file_path}. Initializing Phase 2 & 3 components...")

    # Initialize configs and pipeline components
    pipeline_config = KGAuditPipelineConfig()
    interceptor_config = LLMInterceptorConfig(pipeline_config)
    retriever_config = GraphRetrieverConfig(pipeline_config)
    evaluator_config = TruthEvaluatorConfig(pipeline_config)

    interceptor = LLMInterceptor(interceptor_config)
    retriever = GraphRetriever(retriever_config)
    evaluator = TruthEvaluator(evaluator_config)

    # Define CSV headers matching the 8D feature vector + metadata & results
    csv_headers = [
        "query",
        "e_sheaf",                  # 1. Cellular Sheaf Coboundary Residual
        "delta_k",                  # 2. Bounded Forman-Ricci Curvature Distortion
        "d_spec",                   # 3. Induced Spectral Laplacian Divergence
        "bayesian_trust",           # 4. Bayesian Source Trust
        "freshness",                # 5. Temporal Freshness Decay
        "independence",             # 6. Source Independence
        "conflict_ratio",           # 7. Active Conflict / Unverified Ratio
        "llm_conf",                 # 8. LLM Self-Reported Confidence (P_LLM)
        "query_belief_score",       # S_Query (Model prediction probability)
        "phase1_kg_fidelity",       # S_KG (Ground truth graph fidelity)
        "final_compound_trust",     # S_Final
        "decision_status",          # VERIFIED / ABSTAIN / HALLUCINATED
        "is_hallucinated"           # True / False
    ]

    os.makedirs(os.path.dirname(output_csv_path) if os.path.dirname(output_csv_path) else ".", exist_ok=True)

    print(f"📊 Running batch execution and saving results to '{output_csv_path}'...")
    
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_headers)

        for idx, q in enumerate(queries, start=1):
            print(f"[{idx}/{len(queries)}] Processing query: '{q[:60]}...'")
            try:
                # --- Phase 2: Intercept Query & Capture ChatGPT Answer + G' ---
                interceptor_art, response_graph_payload = interceptor.intercept_and_generate(user_query=q)

                # --- Phase 3A: Fetch Active Subgraph G from Neo4j Cloud ---
                target_entities = response_graph_payload.get("entities", [])
                retriever_art = retriever.initiate_graph_retrieval(query_entities=target_entities)

                # --- Phase 3B: Compute CASSE Tensor, 8D Features & Calibrated Trust ---
                metadata = {
                    "llm_extraction_confidence": interceptor_art.llm_self_reported_accuracy
                }
                eval_art = evaluator.evaluate_truth(
                    retriever_artifact=retriever_art,
                    response_nodes=response_graph_payload.get("nodes", []),
                    response_edges=response_graph_payload.get("edges", []),
                    metadata=metadata
                )

                # Unpack 8D Feature vector [e_sheaf, delta_k, d_spec, bayesian_trust, freshness, independence, conflict_ratio, llm_conf]
                f_vec = eval_art.feature_vector

                row = [
                    q,
                    round(f_vec[0], 6),
                    round(f_vec[1], 6),
                    round(f_vec[2], 6),
                    round(f_vec[3], 6),
                    round(f_vec[4], 6),
                    round(f_vec[5], 6),
                    round(f_vec[6], 6),
                    round(f_vec[7], 6),
                    round(eval_art.query_belief_score, 6),
                    round(eval_art.phase1_kg_fidelity_score, 6),
                    round(eval_art.final_accuracy_score, 6),
                    eval_art.decision_status,
                    eval_art.is_hallucinated
                ]
                writer.writerow(row)
                csv_file.flush() # Ensure it writes immediately per iteration

            except Exception as e:
                logging.error(f"Failed processing query '{q}': {e}")
                print(f"⚠️ Error on query '{q}': {e}. Skipping row.")

    print(f"\n🎉 Batch evaluation complete! Successfully processed {len(queries)} queries. Results stored in '{output_csv_path}'.")


if __name__ == "__main__":
    # Example input/output paths (customize as needed)
    QUERY_FILE = "query.txt"
    OUTPUT_CSV = "output.csv"
    
    run_batch_evaluation(query_file_path=QUERY_FILE, output_csv_path=OUTPUT_CSV)

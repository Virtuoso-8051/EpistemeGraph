import os
import sys
import mlflow
import dagshub

from kg_audit.entity.config_entity import (
    KGAuditPipelineConfig,
    LLMInterceptorConfig,
    GraphRetrieverConfig,
    TruthEvaluatorConfig
)
from kg_audit.entity.artifact_entity import TruthEvaluatorArtifact
from kg_audit.components.llm_interceptor import LLMInterceptor
from kg_audit.components.graph_retriever import GraphRetriever
from kg_audit.components.truth_evaluator import TruthEvaluator
from kg_audit.exceptions.exception import KGAuditException
from kg_audit.logging.logger import logging

class EvaluationPipeline:
    def __init__(self):
        try:
            self.pipeline_config = KGAuditPipelineConfig()
            self.interceptor_config = LLMInterceptorConfig(self.pipeline_config)
            self.retriever_config = GraphRetrieverConfig(self.pipeline_config)
            self.evaluator_config = TruthEvaluatorConfig(self.pipeline_config)

            # Initialize MLflow tracking via DagsHub / Remote backend
            repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "Anurag-Raj")
            repo_name = os.getenv("DAGSHUB_REPO_NAME", "KG-Audit-Research")
            
            if os.getenv("DAGSHUB_TOKEN"):
                dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            else:
                mlflow.set_tracking_uri(f"file:///{os.path.abspath('mlruns')}")

            mlflow.set_experiment("CASSE_LLM_Verification_Research")

        except Exception as e:
            raise KGAuditException(e, sys)

    def run_pipeline(self, user_query: str, source_documents: list = None) -> TruthEvaluatorArtifact:
        try:
            logging.info("Starting Master KGAudit Evaluation Pipeline...")

            with mlflow.start_run(run_name=f"run_{self.pipeline_config.timestamp}"):
                # Log Config Parameters
                mlflow.log_param("llm_model", self.interceptor_config.llm_model_name)
                mlflow.log_param("temperature", self.interceptor_config.temperature)
                mlflow.log_param("max_hop_depth", self.retriever_config.max_hop_depth)
                mlflow.log_param("weight_sheaf", self.evaluator_config.entailment_weight)
                mlflow.log_param("weight_curvature", self.evaluator_config.triple_match_weight)
                mlflow.log_param("weight_spectral", self.evaluator_config.topological_weight)

                # Step 1: Intercept Query & Generate Response Graph G'
                logging.info("--- Step 1: Query Interception & Entity Extraction ---")
                interceptor = LLMInterceptor(self.interceptor_config)
                interceptor_artifact, response_graph_payload = interceptor.intercept_and_generate(
                    user_query=user_query,
                    source_documents=source_documents
                )
                mlflow.log_artifact(interceptor_artifact.captured_payload_path)

                # Step 2: Fetch Ground-Truth Subgraph G from Neo4j
                logging.info("--- Step 2: Neo4j Ground-Truth Subgraph Retrieval ---")
                retriever = GraphRetriever(self.retriever_config)
                target_entities = response_graph_payload.get("entities", [])
                retriever_artifact = retriever.initiate_graph_retrieval(query_entities=target_entities)
                mlflow.log_artifact(retriever_artifact.subgraph_file_path)

                # Step 3: Compute CASSE Mathematical Verification Engine
                logging.info("--- Step 3: CASSE Sheaf-Spectral Evaluation ---")
                evaluator = TruthEvaluator(self.evaluator_config)
                evaluator_artifact = evaluator.evaluate_truth(
                    retriever_artifact=retriever_artifact,
                    response_nodes=response_graph_payload.get("nodes", []),
                    response_edges=response_graph_payload.get("edges", [])
                )

                # Log CASSE Mathematical Metrics to MLflow
                mlflow.log_metric("sheaf_laplacian_energy", evaluator_artifact.semantic_entailment_score)
                mlflow.log_metric("forman_ricci_distortion", evaluator_artifact.triple_match_score)
                mlflow.log_metric("spectral_laplacian_divergence", evaluator_artifact.topological_similarity_score)
                mlflow.log_metric("final_accuracy_score", evaluator_artifact.final_accuracy_score)
                mlflow.log_metric("is_hallucinated", int(evaluator_artifact.is_hallucinated))
                
                mlflow.log_artifact(evaluator_artifact.evaluation_report_path)

                logging.info(f"Pipeline finished successfully. Accuracy Score: {evaluator_artifact.final_accuracy_score:.4f}")
                return evaluator_artifact

        except Exception as e:
            raise KGAuditException(e, sys)

if __name__ == "__main__":
    test_query = "What is the relationship between Entity A and Entity B?"
    pipeline = EvaluationPipeline()
    result = pipeline.run_pipeline(user_query=test_query)
    print("Evaluation Result:", result)
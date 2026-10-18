import os
import sys
from typing import List, Tuple
import mlflow
import dagshub

from kg_audit.entity.config_entity import (
    KGAuditPipelineConfig,
    KGBuilderConfig,
    LLMInterceptorConfig,
    GraphRetrieverConfig,
    TruthEvaluatorConfig,
    ModelTrainerConfig
)
from kg_audit.entity.artifact_entity import (
    KGBuilderArtifact,
    LLMInterceptorArtifact,
    GraphRetrieverArtifact,
    TruthEvaluatorArtifact
)
from kg_audit.components.kg_builder import KGBuilder
from kg_audit.components.llm_interceptor import LLMInterceptor
from kg_audit.components.graph_retriever import GraphRetriever
from kg_audit.components.truth_evaluator import TruthEvaluator
from kg_audit.components.model_trainer import ModelTrainer
from kg_audit.exceptions.exception import KGAuditException
from kg_audit.logging.logger import logging


class EvaluationPipeline:
    def __init__(self):
        try:
            self.pipeline_config = KGAuditPipelineConfig()
            self.kg_builder_config = KGBuilderConfig(self.pipeline_config)
            self.interceptor_config = LLMInterceptorConfig(self.pipeline_config)
            self.retriever_config = GraphRetrieverConfig(self.pipeline_config)
            self.evaluator_config = TruthEvaluatorConfig(self.pipeline_config)
            self.trainer_config = ModelTrainerConfig(self.pipeline_config)

            repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "Virtuoso-8051")
            repo_name = os.getenv("DAGSHUB_REPO_NAME", "EpistemeGraph")

            if os.getenv("DAGSHUB_TOKEN"):
                dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            else:
                os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
                mlflow.set_tracking_uri("sqlite:///mlflow.db")

            mlflow.set_experiment("EpistemeGraph_CASSE_Verification")

            # Ensure calibrated model.pkl exists before query evaluation
            if not os.path.exists(self.trainer_config.final_model_file_path):
                logging.info("No pre-trained model.pkl found. Running ModelTrainer bootstrap...")
                trainer = ModelTrainer(self.trainer_config)
                trainer.initiate_model_trainer()

        except Exception as e:
            raise KGAuditException(e, sys)

    def run_phase1_kg_construction(self, file_paths: List[str] = None) -> KGBuilderArtifact:
        """
        Executes Phase 1: Parses PDFs, extracts KG via OpenAI, computes 4-Pillar S_KG,
        saves local state, and pushes to Neo4j AuraDB Cloud.
        """
        try:
            with mlflow.start_run(run_name=f"phase1_kg_build_{self.pipeline_config.timestamp}"):
                builder = KGBuilder(self.kg_builder_config)
                if file_paths:
                    kg_artifact = builder.initiate_kg_construction(file_paths)
                else:
                    kg_artifact = builder.initiate_kg_from_directory()

                mlflow.log_metric("phase1_overall_kg_accuracy_S_KG", kg_artifact.overall_kg_accuracy_score)
                mlflow.log_metric("phase1_provenance_grounding", kg_artifact.provenance_grounding_score)
                mlflow.log_metric("phase1_semantic_coverage", kg_artifact.semantic_coverage_score)
                mlflow.log_metric("phase1_topological_coherence", kg_artifact.topological_coherence_score)
                mlflow.log_metric("phase1_internal_consistency", kg_artifact.internal_consistency_score)
                mlflow.log_metric("verified_triplets_pushed", kg_artifact.verified_triplets_pushed)
                mlflow.log_artifact(kg_artifact.extracted_kg_file_path)

                return kg_artifact
        except Exception as e:
            raise KGAuditException(e, sys)

    def run_pipeline(
        self, user_query: str, source_documents: List[str] = None
    ) -> Tuple[LLMInterceptorArtifact, GraphRetrieverArtifact, TruthEvaluatorArtifact]:
        """
        Executes Phase 2 (Interception + LLM Self-Reported Accuracy) and
        Phase 3 (Neo4j Subgraph Retrieval + CASSE + Calibrated Compound Scoring).
        """
        try:
            logging.info("Starting Master EpistemeGraph Verification Pipeline...")

            with mlflow.start_run(run_name=f"verify_{self.pipeline_config.timestamp}"):
                mlflow.log_param("llm_model", self.interceptor_config.llm_model_name)

                # Phase 2: Intercept Query & Capture ChatGPT Answer + Self-Reported Accuracy
                interceptor = LLMInterceptor(self.interceptor_config)
                interceptor_artifact, response_graph_payload = interceptor.intercept_and_generate(
                    user_query=user_query,
                    source_documents=source_documents
                )
                mlflow.log_artifact(interceptor_artifact.captured_payload_path)

                # Phase 3A: Fetch Ground-Truth Subgraph from Neo4j AuraDB Cloud
                retriever = GraphRetriever(self.retriever_config)
                target_entities = response_graph_payload.get("entities", [])
                retriever_artifact = retriever.initiate_graph_retrieval(query_entities=target_entities)
                mlflow.log_artifact(retriever_artifact.subgraph_file_path)

                # Phase 3B: Compute CASSE + Pass LLM's Self-Reported Accuracy into Metadata
                evaluator = TruthEvaluator(self.evaluator_config)
                metadata = {
                    "llm_extraction_confidence": interceptor_artifact.llm_self_reported_accuracy
                }
                evaluator_artifact = evaluator.evaluate_truth(
                    retriever_artifact=retriever_artifact,
                    response_nodes=response_graph_payload.get("nodes", []),
                    response_edges=response_graph_payload.get("edges", []),
                    metadata=metadata
                )

                # Phase 4: MLOps Continuous Telemetry Logging & Self-Improvement Loop
                trainer = ModelTrainer(self.trainer_config)
                live_count = trainer.append_live_telemetry(
                    feature_vector=evaluator_artifact.feature_vector,
                    query_text=user_query
                )
                # Automatically retrain & evaluate Challenger vs. Champion every 3 live queries (or if metrics file missing)
                if live_count % 3 == 0 or not os.path.exists(trainer.production_metrics_path):
                    trainer_art = trainer.initiate_model_trainer()
                    mlflow.log_metric("champion_model_roc_auc", trainer_art.roc_auc_score)
                    mlflow.log_metric("champion_model_brier_loss", trainer_art.brier_score)

                # Log Metrics to MLflow
                mlflow.log_metric("llm_self_reported_accuracy", interceptor_artifact.llm_self_reported_accuracy)
                mlflow.log_metric("sheaf_laplacian_energy", evaluator_artifact.semantic_entailment_score)
                mlflow.log_metric("forman_ricci_distortion", evaluator_artifact.triple_match_score)
                mlflow.log_metric("spectral_laplacian_divergence", evaluator_artifact.topological_similarity_score)
                mlflow.log_metric("query_belief_score_S_Query", evaluator_artifact.query_belief_score)
                mlflow.log_metric("phase1_kg_fidelity_S_KG", evaluator_artifact.phase1_kg_fidelity_score)
                mlflow.log_metric("final_compound_accuracy_score", evaluator_artifact.final_accuracy_score)
                mlflow.log_metric("is_hallucinated", int(evaluator_artifact.is_hallucinated))
                mlflow.log_artifact(evaluator_artifact.evaluation_report_path)

                return interceptor_artifact, retriever_artifact, evaluator_artifact

        except Exception as e:
            raise KGAuditException(e, sys)
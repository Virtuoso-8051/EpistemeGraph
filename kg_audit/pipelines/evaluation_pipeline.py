"""
Master orchestration for EpistemeGraph.

  run_phase1_kg_construction : incremental PDF -> KG ingestion (Phase 1)
  run_pipeline               : query -> answer -> G' -> GT subgraph -> x in R^8 -> decision (Phases 2-3)
                               + persistent query history (UI auto-reload) + unlabelled telemetry
  retrain                    : Phase 4 tournament on the labelled feature store

No model is bootstrapped from synthetic data any more: until labelled responses exist, the
evaluator returns the 8 features with decision 'UNCALIBRATED'.
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import mlflow

from kg_audit.components.graph_retriever import GraphRetriever
from kg_audit.components.kg_builder import KGBuilder
from kg_audit.components.llm_interceptor import LLMInterceptor
from kg_audit.components.model_trainer import ModelTrainer
from kg_audit.components.truth_evaluator import TruthEvaluator
from kg_audit.entity.artifact_entity import (
    GraphRetrieverArtifact,
    KGBuilderArtifact,
    LLMInterceptorArtifact,
    ModelTrainerArtifact,
    TruthEvaluatorArtifact,
)
from kg_audit.entity.config_entity import (
    GraphRetrieverConfig,
    KGAuditPipelineConfig,
    KGBuilderConfig,
    LLMInterceptorConfig,
    ModelTrainerConfig,
    TruthEvaluatorConfig,
)
from kg_audit.exceptions.exception import KGAuditException
from kg_audit.utils.api_errors import FatalAPIError
from kg_audit.logging.logger import logging
from kg_audit.utils.main_utils import append_jsonl, limit_cpu_threads

_MLFLOW_READY = False


def _init_mlflow() -> bool:
    global _MLFLOW_READY
    if _MLFLOW_READY:
        return True
    try:
        if os.getenv("DAGSHUB_TOKEN"):
            import dagshub
            dagshub.init(repo_owner=os.getenv("DAGSHUB_REPO_OWNER", ""), repo_name=os.getenv("DAGSHUB_REPO_NAME", "EpistemeGraph"), mlflow=True)
        else:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("EpistemeGraph_CASSE_v2")
        _MLFLOW_READY = True
    except Exception as e:
        logging.warning(f"MLflow unavailable: {e}")
    return _MLFLOW_READY


class EvaluationPipeline:
    def __init__(self, use_mlflow: bool = True, workspace_id: Optional[str] = None):
        try:
            limit_cpu_threads()
            self.pipeline_config = KGAuditPipelineConfig(workspace_id=workspace_id)
            self.workspace_id = self.pipeline_config.workspace_id
            self.kg_builder_config = KGBuilderConfig(self.pipeline_config)
            self.interceptor_config = LLMInterceptorConfig(self.pipeline_config)
            self.retriever_config = GraphRetrieverConfig(self.pipeline_config)
            self.evaluator_config = TruthEvaluatorConfig(self.pipeline_config)
            self.trainer_config = ModelTrainerConfig(self.pipeline_config)
            self.use_mlflow = use_mlflow and _init_mlflow()
            self._interceptor: Optional[LLMInterceptor] = None
            self._evaluator: Optional[TruthEvaluator] = None
        except Exception as e:
            raise KGAuditException(e, sys)

    # lazily constructed, re-used across queries (no per-query model / corpus reloads)
    @property
    def interceptor(self) -> LLMInterceptor:
        if self._interceptor is None:
            self._interceptor = LLMInterceptor(self.interceptor_config)
        return self._interceptor

    @property
    def evaluator(self) -> TruthEvaluator:
        if self._evaluator is None:
            self._evaluator = TruthEvaluator(self.evaluator_config)
        return self._evaluator

    def _fresh_artifact_paths(self, artifact_dir: Optional[str] = None) -> None:
        """Each query gets its own artifact folder (no overwriting between queries)."""
        base = artifact_dir or os.path.join(self.pipeline_config.workspace_dir, "artifacts",
                                            datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f"))
        self.interceptor_config.captured_payload_file_path = os.path.join(base, "llm_interceptor", "interception_payload.json")
        self.retriever_config.subgraph_export_file_path = os.path.join(base, "graph_retriever", "extracted_subgraph.json")
        self.evaluator_config.evaluation_report_file_path = os.path.join(base, "truth_evaluator", "accuracy_report.json")

    # ------------------------------------------------------------------ Phase 1
    def run_phase1_kg_construction(self, file_paths: List[str] = None, reset: bool = False,
                                   progress_callback=None) -> KGBuilderArtifact:
        try:
            builder = KGBuilder(self.kg_builder_config)
            art = (builder.initiate_kg_construction(file_paths, progress_callback=progress_callback, reset=reset)
                   if file_paths else builder.initiate_kg_from_directory())
            GraphRetriever.invalidate_snapshot()
            if self.use_mlflow:
                with mlflow.start_run(run_name=f"phase1_{self.pipeline_config.timestamp}"):
                    mlflow.log_metrics({
                        "S_KG": art.overall_kg_accuracy_score, "phi_prov": art.provenance_grounding_score,
                        "phi_cov": art.semantic_coverage_score, "phi_topo": art.topological_coherence_score,
                        "phi_cons": art.internal_consistency_score, "verified_triples": art.verified_triplets_pushed,
                    })
            return art
        except Exception as e:
            raise KGAuditException(e, sys)

    # ------------------------------------------------------------------ Phases 2 + 3
    def run_pipeline(
        self,
        user_query: str,
        source_documents: Optional[List[str]] = None,
        mode: str = "closed_book",
        log_history: bool = True,
        log_telemetry: bool = True,
        precomputed_answer: Optional[Dict[str, Any]] = None,
        artifact_dir: Optional[str] = None,
    ) -> Tuple[LLMInterceptorArtifact, GraphRetrieverArtifact, TruthEvaluatorArtifact]:
        try:
            self._fresh_artifact_paths(artifact_dir)
            inter_art, resp = self.interceptor.intercept_and_generate(
                user_query, source_documents=source_documents, mode=mode, precomputed_answer=precomputed_answer)

            retr_art = GraphRetriever(self.retriever_config).initiate_graph_retrieval(resp.get("entities", []))
            eval_art = self.evaluator.evaluate_truth(retr_art, resp.get("nodes", []), resp.get("edges", []), resp.get("metadata", {}))

            if log_telemetry:
                ModelTrainer(self.trainer_config).append_live_telemetry(
                    eval_art.features, eval_art.local_kg_fidelity_score, eval_art.decision_status,
                    eval_art.final_accuracy_score, user_query)
            if log_history:
                append_jsonl(self.interceptor_config.history_path, self.history_record(inter_art, retr_art, eval_art, resp))
            if self.use_mlflow:
                with mlflow.start_run(run_name=f"verify_{self.pipeline_config.timestamp}"):
                    mlflow.log_params({"llm_model": self.interceptor_config.llm_model_name, "mode": mode})
                    mlflow.log_metrics({k: float(v) for k, v in eval_art.features.items()})
                    if eval_art.final_accuracy_score is not None:
                        mlflow.log_metric("S_final", eval_art.final_accuracy_score)
            return inter_art, retr_art, eval_art
        except FatalAPIError:
            raise
        except Exception as e:
            raise KGAuditException(e, sys)

    @staticmethod
    def history_record(inter_art, retr_art, eval_art, resp) -> Dict[str, Any]:
        """Everything the UI needs to re-render a past audit without any recomputation."""
        return {
            "id": uuid.uuid4().hex[:12],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": inter_art.user_query, "mode": inter_art.mode,
            "answer": inter_art.llm_response,
            "llm_confidence": inter_art.llm_self_reported_accuracy,
            "llm_confidence_reasoning": inter_art.llm_confidence_reasoning,
            "triples": resp.get("triples", []),
            "gt_nodes": len(retr_art.extracted_nodes), "gt_edges": len(retr_art.extracted_edges),
            "features": eval_art.features,
            "S_query": eval_art.query_belief_score, "S_final": eval_art.final_accuracy_score,
            "S_KG": eval_art.phase1_kg_fidelity_score, "S_KG_local": eval_art.local_kg_fidelity_score,
            "decision": eval_art.decision_status, "prediction_set": eval_art.conformal_prediction_set,
            # "how much": calibrated probability AND claim-level extent
            "hallucination_probability": eval_art.diagnostics.get("hallucination_probability"),
            "hallucination_extent": eval_art.diagnostics.get("hallucination_extent"),
            "unsupported_claims": eval_art.diagnostics.get("unsupported_claims"),
            "asserted_claims": eval_art.diagnostics.get("asserted_claims"),
            "brain": eval_art.diagnostics.get("brain"),
            "edge_audit_details": eval_art.edge_audit_details, "diagnostics": eval_art.diagnostics,
            "report_path": eval_art.evaluation_report_path,
        }

    # ------------------------------------------------------------------ Phase 4
    def retrain(self) -> ModelTrainerArtifact:
        art = ModelTrainer(self.trainer_config).initiate_model_trainer()
        if self.use_mlflow:
            with mlflow.start_run(run_name=f"phase4_{self.pipeline_config.timestamp}"):
                mlflow.log_param("champion", art.best_model_name)
                mlflow.log_metrics({"test_auc": art.roc_auc_score, "test_log_loss": art.log_loss,
                                    "test_brier": art.brier_score, "test_ece": art.expected_calibration_error,
                                    "promoted": int(art.is_model_accepted)})
        return art

import os
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

from kg_audit.utils.llm import model_for
from kg_audit.utils import workspace as ws

# Automatically load .env from project root
load_dotenv()


class KGAuditPipelineConfig:
    """
    All state lives in the active corpus workspace (final_model/workspaces/<id>/).
    Only the API-response caches are global, so identical LLM calls are free across workspaces.
    """

    def __init__(self, timestamp=None, workspace_id: Optional[str] = None):
        if timestamp is None:
            timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.workspace_id: str = workspace_id or ws.active_workspace_id()
        self.workspace_dir: str = ws.workspace_dir(self.workspace_id)
        self.kg_id: str = ws.kg_id(self.workspace_id)
        self.pipeline_name: str = "EpistemeGraph_Research_Pipeline"
        self.artifact_dir: str = os.path.join(self.workspace_dir, "artifacts", timestamp_str)
        self.model_dir: str = self.workspace_dir
        self.data_dir: str = self.workspace_dir
        self.data_uploads_dir: str = os.path.join(self.workspace_dir, "uploads")
        self.cache_dir: str = os.path.join(self.workspace_dir, "cache")
        # global, content-addressed API caches ($0 re-runs in every workspace)
        self.llm_cache_dir: str = os.path.join("Artifacts", "llm_cache")
        self.query_cache_dir: str = os.path.join("Artifacts", "query_cache")
        self.timestamp: str = timestamp_str


def _neo4j_env(obj) -> None:
    obj.neo4j_uri = os.getenv("NEO4J_URI", "neo4j+s://localhost:7687")
    obj.neo4j_user = os.getenv("NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j"))
    obj.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    obj.neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")


class KGBuilderConfig:
    def __init__(self, pipeline_config: KGAuditPipelineConfig):
        self.kg_id: str = pipeline_config.kg_id
        self.builder_dir: str = os.path.join(pipeline_config.artifact_dir, "kg_builder")
        self.extracted_kg_file_path: str = os.path.join(self.builder_dir, "extracted_knowledge_graph.json")
        self.kg_fidelity_state_path: str = os.path.join(pipeline_config.model_dir, "kg_fidelity_state.json")
        self.latest_chunks_path: str = os.path.join(pipeline_config.model_dir, "latest_chunks.json")
        self.corpus_registry_path: str = os.path.join(pipeline_config.model_dir, "corpus_registry.json")
        self.kg_store_path: str = os.path.join(pipeline_config.model_dir, "kg_store.json")
        self.data_uploads_dir: str = pipeline_config.data_uploads_dir
        self.llm_cache_dir: str = pipeline_config.llm_cache_dir
        _neo4j_env(self)
        self.llm_model_name: str = model_for("kg")
        self.chunk_size: int = 2200
        self.chunk_overlap: int = 200
        self.max_chunks_per_doc: Optional[int] = None   # None = every chunk of every document
        self.extraction_workers: int = 5
        self.entity_sim_threshold: float = 0.85
        # Reset is SCOPED to this workspace's kg_id; other corpora in Neo4j are never touched.
        self.clear_neo4j_on_build: bool = False


class LLMInterceptorConfig:
    def __init__(self, pipeline_config: KGAuditPipelineConfig):
        self.interceptor_dir: str = os.path.join(pipeline_config.artifact_dir, "llm_interceptor")
        self.captured_payload_file_path: str = os.path.join(self.interceptor_dir, "interception_payload.json")
        self.latest_chunks_path: str = os.path.join(pipeline_config.model_dir, "latest_chunks.json")
        self.kg_store_path: str = os.path.join(pipeline_config.model_dir, "kg_store.json")
        self.cache_dir: str = pipeline_config.cache_dir
        self.query_cache_dir: str = pipeline_config.query_cache_dir
        self.history_path: str = os.path.join(pipeline_config.data_dir, "query_history.jsonl")
        self.llm_model_name: str = model_for("answer")        # the model under audit
        self.extractor_model_name: str = model_for("extract")
        self.temperature: float = 0.0
        self.max_entity_hints: int = 25
        self.request_timeout_s: float = 60.0


class GraphRetrieverConfig:
    def __init__(self, pipeline_config: KGAuditPipelineConfig):
        self.kg_id: str = pipeline_config.kg_id
        self.retriever_dir: str = os.path.join(pipeline_config.artifact_dir, "graph_retriever")
        self.subgraph_export_file_path: str = os.path.join(self.retriever_dir, "extracted_subgraph.json")
        self.kg_store_path: str = os.path.join(pipeline_config.model_dir, "kg_store.json")
        _neo4j_env(self)
        self.max_hop_depth: int = 2
        self.max_subgraph_nodes: int = 400
        # one in-memory KG snapshot per (kg_id, local store version) for this many seconds
        self.snapshot_ttl_s: int = 600


class TruthEvaluatorConfig:
    def __init__(self, pipeline_config: KGAuditPipelineConfig):
        self.workspace_id: str = pipeline_config.workspace_id
        self.evaluator_dir: str = os.path.join(pipeline_config.artifact_dir, "truth_evaluator")
        self.evaluation_report_file_path: str = os.path.join(self.evaluator_dir, "accuracy_report.json")
        self.trained_model_path: str = os.path.join(pipeline_config.model_dir, "model.pkl")
        self.production_metrics_path: str = os.path.join(pipeline_config.model_dir, "production_model_metrics.json")
        self.kg_fidelity_state_path: str = os.path.join(pipeline_config.model_dir, "kg_fidelity_state.json")
        self.latest_chunks_path: str = os.path.join(pipeline_config.model_dir, "latest_chunks.json")
        self.cache_dir: str = pipeline_config.cache_dir
        self.embedding_model_name: str = "all-MiniLM-L6-v2"
        # Same value as KGBuilderConfig.entity_sim_threshold: one definition of "same entity".
        self.entity_align_threshold: float = 0.85
        self.max_hop_depth: int = 2
        # If this corpus has no calibrated brain yet, borrow the most recent one from another
        # workspace (flagged as 'transferred': the conformal guarantee is not certified here).
        self.allow_brain_transfer: bool = os.getenv("EPISTEME_BRAIN_TRANSFER", "1") == "1"


class ModelTrainerConfig:
    def __init__(self, pipeline_config: KGAuditPipelineConfig):
        self.model_trainer_dir: str = os.path.join(pipeline_config.artifact_dir, "model_trainer")
        self.trained_model_file_path: str = os.path.join(self.model_trainer_dir, "model.pkl")
        self.final_model_file_path: str = os.path.join(pipeline_config.model_dir, "model.pkl")
        self.metrics_report_file_path: str = os.path.join(self.model_trainer_dir, "calibration_metrics.json")
        self.production_metrics_path: str = os.path.join(pipeline_config.model_dir, "production_model_metrics.json")
        self.labelled_store_path: str = os.path.join(pipeline_config.data_dir, "labelled_features.csv")
        self.telemetry_path: str = os.path.join(pipeline_config.data_dir, "live_telemetry.csv")
        # Statistical knobs (guarantees, not heuristics)
        self.conformal_alpha: float = float(os.getenv("EPISTEME_ALPHA", "0.10"))
        self.test_fraction: float = 0.20
        self.calibration_fraction: float = 0.20
        self.cv_folds: int = 5
        self.min_labelled_samples: int = 60
        self.random_state: int = 42
        self.n_jobs: int = int(os.getenv("EPISTEME_CPU_THREADS", "2"))

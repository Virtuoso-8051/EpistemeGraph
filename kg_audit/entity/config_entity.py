import os
from datetime import datetime
from dotenv import load_dotenv

# Automatically load .env from project root
load_dotenv()

class KGAuditPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        timestamp_str = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name: str = "EpistemeGraph_Research_Pipeline"
        self.artifact_dir: str = os.path.join("Artifacts", timestamp_str)
        self.model_dir: str = os.path.join("final_model")
        self.data_uploads_dir: str = os.path.join("data", "uploads")
        self.llm_cache_dir: str = os.path.join("Artifacts", "llm_cache")
        self.timestamp: str = timestamp_str

class KGBuilderConfig:
    def __init__(self, pipeline_config: KGAuditPipelineConfig):
        self.builder_dir: str = os.path.join(
            pipeline_config.artifact_dir, "kg_builder"
        )
        self.extracted_kg_file_path: str = os.path.join(
            self.builder_dir, "extracted_knowledge_graph.json"
        )
        self.kg_fidelity_state_path: str = os.path.join(
            pipeline_config.model_dir, "kg_fidelity_state.json"
        )
        self.data_uploads_dir: str = pipeline_config.data_uploads_dir
        self.llm_cache_dir: str = pipeline_config.llm_cache_dir
        self.neo4j_uri: str = os.getenv("NEO4J_URI", "neo4j+s://localhost:7687")
        self.neo4j_user: str = os.getenv("NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j"))
        self.neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
        self.neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
        self.llm_model_name: str = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
        self.chunk_size: int = 2400  # Larger chunks = fewer API calls = saves $3 budget
        self.chunk_overlap: int = 200
        self.entity_sim_threshold: float = 0.86

class LLMInterceptorConfig:
    def __init__(self, pipeline_config: KGAuditPipelineConfig):
        self.interceptor_dir: str = os.path.join(
            pipeline_config.artifact_dir, "llm_interceptor"
        )
        self.captured_payload_file_path: str = os.path.join(
            self.interceptor_dir, "interception_payload.json"
        )
        self.llm_model_name: str = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
        self.temperature: float = 0.0

class GraphRetrieverConfig:
    def __init__(self, pipeline_config: KGAuditPipelineConfig):
        self.retriever_dir: str = os.path.join(
            pipeline_config.artifact_dir, "graph_retriever"
        )
        self.subgraph_export_file_path: str = os.path.join(
            self.retriever_dir, "extracted_subgraph.json"
        )
        self.neo4j_uri: str = os.getenv("NEO4J_URI", "neo4j+s://localhost:7687")
        self.neo4j_user: str = os.getenv("NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j"))
        self.neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
        self.neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
        self.max_hop_depth: int = 2
        self.top_k_entities: int = 10

class TruthEvaluatorConfig:
    def __init__(self, pipeline_config: KGAuditPipelineConfig):
        self.evaluator_dir: str = os.path.join(
            pipeline_config.artifact_dir, "truth_evaluator"
        )
        self.evaluation_report_file_path: str = os.path.join(
            self.evaluator_dir, "accuracy_report.json"
        )
        self.trained_model_path: str = os.path.join(
            pipeline_config.model_dir, "model.pkl"
        )
        self.kg_fidelity_state_path: str = os.path.join(
            pipeline_config.model_dir, "kg_fidelity_state.json"
        )
        self.embedding_model_name: str = "all-MiniLM-L6-v2"
        self.entailment_weight: float = 0.4
        self.triple_match_weight: float = 0.4
        self.topological_weight: float = 0.2
        self.accuracy_pass_threshold: float = 0.75
        self.abstention_threshold: float = 0.50

class ModelTrainerConfig:
    def __init__(self, pipeline_config: KGAuditPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            pipeline_config.artifact_dir, "model_trainer"
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, "model.pkl"
        )
        self.final_model_file_path: str = os.path.join(
            pipeline_config.model_dir, "model.pkl"
        )
        self.metrics_report_file_path: str = os.path.join(
            self.model_trainer_dir, "calibration_metrics.json"
        )
        self.expected_roc_auc: float = 0.75
        self.max_brier_score: float = 0.20
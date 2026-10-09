import os
from datetime import datetime

class KGAuditPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        timestamp_str = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name: str = "KGAudit_Research_Pipeline"
        self.artifact_dir: str = os.path.join("Artifacts", timestamp_str)
        self.model_dir: str = os.path.join("final_model")
        self.timestamp: str = timestamp_str

class LLMInterceptorConfig:
    def __init__(self, pipeline_config: KGAuditPipelineConfig):
        self.interceptor_dir: str = os.path.join(
            pipeline_config.artifact_dir, "llm_interceptor"
        )
        self.captured_payload_file_path: str = os.path.join(
            self.interceptor_dir, "interception_payload.json"
        )
        self.llm_model_name: str = "gpt-4o"
        self.temperature: float = 0.0

class GraphRetrieverConfig:
    def __init__(self, pipeline_config: KGAuditPipelineConfig):
        self.retriever_dir: str = os.path.join(
            pipeline_config.artifact_dir, "graph_retriever"
        )
        self.subgraph_export_file_path: str = os.path.join(
            self.retriever_dir, "extracted_subgraph.json"
        )
        self.neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
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
        self.entailment_weight: float = 0.4
        self.triple_match_weight: float = 0.4
        self.topological_weight: float = 0.2
        self.accuracy_pass_threshold: float = 0.75
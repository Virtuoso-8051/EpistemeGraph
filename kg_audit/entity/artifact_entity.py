from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class LLMInterceptorArtifact:
    user_query: str
    llm_response: str
    source_documents: List[str]
    captured_payload_path: str

@dataclass
class GraphRetrieverArtifact:
    query_entities: List[str]
    extracted_nodes: List[Dict[str, Any]]
    extracted_edges: List[Dict[str, Any]]
    subgraph_file_path: str
    retrieval_status: bool

@dataclass
class TruthEvaluatorArtifact:
    semantic_entailment_score: float
    triple_match_score: float
    topological_similarity_score: float
    bayesian_source_trust: float
    temporal_freshness: float
    source_independence: float
    conflict_ratio: float
    feature_vector: List[float]
    final_accuracy_score: float
    decision_status: str  # "VERIFIED", "ABSTAIN", or "HALLUCINATED"
    is_hallucinated: bool
    evaluation_report_path: str

@dataclass
class ModelTrainerArtifact:
    best_model_name: str
    trained_model_file_path: str
    roc_auc_score: float
    f1_score: float
    brier_score: float
    expected_calibration_error: float
    is_model_accepted: bool
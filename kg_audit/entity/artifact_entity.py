from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class KGBuilderArtifact:
    total_chunks_processed: int
    raw_triplets_extracted: int
    verified_triplets_pushed: int
    resolved_entity_count: int
    provenance_grounding_score: float
    semantic_coverage_score: float
    topological_coherence_score: float
    internal_consistency_score: float
    overall_kg_accuracy_score: float
    kg_fidelity_state_path: str
    extracted_kg_file_path: str
    neo4j_cloud_pushed: bool
    new_documents: List[str] = field(default_factory=list)
    skipped_documents: List[str] = field(default_factory=list)


@dataclass
class LLMInterceptorArtifact:
    user_query: str
    llm_response: str
    llm_self_reported_accuracy: float
    llm_confidence_reasoning: str
    source_documents: List[str]
    captured_payload_path: str
    mode: str = "closed_book"


@dataclass
class GraphRetrieverArtifact:
    query_entities: List[str]
    extracted_nodes: List[Dict[str, Any]]
    extracted_edges: List[Dict[str, Any]]
    subgraph_file_path: str
    retrieval_status: bool
    seed_count: int = 0


@dataclass
class TruthEvaluatorArtifact:
    feature_vector: List[float]
    feature_names: List[str]
    features: Dict[str, float]
    query_belief_score: Optional[float]          # S_query  (None until a model is calibrated)
    phase1_kg_fidelity_score: float              # global S_KG
    local_kg_fidelity_score: float               # S_KG(q)
    final_accuracy_score: Optional[float]        # S_final
    decision_status: str                         # VERIFIED | ABSTAIN | HALLUCINATED | UNCALIBRATED
    is_hallucinated: bool
    conformal_prediction_set: List[int]
    evaluation_report_path: str
    edge_audit_details: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelTrainerArtifact:
    best_model_name: str
    trained_model_file_path: str
    roc_auc_score: float
    f1_score: float
    brier_score: float
    expected_calibration_error: float
    log_loss: float
    is_model_accepted: bool
    n_labelled_samples: int
    conformal: Dict[str, Any] = field(default_factory=dict)
    pool: Dict[str, Any] = field(default_factory=dict)
    learned_feature_weights: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, Any] = field(default_factory=dict)

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


@dataclass
class LLMInterceptorArtifact:
    user_query: str
    llm_response: str
    llm_self_reported_accuracy: float
    llm_confidence_reasoning: str
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
    llm_self_reported_accuracy: float
    query_belief_score: float
    phase1_kg_fidelity_score: float
    feature_vector: List[float]
    final_accuracy_score: float
    decision_status: str
    is_hallucinated: bool
    evaluation_report_path: str
    edge_audit_details: Optional[List[Dict[str, Any]]] = field(default_factory=list)


@dataclass
class ModelTrainerArtifact:
    best_model_name: str
    trained_model_file_path: str
    roc_auc_score: float
    f1_score: float
    brier_score: float
    expected_calibration_error: float
    is_model_accepted: bool
    learned_feature_weights: Optional[Dict[str, float]] = field(default_factory=dict)
    optimal_pass_threshold: float = 0.72
    optimal_abstain_threshold: float = 0.45
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from kg_audit.pipelines.evaluation_pipeline import EvaluationPipeline
from kg_audit.logging.logger import logging

app = FastAPI(
    title="EpistemeGraph: KG Audit & Verification Engine",
    description="Dual-Path Phase 1 Multi-PDF KG Builder + Phase 3 CASSE Hallucination Detector",
    version="0.0.2"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BuildKGRequest(BaseModel):
    file_paths: Optional[List[str]] = None

class VerificationRequest(BaseModel):
    user_query: str
    source_documents: Optional[List[str]] = []

@app.get("/", tags=["Root"])
async def index():
    return RedirectResponse(url="/docs")

@app.post("/build-kg", tags=["Phase 1: KG Construction & Fidelity"])
async def build_knowledge_graph(request: BuildKGRequest):
    try:
        pipeline = EvaluationPipeline()
        artifact = pipeline.run_phase1_kg_construction(file_paths=request.file_paths)
        return {
            "status": "success",
            "overall_kg_accuracy_score": artifact.overall_kg_accuracy_score,
            "verified_triplets_pushed": artifact.verified_triplets_pushed,
            "resolved_entity_count": artifact.resolved_entity_count,
            "neo4j_cloud_pushed": artifact.neo4j_cloud_pushed,
            "extracted_kg_file_path": artifact.extracted_kg_file_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify", tags=["Phase 2 & 3: Query Interception & CASSE Audit"])
async def verify_llm_response(request: VerificationRequest):
    try:
        pipeline = EvaluationPipeline()
        interceptor_art, retriever_art, eval_art = pipeline.run_pipeline(
            user_query=request.user_query,
            source_documents=request.source_documents
        )
        return {
            "query": request.user_query,
            "path1_chatgpt": {
                "llm_response": interceptor_art.llm_response,
                "llm_self_reported_accuracy": interceptor_art.llm_self_reported_accuracy,
                "llm_confidence_reasoning": interceptor_art.llm_confidence_reasoning
            },
            "path2_epistemegraph_audit": {
                "decision_status": eval_art.decision_status,
                "is_hallucinated": eval_art.is_hallucinated,
                "final_compound_accuracy_score": eval_art.final_accuracy_score,
                "query_belief_score": eval_art.query_belief_score,
                "phase1_kg_fidelity_score": eval_art.phase1_kg_fidelity_score,
                "sheaf_laplacian_energy": eval_art.semantic_entailment_score,
                "curvature_distortion_score": eval_art.triple_match_score,
                "spectral_divergence_score": eval_art.topological_similarity_score,
                "conflict_ratio": eval_art.conflict_ratio
            },
            "report_path": eval_art.evaluation_report_path
        }
    except Exception as e:
        logging.error(f"Verification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
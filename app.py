import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from kg_audit.pipelines.evaluation_pipeline import EvaluationPipeline
from kg_audit.exceptions.exception import KGAuditException
from kg_audit.logging.logger import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

app = FastAPI(
    title="KG Audit & Verification Engine",
    description="Research API for CASSE-based LLM Hallucination Detection via Neo4j Knowledge Graphs.",
    version="0.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic schema for the incoming request
class VerificationRequest(BaseModel):
    user_query: str
    source_documents: Optional[List[str]] = []

@app.get("/", tags=["Root"])
async def index():
    return RedirectResponse(url="/docs")

@app.post("/verify", tags=["Research Engine"])
async def verify_llm_response(request: VerificationRequest):
    try:
        logging.info(f"Received verification request for query: {request.user_query}")
        
        # Initialize and run the MLflow tracked pipeline
        pipeline = EvaluationPipeline()
        eval_artifact = pipeline.run_pipeline(
            user_query=request.user_query,
            source_documents=request.source_documents
        )
        
        # Construct the API response
        response_payload = {
            "query": request.user_query,
            "casse_evaluation": {
                "semantic_entailment_score": eval_artifact.semantic_entailment_score,
                "curvature_distortion_score": eval_artifact.triple_match_score,
                "spectral_divergence_score": eval_artifact.topological_similarity_score,
                "final_accuracy": eval_artifact.final_accuracy_score,
                "hallucination_detected": eval_artifact.is_hallucinated
            },
            "report_path": eval_artifact.evaluation_report_path
        }
        
        logging.info("Successfully returned verification response.")
        return response_payload

    except Exception as e:
        logging.error(f"Verification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
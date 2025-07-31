from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import time

from ..core.rag_metrics import RAGQualityMetrics

router = APIRouter()

# Pydantic models
class EvaluationRequest(BaseModel):
    query: str = Field(..., description="Original user query")
    context: str = Field(..., description="Retrieved context")
    response: str = Field(..., description="Generated response")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")

class EvaluationResponse(BaseModel):
    metrics: Dict[str, float]
    evaluation_id: str
    timestamp: float

class MetricsResponse(BaseModel):
    current_metrics: Dict[str, Any]
    timestamp: float

class FeedbackRequest(BaseModel):
    conversation_id: str = Field(..., description="Conversation ID")
    satisfaction_score: float = Field(..., ge=0.0, le=1.0, description="User satisfaction score (0.0-1.0)")
    feedback_text: Optional[str] = Field(None, description="Optional feedback text")

# Initialize metrics collector
metrics_collector = RAGQualityMetrics()

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_response(request: EvaluationRequest):
    """Evaluate the quality of a RAG response."""
    try:
        start_time = time.time()
        
        # Evaluate response quality
        metrics = await metrics_collector.evaluate_response(
            query=request.query,
            context=request.context,
            response=request.response
        )
        
        # Record processing time
        processing_time = time.time() - start_time
        metrics_collector.record_response_time(processing_time)
        
        evaluation_id = f"eval_{int(time.time() * 1000)}"
        
        return EvaluationResponse(
            metrics=metrics,
            evaluation_id=evaluation_id,
            timestamp=time.time()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get current system metrics."""
    try:
        current_metrics = metrics_collector.get_current_metrics()
        
        return MetricsResponse(
            current_metrics=current_metrics,
            timestamp=time.time()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )

@router.post("/feedback")
async def record_feedback(request: FeedbackRequest):
    """Record user feedback for a conversation."""
    try:
        # Record satisfaction score
        metrics_collector.record_user_feedback(request.satisfaction_score)
        
        return {
            "message": "Feedback recorded successfully",
            "conversation_id": request.conversation_id,
            "satisfaction_score": request.satisfaction_score
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record feedback: {str(e)}"
        )

@router.get("/health-metrics")
async def get_health_metrics():
    """Get service health and performance metrics."""
    try:
        current_metrics = metrics_collector.get_current_metrics()
        
        # Calculate health indicators
        total_queries = current_metrics.get('total_queries', 0)
        failed_queries = current_metrics.get('failed_queries', 0)
        
        success_rate = 1.0
        if total_queries > 0:
            success_rate = (total_queries - failed_queries) / total_queries
        
        health_status = "healthy" if success_rate > 0.95 else "degraded"
        
        return {
            "status": health_status,
            "success_rate": success_rate,
            "total_queries": total_queries,
            "failed_queries": failed_queries,
            "avg_response_time": current_metrics.get('avg_response_time', 0.0),
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get health metrics: {str(e)}"
        )
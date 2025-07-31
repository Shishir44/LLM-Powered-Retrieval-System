from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated response")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Based on the documentation, Docker is a containerization platform...",
                "conversation_id": "conv_123",
                "sources": [{"id": "doc_1", "title": "Docker Overview", "relevance_score": 0.95}],
                "metadata": {
                    "query_analysis": {"complexity": "SIMPLE", "intent": "INFORMATIONAL"},
                    "quality_metrics": {"accuracy": 4.5, "completeness": 4.2},
                    "verification_result": {"factual_accuracy_score": 0.92},
                    "processing_time": 1.23,
                    "context_info": {"context_size": 5, "context_strategy": "recency_based"},
                    "synthesis_info": {"strategy": "convergent", "source_count": 3}
                }
            }
        }

class EnhancedChatResponse(ChatResponse):
    """Enhanced response with Phase 2 intelligence features."""
    
    query_analysis: Dict[str, Any] = Field(default_factory=dict, description="Query analysis results")
    verification_result: Dict[str, Any] = Field(default_factory=dict, description="Fact verification results")
    synthesis_info: Dict[str, Any] = Field(default_factory=dict, description="Multi-source synthesis information")
    context_info: Dict[str, Any] = Field(default_factory=dict, description="Context management information")
    quality_metrics: Dict[str, Any] = Field(default_factory=dict, description="Response quality metrics")
    processing_time: float = Field(0.0, description="Total processing time in seconds")

class VerificationResponse(BaseModel):
    verification_result: Dict[str, Any] = Field(..., description="Fact verification results")
    timestamp: str = Field(..., description="Verification timestamp")

class SynthesisResponse(BaseModel):
    synthesis_result: Dict[str, Any] = Field(..., description="Multi-source synthesis results")
    timestamp: str = Field(..., description="Synthesis timestamp")

class ContextResponse(BaseModel):
    conversation_id: str = Field(..., description="Conversation identifier")
    context: Dict[str, Any] = Field(..., description="Conversation context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Context metadata")
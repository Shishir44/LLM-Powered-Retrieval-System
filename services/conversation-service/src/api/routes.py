from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import json
import logging

logger = logging.getLogger(__name__)

# from ..core.context_manager import ConversationContextManager, ConversationState  # Removed - using enhanced pipeline directly
from ..core.adaptive_rag_pipeline import EnhancedRAGPipeline, HTTPKnowledgeRetriever
from ..core.streaming import StreamingRAGResponse
from ..models.requests import ChatRequest, VerificationRequest, SynthesisRequest, ContextRequest
from ..models.responses import ChatResponse

router = APIRouter()

# Pydantic models
class ConversationRequest(BaseModel):
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

class ConversationResponse(BaseModel):
    response: str
    conversation_id: str
    context: Dict[str, Any]
    metadata: Dict[str, Any] = {}

class StreamingConversationRequest(BaseModel):
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    stream: bool = Field(default=True, description="Enable streaming")

# Initialize components
# context_manager = ConversationContextManager()  # Removed - using enhanced pipeline directly
import os
# Initialize HTTP-based knowledge retriever
knowledge_base_url = os.getenv("KNOWLEDGE_BASE_URL", "http://knowledge-base-service:8002")
knowledge_retriever = HTTPKnowledgeRetriever(
    base_url=knowledge_base_url
)

rag_pipeline = EnhancedRAGPipeline(
    knowledge_retriever=knowledge_retriever,
    conversation_context_manager=None,
    config=None
)
streaming_service = StreamingRAGResponse()

@router.post("/chat", response_model=ConversationResponse)
async def chat(request: ConversationRequest):
    """Process a conversation message."""
    try:
        # Create conversation identifiers
        conversation_id = request.conversation_id or "new_conversation"
        
        # Process through enhanced RAG pipeline
        rag_response = await rag_pipeline.process_query(
            conversation_id=conversation_id,
            user_message=request.message,
            user_profile=None,
            conversation_context=""
        )
        
        # Get the generated response from the enhanced pipeline
        response_text = rag_response.response
        
        # Extract metadata from the enhanced processing
        context_data = {
            'query_analysis': rag_response.query_analysis.__dict__,
            'contextual_info': rag_response.contextual_info.__dict__,
            'quality_metrics': rag_response.quality_metrics.__dict__,
            'processing_metadata': {
                'processing_time': rag_response.processing_time,
                'confidence_score': rag_response.confidence_score
            }
        }
        processing_metadata = context_data['processing_metadata']
        quality_metrics = context_data['quality_metrics']
        
        return ConversationResponse(
            response=response_text,
            conversation_id=conversation_id,
            context=context_data,
            metadata={
                "processing_time": processing_metadata.get('processing_time', 0.5),
                "confidence_score": processing_metadata.get('confidence_score', 0.7),
                "quality_score": quality_metrics.get('overall_score', 3.5),
                "query_analysis": context_data.get('query_analysis', {}),
                "enhanced_pipeline": True
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Conversation processing failed: {str(e)}"
        )

@router.post("/chat/stream")
async def chat_stream(request: StreamingConversationRequest):
    """Process a conversation message with streaming response."""
    try:
        # Create conversation identifiers
        conversation_id = request.conversation_id or "new_conversation"
        
        async def generate_response():
            # Stream response chunks
            async for chunk in streaming_service.stream_response(request.message, conversation_id):
                yield f"data: {json.dumps({'chunk': chunk})}\\n\\n"
            
            yield f"data: {json.dumps({'done': True})}\\n\\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Streaming chat failed: {str(e)}"
        )

@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history."""
    try:
        # Retrieve conversation (implementation needed)
        return {
            "conversation_id": conversation_id,
            "history": [],
            "metadata": {}
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation: {str(e)}"
        )

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    try:
        # Delete conversation (implementation needed)
        return {"message": "Conversation deleted successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}"
        )

@router.get("/pipeline/stats")
async def get_pipeline_statistics():
    """Get RAG pipeline performance statistics."""
    try:
        stats = await rag_pipeline.get_pipeline_stats()
        return {
            "status": "success",
            "statistics": stats,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pipeline statistics: {str(e)}"
        )

@router.post("/pipeline/optimize")
async def optimize_pipeline():
    """Optimize RAG pipeline based on performance history."""
    try:
        optimization_result = await rag_pipeline.optimize_pipeline()
        return {
            "status": "success",
            "optimization_result": optimization_result,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to optimize pipeline: {str(e)}"
        )

# Add customer context endpoints

@router.post("/enhanced-chat")
async def enhanced_chat(request: ChatRequest):
    """Enhanced chat endpoint with customer context support."""
    
    try:
        # Extract customer ID from request
        customer_id = getattr(request, 'customer_id', None) or getattr(request, 'user_id', None)
        
        # Process query with customer context
        result = await rag_pipeline.process_query(
            conversation_id=request.conversation_id,
            user_message=request.message,
            user_profile=None,
            conversation_context=""
        )
        
        # Handle both dict and RAGResponse objects
        if isinstance(result, dict):
            return ChatResponse(
                response=result.get("response", "I apologize, but I encountered an error processing your request."),
                conversation_id=request.conversation_id,
                metadata=result.get("metadata", {})
            )
        else:
            return ChatResponse(
                response=result.response,
                conversation_id=request.conversation_id,
                metadata={
                    "processing_time": result.processing_time,
                    "confidence_score": result.confidence_score,
                    "query_analysis": result.query_analysis.__dict__ if hasattr(result.query_analysis, '__dict__') else {},
                    "quality_metrics": result.quality_metrics.__dict__ if hasattr(result.quality_metrics, '__dict__') else {}
                }
        )
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Enhanced chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/customers/{customer_id}/profile")
async def update_customer_profile(
    customer_id: str,
    profile_updates: Dict[str, Any]
):
    """Update customer profile and preferences."""
    
    try:
        updated_profile = await rag_pipeline.customer_profile_manager.update_profile(
            customer_id, profile_updates
        )
        
        return {
            "status": "success",
            "customer_id": customer_id,
            "updated_at": updated_profile.updated_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Profile update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/customers/{customer_id}/context")
async def get_customer_context(customer_id: str):
    """Get customer context and preferences."""
    
    try:
        context = await rag_pipeline.customer_profile_manager.analyze_context(customer_id, "")
        
        return {
            "customer_id": customer_id,
            "context": context
        }
        
    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/customers/{customer_id}/feedback")
async def record_customer_feedback(
    customer_id: str,
    feedback: Dict[str, Any]
):
    """Record customer feedback for interaction improvement."""
    
    try:
        # Update satisfaction score and preferences based on feedback
        satisfaction_score = feedback.get("satisfaction_score")
        if satisfaction_score is not None:
            await rag_pipeline.customer_profile_manager.record_interaction(
                customer_id=customer_id,
                query=feedback.get("query", ""),
                response=feedback.get("response", ""),
                satisfaction_score=satisfaction_score,
                resolution_status=feedback.get("resolution_status", "resolved")
            )
        
        return {"status": "success", "message": "Feedback recorded"}
        
    except Exception as e:
        logger.error(f"Feedback recording failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

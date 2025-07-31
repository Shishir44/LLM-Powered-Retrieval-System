"""
Streaming response handler for RAG system.
"""

import asyncio
import json
from typing import AsyncGenerator, Optional
from .adaptive_rag_pipeline import EnhancedRAGPipeline
import os


class StreamingRAGResponse:
    """Handles streaming responses for the RAG system."""
    
    def __init__(self):
        """Initialize the streaming service."""
        self.rag_pipeline = EnhancedRAGPipeline(
            knowledge_retriever=None,
            conversation_context_manager=None,
            config=None
        )
    
    async def stream_response(self, message: str, conversation_id: str) -> AsyncGenerator[str, None]:
        """
        Stream response chunks for a given message.
        
        Args:
            message: User message to process
            conversation_id: Conversation identifier
            
        Yields:
            Response chunks as strings
        """
        try:
            # Process through enhanced RAG pipeline
            rag_response = await self.rag_pipeline.process_query(
                conversation_id=conversation_id,
                user_message=message,
                user_profile=None,
                conversation_context=""
            )
            
            # Get the full response
            full_response = rag_response.response
            
            # Stream the response in chunks
            chunk_size = 50  # Characters per chunk
            for i in range(0, len(full_response), chunk_size):
                chunk = full_response[i:i + chunk_size]
                yield chunk
                # Small delay to simulate streaming
                await asyncio.sleep(0.05)
                
        except Exception as e:
            # Yield error information
            yield f"Error processing request: {str(e)}"
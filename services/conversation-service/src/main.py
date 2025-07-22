"""
Conversation Service - Main Application

Independent microservice for conversation management and RAG pipeline processing.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging

# from .core.context_manager import ConversationContextManager, ConversationState  # Removed - using enhanced pipeline directly
from .core.adaptive_rag_pipeline import EnhancedRAGPipeline
from .core.streaming import StreamingRAGResponse
from .api.routes import router as api_router

# Configuration
class ConversationConfig:
    def __init__(self):
        self.service_name = "conversation-service"
        self.version = "1.0.0"
        self.host = "0.0.0.0"
        self.port = 8001

config = ConversationConfig()

# FastAPI app
app = FastAPI(
    title="Conversation Service",
    description="Independent microservice for conversation management and RAG processing",
    version=config.version
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Health check
@app.get("/health")
async def health_check():
    """Service health check."""
    return {
        "status": "healthy",
        "service": config.service_name,
        "version": config.version
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": config.service_name,
        "version": config.version,
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(level=logging.INFO)
    
    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=True
    )
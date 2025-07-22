"""
Knowledge Base Service - Main Application

Independent microservice for document storage, vector search, and RAG operations.
"""

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from uuid import uuid4
import logging
import time

from .core.retrieval import AdvancedRAGRetriever
from .core.chunking import DocumentChunker
from .core.cache import VectorCache
from .api.routes import router as api_router

# Configuration
class KnowledgeBaseConfig:
    def __init__(self):
        self.service_name = "knowledge-base-service"
        self.version = "1.0.0"
        self.host = "0.0.0.0"
        self.port = 8002

config = KnowledgeBaseConfig()

# FastAPI app
app = FastAPI(
    title="Knowledge Base Service",
    description="Independent microservice for document storage and RAG operations",
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
#!/usr/bin/env python3
"""
Simple runner for the Customer Support Platform API Gateway
This starts the API Gateway with minimal dependencies for development.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Simple FastAPI app for development
app = FastAPI(
    title="Customer Support Platform API",
    description="AI-powered customer support platform with LangChain and LangGraph",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ConversationRequest(BaseModel):
    message: str
    metadata: dict = {}

class ConversationResponse(BaseModel):
    conversation_id: str
    session_id: str
    response: str
    metadata: dict

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="api-gateway",
        version="1.0.0"
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Customer Support Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Mock conversation endpoint
@app.post("/api/v1/conversations", response_model=ConversationResponse)
async def create_conversation(request: ConversationRequest):
    """Create a new conversation (mock implementation)."""
    return ConversationResponse(
        conversation_id="conv_123456",
        session_id="session_789",
        response="Hello! This is a mock response. To enable full functionality, please configure your OpenAI API key in the .env file.",
        metadata={
            "intent": "greeting",
            "sentiment": "neutral",
            "confidence": 0.8,
            "mock": True
        }
    )

# Mock knowledge base endpoint
@app.get("/api/v1/knowledge-base/search")
async def search_knowledge(q: str, limit: int = 5):
    """Search knowledge base (mock implementation)."""
    return {
        "query": q,
        "results": [
            {
                "id": "doc_1",
                "title": "Sample Document",
                "content": "This is a mock search result.",
                "score": 0.9,
                "category": "faq"
            }
        ],
        "total": 1,
        "mock": True
    }

if __name__ == "__main__":
    print("🚀 Starting Customer Support Platform API Gateway...")
    print("📚 API Documentation: http://localhost:8080/docs")
    print("❤️  Health Check: http://localhost:8080/health")
    print()
    print("💡 To enable full functionality:")
    print("   1. Add your OpenAI API key to .env file: OPENAI_API_KEY=your_key_here")
    print("   2. Use 'make dev' for full Docker setup with all services")
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
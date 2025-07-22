"""
Analytics Service - Main Application

Independent microservice for metrics collection and RAG quality evaluation.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
import logging

from .core.rag_metrics import RAGQualityMetrics
from .api.routes import router as api_router

# Configuration
class AnalyticsConfig:
    def __init__(self):
        self.service_name = "analytics-service"
        self.version = "1.0.0"
        self.host = "0.0.0.0"
        self.port = 8005

config = AnalyticsConfig()

# FastAPI app
app = FastAPI(
    title="Analytics Service",
    description="Independent microservice for metrics and RAG quality evaluation",
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

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

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
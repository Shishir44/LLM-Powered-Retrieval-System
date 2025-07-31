"""Test cross-encoder reranking functionality."""

import asyncio
import pytest
from services.knowledge_base_service.src.core.reranker import CrossEncoderReranker
from services.shared.config import EnhancedRAGConfig

class TestCrossEncoderReranking:
    
    @pytest.fixture
    async def reranker(self):
        config = EnhancedRAGConfig()
        return CrossEncoderReranker(config)
    
    async def test_document_reranking(self, reranker):
        """Test document reranking with cross-encoder."""
        
        query = "How to deploy applications using Docker containers?"
        
        # Mock documents with varying relevance
        documents = [
            {
                "id": "doc1",
                "content": "Docker is a containerization platform that helps developers package applications.",
                "title": "Introduction to Docker"
            },
            {
                "id": "doc2", 
                "content": "Kubernetes orchestrates Docker containers in production environments for deployment.",
                "title": "Kubernetes Deployment Guide"
            },
            {
                "id": "doc3",
                "content": "Python programming language basics and syntax overview.",
                "title": "Python Tutorial"
            },
            {
                "id": "doc4",
                "content": "Docker deployment strategies include blue-green deployments and rolling updates.",
                "title": "Docker Deployment Strategies"
            }
        ]
        
        result = await reranker.rerank_documents(query, documents, top_k=3)
        
        assert len(result.reranked_documents) <= 3
        assert len(result.relevance_scores) == len(result.reranked_documents)
        assert result.reranking_time > 0
        
        # Check that Docker-related documents are ranked higher
        top_doc = result.reranked_documents[0]
        assert "docker" in top_doc["content"].lower() or "docker" in top_doc["title"].lower()
        
        print(f"✅ Reranking completed in {result.reranking_time:.3f}s")
        print(f"📊 Model used: {result.model_used}")
        print(f"🎯 Diversity score: {result.diversity_score:.3f}")
        
        for i, (doc, score) in enumerate(zip(result.reranked_documents, result.relevance_scores)):
            print(f"  {i+1}. {doc['title']} (Score: {score:.3f})")

if __name__ == "__main__":
    asyncio.run(TestCrossEncoderReranking().test_document_reranking(None))
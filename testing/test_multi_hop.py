"""Test multi-hop query processing functionality."""

import asyncio
import pytest
from services.conversation_service.src.core.multi_hop_processor import MultiHopProcessor
from services.conversation_service.src.core.query_decomposer import QueryDecomposer
from services.shared.config import EnhancedRAGConfig

class TestMultiHopProcessing:
    
    @pytest.fixture
    async def processor(self):
        config = EnhancedRAGConfig()
        return MultiHopProcessor(config)
    
    async def test_query_decomposition(self, processor):
        """Test complex query decomposition."""
        
        complex_query = "How do I set up Docker containers and then deploy them to Kubernetes, and what are the differences between Docker Swarm and Kubernetes for orchestration?"
        
        # Mock query analysis
        from services.conversation_service.src.models.query_analysis import QueryAnalysis, QueryType
        query_analysis = QueryAnalysis(
            query_type=QueryType.COMPARISON,
            complexity_score=0.8,
            intent="setup_and_compare"
        )
        
        result = await processor.process_multi_hop_query(complex_query, query_analysis)
        
        assert result.original_query == complex_query
        assert len(result.execution_path) > 0
        assert result.total_processing_time > 0
        
        print(f"✅ Multi-hop processing completed")
        print(f"📊 Execution path: {' → '.join(result.execution_path)}")
        print(f"⏱️ Processing time: {result.total_processing_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(TestMultiHopProcessing().test_query_decomposition(None))
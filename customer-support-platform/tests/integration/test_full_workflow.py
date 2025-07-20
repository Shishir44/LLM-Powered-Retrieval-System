"""Integration tests for full workflow scenarios."""

import pytest
from unittest.mock import patch, Mock
from httpx import AsyncClient

from tests.conftest import (
    assert_response_success,
    assert_conversation_response,
    assert_message_response,
    assert_document_response,
    assert_search_response
)


class TestFullConversationWorkflow:
    """Test complete conversation workflow from start to finish."""
    
    @pytest.mark.asyncio
    async def test_complete_conversation_flow(
        self,
        api_gateway_client: AsyncClient,
        auth_headers: dict,
        sample_conversation_data: dict,
        sample_message_data: dict
    ):
        """Test complete conversation workflow."""
        
        # Mock all service calls
        with patch('services.api_gateway.main.service_client.make_request') as mock_request:
            # Step 1: Create conversation
            mock_request.return_value = {
                "conversation_id": "test_conv_id",
                "session_id": "test_session_id",
                "response": "Hello! How can I help you?",
                "metadata": {
                    "intent": "question",
                    "sentiment": "neutral",
                    "confidence": 0.8
                }
            }
            
            response = await api_gateway_client.post(
                "/api/v1/conversations",
                json=sample_conversation_data,
                headers=auth_headers
            )
            
            assert_response_success(response)
            conv_data = response.json()
            assert_conversation_response(conv_data)
            
            conversation_id = conv_data["conversation_id"]
            
            # Step 2: Send follow-up message
            mock_request.return_value = {
                "message_id": "test_msg_id",
                "response": "I can help you with account issues. What specific problem are you experiencing?",
                "metadata": {
                    "intent": "account_support",
                    "sentiment": "neutral",
                    "confidence": 0.9
                }
            }
            
            response = await api_gateway_client.post(
                f"/api/v1/conversations/{conversation_id}/messages",
                json=sample_message_data,
                headers=auth_headers
            )
            
            assert_response_success(response)
            msg_data = response.json()
            assert_message_response(msg_data)
            
            # Step 3: Get conversation history
            mock_request.return_value = {
                "conversation_id": conversation_id,
                "user_id": "test_user_id",
                "status": "active",
                "messages": [
                    {
                        "id": "msg1",
                        "content": "Hello, I need help with my account",
                        "type": "user",
                        "timestamp": "2023-01-01T00:00:00Z"
                    },
                    {
                        "id": "msg2",
                        "content": "Hello! How can I help you?",
                        "type": "assistant",
                        "timestamp": "2023-01-01T00:00:01Z"
                    }
                ]
            }
            
            response = await api_gateway_client.get(
                f"/api/v1/conversations/{conversation_id}",
                headers=auth_headers
            )
            
            assert_response_success(response)
            conv_history = response.json()
            assert "messages" in conv_history
            assert len(conv_history["messages"]) == 2
    
    @pytest.mark.asyncio
    async def test_escalation_workflow(
        self,
        api_gateway_client: AsyncClient,
        auth_headers: dict
    ):
        """Test escalation workflow for complex issues."""
        
        with patch('services.api_gateway.main.service_client.make_request') as mock_request:
            # Create conversation with negative sentiment
            mock_request.return_value = {
                "conversation_id": "test_conv_id",
                "session_id": "test_session_id",
                "response": "I understand this is frustrating. Let me connect you with one of our human agents.",
                "metadata": {
                    "intent": "complaint",
                    "sentiment": "negative",
                    "confidence": 0.6,
                    "escalation_needed": True
                }
            }
            
            response = await api_gateway_client.post(
                "/api/v1/conversations",
                json={
                    "message": "I'm really frustrated with this service! Nothing works!",
                    "metadata": {"channel": "web"}
                },
                headers=auth_headers
            )
            
            assert_response_success(response)
            conv_data = response.json()
            assert conv_data["metadata"]["escalation_needed"] is True
            assert conv_data["metadata"]["sentiment"] == "negative"


class TestKnowledgeBaseWorkflow:
    """Test knowledge base operations workflow."""
    
    @pytest.mark.asyncio
    async def test_document_lifecycle(
        self,
        api_gateway_client: AsyncClient,
        auth_headers: dict,
        sample_document_data: dict
    ):
        """Test complete document lifecycle."""
        
        with patch('services.api_gateway.main.service_client.make_request') as mock_request:
            # Step 1: Create document
            mock_request.return_value = {
                "id": "test_doc_id",
                "title": sample_document_data["title"],
                "content": sample_document_data["content"],
                "category": sample_document_data["category"],
                "tags": sample_document_data["tags"],
                "metadata": sample_document_data["metadata"]
            }
            
            response = await api_gateway_client.post(
                "/api/v1/knowledge-base/documents",
                json=sample_document_data,
                headers=auth_headers
            )
            
            assert_response_success(response)
            doc_data = response.json()
            assert_document_response(doc_data)
            
            document_id = doc_data["id"]
            
            # Step 2: Search for document
            mock_request.return_value = {
                "results": [
                    {
                        "id": document_id,
                        "title": sample_document_data["title"],
                        "content": sample_document_data["content"],
                        "category": sample_document_data["category"],
                        "tags": sample_document_data["tags"],
                        "score": 0.95,
                        "metadata": sample_document_data["metadata"]
                    }
                ],
                "total": 1,
                "query": "reset password"
            }
            
            response = await api_gateway_client.get(
                "/api/v1/knowledge-base/search",
                params={"q": "reset password"},
                headers=auth_headers
            )
            
            assert_response_success(response)
            search_data = response.json()
            assert_search_response(search_data)
            assert len(search_data["results"]) == 1
            assert search_data["results"][0]["id"] == document_id
    
    @pytest.mark.asyncio
    async def test_search_with_filters(
        self,
        api_gateway_client: AsyncClient,
        auth_headers: dict
    ):
        """Test search with various filters."""
        
        with patch('services.api_gateway.main.service_client.make_request') as mock_request:
            mock_request.return_value = {
                "results": [
                    {
                        "id": "doc1",
                        "title": "Account Management",
                        "category": "account",
                        "tags": ["account", "management"],
                        "score": 0.9,
                        "metadata": {}
                    }
                ],
                "total": 1,
                "query": "account management"
            }
            
            response = await api_gateway_client.get(
                "/api/v1/knowledge-base/search",
                params={
                    "q": "account management",
                    "category": "account",
                    "limit": 5
                },
                headers=auth_headers
            )
            
            assert_response_success(response)
            search_data = response.json()
            assert search_data["total"] == 1
            assert search_data["results"][0]["category"] == "account"


class TestNLPWorkflow:
    """Test NLP processing workflow."""
    
    @pytest.mark.asyncio
    async def test_text_analysis_workflow(
        self,
        api_gateway_client: AsyncClient,
        auth_headers: dict
    ):
        """Test complete text analysis workflow."""
        
        with patch('services.api_gateway.main.service_client.make_request') as mock_request:
            mock_request.return_value = {
                "sentiment": {
                    "label": "negative",
                    "score": 0.85,
                    "confidence": 0.92
                },
                "intent": {
                    "label": "complaint",
                    "score": 0.78,
                    "confidence": 0.89
                },
                "spam_detection": {
                    "is_spam": False,
                    "confidence": 0.95
                },
                "metadata": {
                    "processing_time": 0.123,
                    "model_version": "v1.2.3"
                }
            }
            
            response = await api_gateway_client.post(
                "/api/v1/nlp/analyze",
                json={
                    "text": "I'm really frustrated with this service",
                    "features": ["sentiment", "intent", "spam_detection"]
                },
                headers=auth_headers
            )
            
            assert_response_success(response)
            analysis_data = response.json()
            
            assert "sentiment" in analysis_data
            assert "intent" in analysis_data
            assert "spam_detection" in analysis_data
            assert analysis_data["sentiment"]["label"] == "negative"
            assert analysis_data["intent"]["label"] == "complaint"


class TestAnalyticsWorkflow:
    """Test analytics and metrics workflow."""
    
    @pytest.mark.asyncio
    async def test_metrics_collection(
        self,
        api_gateway_client: AsyncClient,
        auth_headers: dict
    ):
        """Test metrics collection and retrieval."""
        
        with patch('services.api_gateway.main.service_client.make_request') as mock_request:
            mock_request.return_value = {
                "metrics": {
                    "total_conversations": 1250,
                    "avg_response_time": 2.3,
                    "resolution_rate": 0.85,
                    "customer_satisfaction": 4.2,
                    "escalation_rate": 0.15
                },
                "time_series": [
                    {
                        "date": "2023-01-01",
                        "conversations": 45,
                        "avg_response_time": 2.1
                    }
                ],
                "period": {
                    "start": "2023-01-01T00:00:00Z",
                    "end": "2023-01-31T23:59:59Z"
                }
            }
            
            response = await api_gateway_client.get(
                "/api/v1/analytics/metrics",
                params={
                    "start_date": "2023-01-01",
                    "end_date": "2023-01-31"
                },
                headers=auth_headers
            )
            
            assert_response_success(response)
            metrics_data = response.json()
            
            assert "metrics" in metrics_data
            assert "time_series" in metrics_data
            assert "period" in metrics_data
            assert metrics_data["metrics"]["total_conversations"] == 1250


class TestErrorHandling:
    """Test error handling across services."""
    
    @pytest.mark.asyncio
    async def test_service_unavailable_handling(
        self,
        api_gateway_client: AsyncClient,
        auth_headers: dict
    ):
        """Test handling when services are unavailable."""
        
        with patch('services.api_gateway.main.service_client.make_request') as mock_request:
            mock_request.side_effect = Exception("Service unavailable")
            
            response = await api_gateway_client.post(
                "/api/v1/conversations",
                json={"message": "Test message"},
                headers=auth_headers
            )
            
            assert response.status_code == 503
            error_data = response.json()
            assert "Service unavailable" in error_data["detail"]
    
    @pytest.mark.asyncio
    async def test_invalid_token_handling(
        self,
        api_gateway_client: AsyncClient
    ):
        """Test handling of invalid authentication tokens."""
        
        response = await api_gateway_client.post(
            "/api/v1/conversations",
            json={"message": "Test message"},
            headers={"Authorization": "Bearer invalid_token"}
        )
        
        assert response.status_code == 401
        error_data = response.json()
        assert "Invalid" in error_data["detail"]
    
    @pytest.mark.asyncio
    async def test_rate_limiting(
        self,
        api_gateway_client: AsyncClient,
        auth_headers: dict
    ):
        """Test rate limiting functionality."""
        
        with patch('services.api_gateway.main.rate_limiter.is_allowed') as mock_rate_limit:
            mock_rate_limit.return_value = False
            
            response = await api_gateway_client.post(
                "/api/v1/conversations",
                json={"message": "Test message"},
                headers=auth_headers
            )
            
            assert response.status_code == 429
            error_data = response.json()
            assert "Rate limit exceeded" in error_data["detail"]


class TestPerformanceScenarios:
    """Test performance-related scenarios."""
    
    @pytest.mark.asyncio
    async def test_concurrent_conversations(
        self,
        api_gateway_client: AsyncClient,
        auth_headers: dict
    ):
        """Test handling of concurrent conversations."""
        
        with patch('services.api_gateway.main.service_client.make_request') as mock_request:
            mock_request.return_value = {
                "conversation_id": "test_conv_id",
                "session_id": "test_session_id",
                "response": "Hello! How can I help you?",
                "metadata": {"intent": "question", "sentiment": "neutral"}
            }
            
            # Create multiple concurrent requests
            tasks = []
            for i in range(5):
                task = api_gateway_client.post(
                    "/api/v1/conversations",
                    json={"message": f"Test message {i}"},
                    headers=auth_headers
                )
                tasks.append(task)
            
            # Wait for all requests to complete
            responses = await asyncio.gather(*tasks)
            
            # Verify all requests succeeded
            for response in responses:
                assert response.status_code == 200
                assert "conversation_id" in response.json()
    
    @pytest.mark.asyncio
    async def test_large_document_processing(
        self,
        api_gateway_client: AsyncClient,
        auth_headers: dict
    ):
        """Test processing of large documents."""
        
        with patch('services.api_gateway.main.service_client.make_request') as mock_request:
            mock_request.return_value = {
                "id": "large_doc_id",
                "title": "Large Document",
                "category": "documentation",
                "tags": ["large", "document"],
                "metadata": {}
            }
            
            # Create a large document
            large_content = "This is a large document. " * 1000  # ~25KB
            
            response = await api_gateway_client.post(
                "/api/v1/knowledge-base/documents",
                json={
                    "title": "Large Document",
                    "content": large_content,
                    "category": "documentation",
                    "tags": ["large", "document"]
                },
                headers=auth_headers
            )
            
            assert_response_success(response)
            doc_data = response.json()
            assert doc_data["id"] == "large_doc_id"
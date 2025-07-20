"""Unit tests for conversation service."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from httpx import AsyncClient

from services.conversation_service.main import (
    ConversationService,
    CustomerSupportWorkflow,
    ConversationState
)
from shared.database.models import Conversation, Message


class TestCustomerSupportWorkflow:
    """Test CustomerSupportWorkflow class."""
    
    def test_workflow_initialization(self):
        """Test workflow initialization."""
        workflow = CustomerSupportWorkflow()
        
        assert workflow.llm is not None
        assert workflow.memory is not None
        assert workflow.workflow is not None
    
    @pytest.mark.asyncio
    async def test_classify_intent(self):
        """Test intent classification."""
        workflow = CustomerSupportWorkflow()
        
        # Mock LLM response
        with patch.object(workflow.llm, 'ainvoke', return_value="question") as mock_llm:
            state = ConversationState(
                conversation_id="test_id",
                user_id="user_id",
                session_id="session_id",
                current_message="How do I reset my password?"
            )
            
            result = await workflow.classify_intent(state)
            
            assert result.intent == "question"
            assert result.confidence == 0.8
            mock_llm.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment(self):
        """Test sentiment analysis."""
        workflow = CustomerSupportWorkflow()
        
        # Mock LLM response
        with patch.object(workflow.llm, 'ainvoke', return_value="positive") as mock_llm:
            state = ConversationState(
                conversation_id="test_id",
                user_id="user_id",
                session_id="session_id",
                current_message="I love this service!"
            )
            
            result = await workflow.analyze_sentiment(state)
            
            assert result.sentiment == "positive"
            mock_llm.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_knowledge(self):
        """Test knowledge search."""
        workflow = CustomerSupportWorkflow()
        
        state = ConversationState(
            conversation_id="test_id",
            user_id="user_id",
            session_id="session_id",
            current_message="How do I reset my password?"
        )
        
        result = await workflow.search_knowledge(state)
        
        assert "knowledge_results" in result.context
        assert "has_knowledge" in result.context
        assert isinstance(result.context["knowledge_results"], list)
    
    @pytest.mark.asyncio
    async def test_generate_response(self):
        """Test response generation."""
        workflow = CustomerSupportWorkflow()
        
        # Mock LLM response
        with patch.object(workflow.llm, 'ainvoke', return_value="Here's how to reset your password...") as mock_llm:
            state = ConversationState(
                conversation_id="test_id",
                user_id="user_id",
                session_id="session_id",
                current_message="How do I reset my password?",
                intent="question",
                sentiment="neutral",
                context={"has_knowledge": True, "knowledge_results": []}
            )
            
            result = await workflow.generate_response(state)
            
            assert "ai_response" in result.context
            assert result.context["ai_response"] == "Here's how to reset your password..."
            mock_llm.assert_called_once()
    
    def test_should_escalate_low_confidence(self):
        """Test escalation decision with low confidence."""
        workflow = CustomerSupportWorkflow()
        
        state = ConversationState(
            conversation_id="test_id",
            user_id="user_id",
            session_id="session_id",
            confidence=0.3,
            sentiment="neutral",
            context={"has_knowledge": True}
        )
        
        result = workflow.should_escalate(state)
        assert result == "escalate"
    
    def test_should_escalate_negative_complaint(self):
        """Test escalation decision with negative complaint."""
        workflow = CustomerSupportWorkflow()
        
        state = ConversationState(
            conversation_id="test_id",
            user_id="user_id",
            session_id="session_id",
            confidence=0.8,
            sentiment="negative",
            intent="complaint",
            context={"has_knowledge": True}
        )
        
        result = workflow.should_escalate(state)
        assert result == "escalate"
    
    def test_should_not_escalate_normal_case(self):
        """Test no escalation for normal case."""
        workflow = CustomerSupportWorkflow()
        
        state = ConversationState(
            conversation_id="test_id",
            user_id="user_id",
            session_id="session_id",
            confidence=0.8,
            sentiment="neutral",
            intent="question",
            context={"has_knowledge": True}
        )
        
        result = workflow.should_escalate(state)
        assert result == "respond"


class TestConversationService:
    """Test ConversationService class."""
    
    def test_service_initialization(self):
        """Test service initialization."""
        service = ConversationService()
        
        assert service.workflow is not None
        assert service.metrics is not None
    
    @pytest.mark.asyncio
    async def test_create_conversation(self, test_session):
        """Test conversation creation."""
        service = ConversationService()
        
        # Mock workflow processing
        with patch.object(service.workflow, 'process_message') as mock_workflow:
            mock_workflow.return_value = ConversationState(
                conversation_id="test_id",
                user_id="user_id",
                session_id="session_id",
                intent="question",
                sentiment="neutral",
                confidence=0.8,
                context={"ai_response": "Hello! How can I help you?"}
            )
            
            result = await service.create_conversation(
                user_id="user_id",
                message="Hello, I need help",
                metadata={"channel": "web"},
                db_session=test_session
            )
            
            assert result.conversation_id is not None
            assert result.session_id is not None
            assert result.response == "Hello! How can I help you?"
            assert result.metadata["intent"] == "question"
            assert result.metadata["sentiment"] == "neutral"
    
    @pytest.mark.asyncio
    async def test_send_message(self, test_session):
        """Test message sending."""
        service = ConversationService()
        
        # Create a conversation first
        conversation = Conversation(
            id="test_conversation_id",
            user_id="user_id",
            session_id="session_id",
            status="active"
        )
        test_session.add(conversation)
        await test_session.commit()
        
        # Mock workflow processing
        with patch.object(service.workflow, 'process_message') as mock_workflow:
            mock_workflow.return_value = ConversationState(
                conversation_id="test_conversation_id",
                user_id="user_id",
                session_id="session_id",
                intent="question",
                sentiment="neutral",
                confidence=0.8,
                context={"ai_response": "I can help with that!"}
            )
            
            result = await service.send_message(
                conversation_id="test_conversation_id",
                user_id="user_id",
                message="I need help with my account",
                metadata={"timestamp": "2023-01-01T00:00:00Z"},
                db_session=test_session
            )
            
            assert result.message_id is not None
            assert result.response == "I can help with that!"
            assert result.metadata["intent"] == "question"
    
    def test_sentiment_to_score_conversion(self):
        """Test sentiment to score conversion."""
        service = ConversationService()
        
        assert service._sentiment_to_score("positive") == 0.8
        assert service._sentiment_to_score("neutral") == 0.5
        assert service._sentiment_to_score("negative") == 0.2
        assert service._sentiment_to_score("unknown") == 0.5


class TestConversationAPI:
    """Test conversation API endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_conversation_endpoint(
        self,
        conversation_client: AsyncClient,
        auth_headers: dict,
        sample_conversation_data: dict
    ):
        """Test conversation creation endpoint."""
        with patch('services.conversation_service.main.conversation_service.create_conversation') as mock_create:
            mock_create.return_value = Mock(
                conversation_id="test_id",
                session_id="session_id",
                response="Hello! How can I help you?",
                metadata={"intent": "question", "sentiment": "neutral"}
            )
            
            response = await conversation_client.post(
                "/conversations",
                json=sample_conversation_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            response_data = response.json()
            assert "conversation_id" in response_data
            assert "session_id" in response_data
            assert "response" in response_data
    
    @pytest.mark.asyncio
    async def test_send_message_endpoint(
        self,
        conversation_client: AsyncClient,
        auth_headers: dict,
        sample_message_data: dict
    ):
        """Test message sending endpoint."""
        with patch('services.conversation_service.main.conversation_service.send_message') as mock_send:
            mock_send.return_value = Mock(
                message_id="msg_id",
                response="I can help with that!",
                metadata={"intent": "question", "sentiment": "neutral"}
            )
            
            response = await conversation_client.post(
                "/conversations/test_id/messages",
                json=sample_message_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            response_data = response.json()
            assert "message_id" in response_data
            assert "response" in response_data
    
    @pytest.mark.asyncio
    async def test_get_conversation_endpoint(
        self,
        conversation_client: AsyncClient,
        auth_headers: dict,
        test_session
    ):
        """Test get conversation endpoint."""
        # Create test conversation
        conversation = Conversation(
            id="test_conversation_id",
            user_id="test_user_id",
            session_id="session_id",
            status="active"
        )
        test_session.add(conversation)
        await test_session.commit()
        
        response = await conversation_client.get(
            "/conversations/test_conversation_id",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["conversation_id"] == "test_conversation_id"
        assert response_data["status"] == "active"
    
    @pytest.mark.asyncio
    async def test_unauthorized_access(
        self,
        conversation_client: AsyncClient,
        sample_conversation_data: dict
    ):
        """Test unauthorized access."""
        response = await conversation_client.post(
            "/conversations",
            json=sample_conversation_data
        )
        
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_health_check(self, conversation_client: AsyncClient):
        """Test health check endpoint."""
        response = await conversation_client.get("/health")
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "healthy"
        assert response_data["service"] == "conversation-service"
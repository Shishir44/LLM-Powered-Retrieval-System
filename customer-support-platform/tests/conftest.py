"""Test configuration and fixtures for the customer support platform."""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from shared.config.settings import Settings
from shared.database.models import Base
from shared.database.connection import AsyncSessionLocal
from shared.auth.jwt_handler import JWTHandler
from services.api_gateway.main import app as api_gateway_app
from services.conversation_service.main import app as conversation_app
from services.knowledge_base_service.main import app as knowledge_base_app


# Test database configuration
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings."""
    return Settings(
        debug=True,
        testing=True,
        database=Settings.DatabaseSettings(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_password"
        ),
        openai=Settings.OpenAISettings(
            api_key="test_openai_key",
            model="gpt-3.5-turbo",
            embedding_model="text-embedding-ada-002"
        ),
        jwt=Settings.JWTSettings(
            secret_key="test_jwt_secret",
            algorithm="HS256"
        )
    )


@pytest.fixture
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture
async def test_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    TestSessionLocal = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with TestSessionLocal() as session:
        yield session


@pytest.fixture
def jwt_handler(test_settings: Settings) -> JWTHandler:
    """Create JWT handler for testing."""
    return JWTHandler(
        secret_key=test_settings.jwt.secret_key,
        algorithm=test_settings.jwt.algorithm
    )


@pytest.fixture
def test_user_token(jwt_handler: JWTHandler) -> str:
    """Create test user token."""
    return jwt_handler.create_token(
        user_id="test_user_id",
        email="test@example.com",
        roles=["user"]
    )


@pytest.fixture
def admin_user_token(jwt_handler: JWTHandler) -> str:
    """Create admin user token."""
    return jwt_handler.create_token(
        user_id="admin_user_id",
        email="admin@example.com",
        roles=["admin"]
    )


@pytest.fixture
def auth_headers(test_user_token: str) -> dict:
    """Create authentication headers."""
    return {"Authorization": f"Bearer {test_user_token}"}


@pytest.fixture
def admin_auth_headers(admin_user_token: str) -> dict:
    """Create admin authentication headers."""
    return {"Authorization": f"Bearer {admin_user_token}"}


@pytest.fixture
async def api_gateway_client(test_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create test client for API Gateway."""
    # Override database dependency
    async def override_get_db():
        yield test_session
    
    api_gateway_app.dependency_overrides[AsyncSessionLocal] = override_get_db
    
    async with AsyncClient(app=api_gateway_app, base_url="http://test") as client:
        yield client
    
    api_gateway_app.dependency_overrides.clear()


@pytest.fixture
async def conversation_client(test_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create test client for Conversation Service."""
    # Override database dependency
    async def override_get_db():
        yield test_session
    
    conversation_app.dependency_overrides[AsyncSessionLocal] = override_get_db
    
    async with AsyncClient(app=conversation_app, base_url="http://test") as client:
        yield client
    
    conversation_app.dependency_overrides.clear()


@pytest.fixture
async def knowledge_base_client(test_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create test client for Knowledge Base Service."""
    # Override database dependency
    async def override_get_db():
        yield test_session
    
    knowledge_base_app.dependency_overrides[AsyncSessionLocal] = override_get_db
    
    async with AsyncClient(app=knowledge_base_app, base_url="http://test") as client:
        yield client
    
    knowledge_base_app.dependency_overrides.clear()


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    mock_client = Mock()
    mock_client.chat.completions.create = AsyncMock()
    mock_client.embeddings.create = AsyncMock()
    
    # Mock chat completion response
    mock_client.chat.completions.create.return_value = Mock(
        choices=[
            Mock(message=Mock(content="Test response"))
        ]
    )
    
    # Mock embedding response
    mock_client.embeddings.create.return_value = Mock(
        data=[
            Mock(embedding=[0.1] * 1536)
        ]
    )
    
    return mock_client


@pytest.fixture
def mock_vector_store():
    """Mock vector store."""
    mock_store = Mock()
    mock_store.add_documents = AsyncMock()
    mock_store.similarity_search_with_score = AsyncMock()
    mock_store.delete = AsyncMock()
    
    # Mock search results
    mock_store.similarity_search_with_score.return_value = [
        (Mock(page_content="Test content", metadata={"id": "test_id"}), 0.9)
    ]
    
    return mock_store


@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collector."""
    mock_collector = Mock()
    mock_collector.record_conversation_created = Mock()
    mock_collector.record_message_sent = Mock()
    mock_collector.record_document_created = Mock()
    mock_collector.record_search_performed = Mock()
    
    return mock_collector


# Sample data fixtures
@pytest.fixture
def sample_conversation_data():
    """Sample conversation data for testing."""
    return {
        "message": "Hello, I need help with my account",
        "metadata": {
            "channel": "web",
            "user_agent": "test-agent"
        }
    }


@pytest.fixture
def sample_message_data():
    """Sample message data for testing."""
    return {
        "message": "I can't log into my account",
        "metadata": {
            "timestamp": "2023-01-01T00:00:00Z"
        }
    }


@pytest.fixture
def sample_document_data():
    """Sample document data for testing."""
    return {
        "title": "How to reset your password",
        "content": "To reset your password, follow these steps: 1. Go to the login page 2. Click 'Forgot Password' 3. Enter your email address",
        "category": "account",
        "subcategory": "password",
        "tags": ["password", "reset", "account"],
        "metadata": {
            "author": "support_team",
            "difficulty": "easy"
        }
    }


@pytest.fixture
def sample_search_data():
    """Sample search data for testing."""
    return {
        "query": "reset password",
        "category": "account",
        "limit": 5
    }


# Utility functions for tests
def assert_response_success(response, expected_status: int = 200):
    """Assert that response is successful."""
    assert response.status_code == expected_status
    assert response.json() is not None


def assert_response_error(response, expected_status: int = 400):
    """Assert that response is an error."""
    assert response.status_code == expected_status
    response_data = response.json()
    assert "detail" in response_data


def assert_conversation_response(response_data: dict):
    """Assert conversation response structure."""
    assert "conversation_id" in response_data
    assert "session_id" in response_data
    assert "response" in response_data
    assert "metadata" in response_data
    assert isinstance(response_data["metadata"], dict)


def assert_message_response(response_data: dict):
    """Assert message response structure."""
    assert "message_id" in response_data
    assert "response" in response_data
    assert "metadata" in response_data


def assert_document_response(response_data: dict):
    """Assert document response structure."""
    assert "id" in response_data
    assert "title" in response_data
    assert "category" in response_data
    assert "tags" in response_data
    assert isinstance(response_data["tags"], list)


def assert_search_response(response_data: dict):
    """Assert search response structure."""
    assert "results" in response_data
    assert "total" in response_data
    assert "query" in response_data
    assert isinstance(response_data["results"], list)
    assert isinstance(response_data["total"], int)


# Test markers
pytestmark = pytest.mark.asyncio
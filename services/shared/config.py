"""
Shared configuration management for all services
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator


class BaseConfig(BaseSettings):
    """Base configuration with common settings"""
    
    # Environment
    environment: str = "development"
    debug: bool = True
    
    # Security
    secret_key: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    cors_credentials: bool = True
    cors_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_headers: List[str] = ["*"]
    
    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "llm_retrieval"
    postgres_user: str = "postgres"
    postgres_password: str = "password"
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_url: Optional[str] = None
    
    # External APIs
    openai_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: Optional[str] = None
    
    # Vector Store
    vector_store_type: str = "pinecone"
    
    # Monitoring
    jaeger_endpoint: Optional[str] = None
    prometheus_enabled: bool = True
    
    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        allowed = ['development', 'staging', 'production']
        if v not in allowed:
            raise ValueError(f'Environment must be one of: {allowed}')
        return v
    
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def redis_connection_url(self) -> str:
        if self.redis_url:
            return self.redis_url
        return f"redis://{self.redis_host}:{self.redis_port}"
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
    
    model_config = {"env_file": ".env", "case_sensitive": False}


class APIGatewayConfig(BaseConfig):
    """API Gateway specific configuration"""
    service_name: str = "api-gateway"
    host: str = "0.0.0.0"
    port: int = 8080
    
    # Service URLs
    conversation_service_url: str = "http://localhost:8001"
    knowledge_base_service_url: str = "http://localhost:8002"
    analytics_service_url: str = "http://localhost:8005"
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Timeouts
    service_timeout: float = 30.0
    connection_timeout: float = 5.0


class ConversationConfig(BaseConfig):
    """Conversation service specific configuration"""
    service_name: str = "conversation-service"
    host: str = "0.0.0.0"
    port: int = 8001
    
    # RAG Configuration
    max_tokens: int = 4000
    temperature: float = 0.7
    top_k: int = 5
    context_window_size: int = 10
    
    # Service dependencies
    knowledge_base_service_url: str = "http://localhost:8002"
    analytics_service_url: str = "http://localhost:8005"


class KnowledgeBaseConfig(BaseConfig):
    """Knowledge base service specific configuration"""
    service_name: str = "knowledge-base-service"
    host: str = "0.0.0.0"
    port: int = 8002
    
    # Chunking configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks_per_doc: int = 100
    
    # Vector search
    similarity_threshold: float = 0.7
    max_results: int = 10
    
    # Cache settings
    cache_ttl: int = 3600  # seconds


class AnalyticsConfig(BaseConfig):
    """Analytics service specific configuration"""
    service_name: str = "analytics-service"
    host: str = "0.0.0.0"
    port: int = 8005
    
    # Metrics retention
    metrics_retention_days: int = 30
    batch_size: int = 100
    
    # Evaluation thresholds
    relevance_threshold: float = 0.7
    precision_threshold: float = 0.8


def get_config(service_type: str) -> BaseConfig:
    """Factory function to get appropriate config based on service type"""
    configs = {
        'api-gateway': APIGatewayConfig,
        'conversation': ConversationConfig,
        'knowledge-base': KnowledgeBaseConfig,
        'analytics': AnalyticsConfig,
    }
    
    config_class = configs.get(service_type, BaseConfig)
    return config_class()
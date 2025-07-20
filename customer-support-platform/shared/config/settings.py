"""Configuration settings for the customer support platform."""

from typing import Any, Dict, List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from functools import lru_cache


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    host: str = Field(default="localhost", env="POSTGRES_HOST")
    port: int = Field(default=5432, env="POSTGRES_PORT")
    database: str = Field(default="customer_support", env="POSTGRES_DB")
    username: str = Field(default="postgres", env="POSTGRES_USER")
    password: str = Field(default="password", env="POSTGRES_PASSWORD")
    
    @property
    def database_url(self) -> str:
        """Generate database URL."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def async_database_url(self) -> str:
        """Generate async database URL."""
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")
    
    @property
    def redis_url(self) -> str:
        """Generate Redis URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class OpenAISettings(BaseSettings):
    """OpenAI API configuration settings."""
    
    api_key: str = Field(env="OPENAI_API_KEY")
    model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    embedding_model: str = Field(default="text-embedding-3-large", env="OPENAI_EMBEDDING_MODEL")
    max_tokens: int = Field(default=2000, env="OPENAI_MAX_TOKENS")
    temperature: float = Field(default=0.1, env="OPENAI_TEMPERATURE")


class VectorStoreSettings(BaseSettings):
    """Vector store configuration settings."""
    
    type: str = Field(default="pinecone", env="VECTOR_STORE_TYPE")
    
    # Pinecone settings
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(default=None, env="PINECONE_ENVIRONMENT")
    pinecone_index_name: Optional[str] = Field(default=None, env="PINECONE_INDEX_NAME")
    
    # Weaviate settings
    weaviate_url: Optional[str] = Field(default=None, env="WEAVIATE_URL")
    weaviate_api_key: Optional[str] = Field(default=None, env="WEAVIATE_API_KEY")
    
    @validator("type")
    def validate_vector_store_type(cls, v):
        """Validate vector store type."""
        allowed_types = ["pinecone", "weaviate", "chroma"]
        if v not in allowed_types:
            raise ValueError(f"Vector store type must be one of: {allowed_types}")
        return v


class JWTSettings(BaseSettings):
    """JWT configuration settings."""
    
    secret_key: str = Field(env="JWT_SECRET_KEY")
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    grafana_port: int = Field(default=3000, env="GRAFANA_PORT")
    jaeger_endpoint: str = Field(default="http://localhost:14268/api/traces", env="JAEGER_ENDPOINT")
    
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return v.split(",")
        return v


class ServiceSettings(BaseSettings):
    """Service URLs and configuration."""
    
    conversation_service_url: str = Field(default="http://localhost:8001", env="CONVERSATION_SERVICE_URL")
    knowledge_base_service_url: str = Field(default="http://localhost:8002", env="KNOWLEDGE_BASE_SERVICE_URL")
    nlp_service_url: str = Field(default="http://localhost:8003", env="NLP_SERVICE_URL")
    integration_service_url: str = Field(default="http://localhost:8004", env="INTEGRATION_SERVICE_URL")
    analytics_service_url: str = Field(default="http://localhost:8005", env="ANALYTICS_SERVICE_URL")


class FeatureFlags(BaseSettings):
    """Feature flags configuration."""
    
    enable_voice_processing: bool = Field(default=True, env="ENABLE_VOICE_PROCESSING")
    enable_multi_language: bool = Field(default=True, env="ENABLE_MULTI_LANGUAGE")
    enable_analytics: bool = Field(default=True, env="ENABLE_ANALYTICS")
    enable_audit_logging: bool = Field(default=True, env="ENABLE_AUDIT_LOGGING")


class Settings(BaseSettings):
    """Main settings class that combines all configuration."""
    
    # Environment
    debug: bool = Field(default=False, env="DEBUG")
    testing: bool = Field(default=False, env="TESTING")
    development: bool = Field(default=True, env="DEVELOPMENT")
    
    # Service configuration
    service_name: str = Field(default="customer-support-platform")
    service_version: str = Field(default="1.0.0")
    
    # Sub-settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    openai: OpenAISettings = OpenAISettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    jwt: JWTSettings = JWTSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    security: SecuritySettings = SecuritySettings()
    services: ServiceSettings = ServiceSettings()
    features: FeatureFlags = FeatureFlags()
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Service-specific settings
class APIGatewaySettings(Settings):
    """API Gateway specific settings."""
    
    host: str = Field(default="0.0.0.0", env="API_GATEWAY_HOST")
    port: int = Field(default=8080, env="API_GATEWAY_PORT")
    workers: int = Field(default=4, env="API_GATEWAY_WORKERS")


class ConversationServiceSettings(Settings):
    """Conversation service specific settings."""
    
    host: str = Field(default="0.0.0.0", env="CONVERSATION_SERVICE_HOST")
    port: int = Field(default=8001, env="CONVERSATION_SERVICE_PORT")
    max_conversation_length: int = Field(default=50, env="MAX_CONVERSATION_LENGTH")
    session_timeout: int = Field(default=3600, env="SESSION_TIMEOUT")


class KnowledgeBaseServiceSettings(Settings):
    """Knowledge base service specific settings."""
    
    host: str = Field(default="0.0.0.0", env="KNOWLEDGE_BASE_SERVICE_HOST")
    port: int = Field(default=8002, env="KNOWLEDGE_BASE_SERVICE_PORT")
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    max_search_results: int = Field(default=10, env="MAX_SEARCH_RESULTS")


class NLPServiceSettings(Settings):
    """NLP service specific settings."""
    
    host: str = Field(default="0.0.0.0", env="NLP_SERVICE_HOST")
    port: int = Field(default=8003, env="NLP_SERVICE_PORT")
    sentiment_threshold: float = Field(default=0.5, env="SENTIMENT_THRESHOLD")
    spam_threshold: float = Field(default=0.8, env="SPAM_THRESHOLD")


class IntegrationServiceSettings(Settings):
    """Integration service specific settings."""
    
    host: str = Field(default="0.0.0.0", env="INTEGRATION_SERVICE_HOST")
    port: int = Field(default=8004, env="INTEGRATION_SERVICE_PORT")
    webhook_timeout: int = Field(default=30, env="WEBHOOK_TIMEOUT")
    retry_attempts: int = Field(default=3, env="RETRY_ATTEMPTS")


class AnalyticsServiceSettings(Settings):
    """Analytics service specific settings."""
    
    host: str = Field(default="0.0.0.0", env="ANALYTICS_SERVICE_HOST")
    port: int = Field(default=8005, env="ANALYTICS_SERVICE_PORT")
    metrics_retention_days: int = Field(default=30, env="METRICS_RETENTION_DAYS")
    batch_size: int = Field(default=100, env="BATCH_SIZE")


# Factory functions for service-specific settings
@lru_cache()
def get_api_gateway_settings() -> APIGatewaySettings:
    """Get API Gateway settings."""
    return APIGatewaySettings()


@lru_cache()
def get_conversation_service_settings() -> ConversationServiceSettings:
    """Get Conversation service settings."""
    return ConversationServiceSettings()


@lru_cache()
def get_knowledge_base_service_settings() -> KnowledgeBaseServiceSettings:
    """Get Knowledge Base service settings."""
    return KnowledgeBaseServiceSettings()


@lru_cache()
def get_nlp_service_settings() -> NLPServiceSettings:
    """Get NLP service settings."""
    return NLPServiceSettings()


@lru_cache()
def get_integration_service_settings() -> IntegrationServiceSettings:
    """Get Integration service settings."""
    return IntegrationServiceSettings()


@lru_cache()
def get_analytics_service_settings() -> AnalyticsServiceSettings:
    """Get Analytics service settings."""
    return AnalyticsServiceSettings()
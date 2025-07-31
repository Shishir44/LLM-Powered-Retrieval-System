"""
Shared configuration management for all services
"""
import os
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import field_validator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


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
    
    # RAG Configuration - optimized for accuracy
    max_tokens: int = 2000  # Reduce for more focused responses
    temperature: float = 0.3  # Lower for more factual responses
    top_k: int = 3  # Fewer sources for clarity
    context_window_size: int = 5  # Smaller context for focus
    
    # Service dependencies
    knowledge_base_service_url: str = "http://localhost:8002"
    analytics_service_url: str = "http://localhost:8005"


class KnowledgeBaseConfig(BaseConfig):
    """Knowledge base service specific configuration"""
    service_name: str = "knowledge-base-service"
    host: str = "0.0.0.0"
    port: int = 8002
    
    # Chunking configuration - optimized for better context
    chunk_size: int = 768  # Optimal for most embedding models
    chunk_overlap: int = 128  # Better context retention
    max_chunks_per_doc: int = 50  # Reduce for performance
    
    # Vector search - optimized for accuracy
    similarity_threshold: float = 0.75  # Higher threshold for better relevance
    max_results: int = 5  # Fewer results for focus
    
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


@dataclass
class AdvancedReasoningConfig(BaseConfig):
    """PHASE 3.1: Advanced reasoning configuration for complex query processing."""
    
    # Core reasoning settings
    enable_advanced_reasoning: bool = True
    enable_chain_of_thought: bool = True
    enable_multi_hop_processing: bool = True
    enable_multi_domain_analysis: bool = True
    
    # LLM settings for reasoning
    reasoning_model: str = "gpt-4"
    reasoning_temperature: float = 0.1  # Low for consistent reasoning
    max_reasoning_tokens: int = 2000
    
    # Query complexity thresholds
    simple_query_word_limit: int = 10
    complex_query_indicators: List[str] = field(default_factory=lambda: [
        "why", "how", "explain", "analyze", "compare", "troubleshoot", "steps"
    ])
    multi_domain_indicators: List[str] = field(default_factory=lambda: [
        "billing and", "account and", "policy and", "technical and"
    ])
    
    # Reasoning chain limits
    max_reasoning_steps: int = 5
    min_confidence_threshold: float = 0.6
    evidence_strength_threshold: float = 0.7
    
    # Performance settings
    reasoning_timeout_seconds: float = 30.0
    enable_reasoning_cache: bool = True
    cache_ttl_minutes: int = 15
    
    # Quality control
    enable_reasoning_validation: bool = True
    fallback_to_simple_on_failure: bool = True
    quality_threshold: float = 0.6

@dataclass 
class EnhancedRAGConfig(BaseConfig):
    """Enhanced RAG configuration - PHASE 3.1: With Advanced Reasoning."""
    
    # PHASE 1.4: Simplified Context Management for accuracy
    context_strategy: str = "recency_based"  # Simple recency strategy
    max_context_messages: int = 5  # Reduce further for focus
    context_compression_threshold: int = 5  # Earlier compression
    context_relevance_threshold: float = 0.8  # Higher relevance threshold for quality
    
    # PHASE 1.4: Disable Fact Verification (complex feature) - BUT enable reasoning
    enable_fact_verification: bool = False  # Keep disabled for accuracy
    verification_confidence_threshold: float = 0.8
    max_claims_per_response: int = 3
    hallucination_risk_threshold: float = 0.3  # More conservative
    
    # PHASE 3.1: Enable Advanced Reasoning (controlled re-enablement)
    enable_advanced_reasoning: bool = True  # NEW: Controlled re-enablement
    reasoning_config: AdvancedReasoningConfig = field(default_factory=AdvancedReasoningConfig)
    
    # PHASE 1.4: Keep Multi-Source Synthesis disabled for now
    enable_multi_source_synthesis: bool = False  # Keep disabled
    synthesis_confidence_threshold: float = 0.8
    max_sources_per_synthesis: int = 3
    synthesis_overlap_threshold: float = 0.7
    
    # PHASE 3.1: Enable Multi-Hop Processing (with safeguards)
    enable_multi_hop_processing: bool = True  # NEW: Re-enabled with reasoning
    multi_hop_max_depth: int = 3  # Limit depth for control
    multi_hop_confidence_threshold: float = 0.7
    
    # PHASE 1.4: Keep Customer Context disabled
    enable_customer_context: bool = False  # Keep disabled
    context_personalization_threshold: float = 0.6
    max_context_history_days: int = 30
    context_privacy_mode: bool = True
    
    # Core retrieval settings (Phase 2 enhanced)
    final_retrieval_top_k: int = 5
    min_response_quality: float = 4.5
    max_processing_time: float = 15.0  # Increased for reasoning
    
    # PHASE 3.1: Reasoning integration settings
    reasoning_fallback_enabled: bool = True
    reasoning_quality_threshold: float = 0.6
    complex_query_threshold: int = 2  # Number of complexity indicators needed

@dataclass
class ChunkingConfig(BaseConfig):
    """Document chunking configuration - PHASE 1.2 optimized."""
    max_chunk_size: int = 1024  # PHASE 1.2: Increased for better context
    chunk_overlap: int = 256    # PHASE 1.2: Better overlap for context retention
    min_chunk_size: int = 100
    enable_semantic_chunking: bool = True
    similarity_threshold: float = 0.8


def get_enhanced_config() -> EnhancedRAGConfig:
    """Get enhanced configuration with environment overrides."""
    
    config = EnhancedRAGConfig()
    
    # Override with environment variables
    config.context_strategy = os.getenv("CONTEXT_STRATEGY", config.context_strategy)
    config.enable_advanced_reasoning = os.getenv("ENABLE_ADVANCED_REASONING", "true").lower() == "true"
    config.enable_multi_hop_processing = os.getenv("ENABLE_MULTI_HOP_PROCESSING", "true").lower() == "true"
    
    return config

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import logging

@dataclass
class VectorDatabaseConfig:
    """Vector database configuration."""
    type: str = "chroma"
    
    # ChromaDB settings
    chroma_persist_directory: str = "./data/chroma"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    
    # Pinecone settings
    pinecone_api_key: str = ""
    pinecone_environment: str = ""
    pinecone_index_name: str = "rag-system-enhanced"
    
    # Weaviate settings
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: str = ""

@dataclass
class EmbeddingConfig:
    """Embedding models configuration."""
    primary_model: str = "all-mpnet-base-v2"
    use_openai_embeddings: bool = True
    openai_embedding_model: str = "text-embedding-3-large"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    spacy_model: str = "en_core_web_sm"

@dataclass
class RetrievalConfig:
    """Retrieval pipeline configuration."""
    initial_retrieval_size: int = 50
    rerank_size: int = 20
    diversity_selection_size: int = 10
    final_results_size: int = 5
    
    similarity_threshold: float = 0.8
    rerank_threshold: float = 0.5
    diversity_threshold: float = 0.7
    
    query_expansion_limit: int = 3
    enable_query_expansion: bool = True

@dataclass
class ChunkingConfig:
    """Chunking configuration - PHASE 1.2 optimized."""
    strategy: str = "recursive"
    max_chunk_size: int = 1024  # PHASE 1.2: Better context
    chunk_overlap: int = 256    # PHASE 1.2: Better overlap
    min_chunk_size: int = 100
    respect_sentence_boundaries: bool = True
    similarity_threshold: float = 0.8
    enable_smart_chunking: bool = True

@dataclass
class ResponseConfig:
    """Response generation configuration."""
    use_structured_templates: bool = True
    default_template_type: str = "analytical"
    synthesis_strategy: str = "auto"
    enable_multi_source_synthesis: bool = True
    max_sources_for_synthesis: int = 10
    
    quality_threshold: float = 3.5
    enable_auto_improvement: bool = True
    max_improvement_iterations: int = 2

@dataclass
class CacheConfig:
    """Caching configuration."""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0
    
    enable_query_cache: bool = True
    enable_embedding_cache: bool = True
    cache_ttl: int = 3600

@dataclass
class ServiceConfig:
    """Service endpoints configuration."""
    knowledge_base_host: str = "localhost"
    knowledge_base_port: int = 8002
    
    conversation_host: str = "localhost"
    conversation_port: int = 8001
    
    analytics_host: str = "localhost"
    analytics_port: int = 8005

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    log_level: str = "INFO"
    log_format: str = "json"
    enable_structured_logging: bool = True
    
    enable_performance_tracking: bool = True
    benchmark_mode: bool = False

@dataclass
class ProductionConfig:
    """Production environment configuration."""
    environment: str = "development"
    debug: bool = True
    
    enable_rate_limiting: bool = False
    requests_per_minute: int = 60
    
    enable_cors: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000", "http://localhost:8501"])
    
    database_url: str = ""
    enable_database_logging: bool = False

@dataclass
class FeatureFlags:
    """Feature flags for enabling/disabling functionality."""
    enable_advanced_chunking: bool = True
    enable_multi_stage_retrieval: bool = True
    enable_cross_encoder_reranking: bool = True
    enable_diversity_selection: bool = True
    enable_response_templates: bool = True
    enable_multi_source_synthesis: bool = True
    
    enable_ab_testing: bool = False
    ab_test_traffic_split: float = 0.5
    
    enable_experimental_features: bool = False
    experimental_embedding_model: str = ""
    experimental_retrieval_strategy: str = ""

@dataclass
class DevelopmentConfig:
    """Development-specific configuration."""
    enable_hot_reload: bool = True
    enable_debug_endpoints: bool = True
    enable_admin_interface: bool = False
    
    load_sample_data: bool = True
    sample_data_path: str = "./data/sample_documents.json"
    
    enable_mock_services: bool = False
    test_data_path: str = "./tests/data/"

@dataclass
class EnhancedRAGConfig:
    """Complete enhanced RAG system configuration."""
    # API Keys - Multiple LLM Providers
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    
    # Gemini Configuration
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-pro"
    
    # Anthropic Configuration
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    
    # LLM Provider Fallback Configuration
    primary_llm_provider: str = "gemini"  # primary provider - using Gemini due to OpenAI quota
    fallback_providers: List[str] = field(default_factory=lambda: ["openai", "anthropic"])
    enable_fallback: bool = True
    fallback_timeout: float = 5.0  # seconds before trying fallback
    
    # Component configurations
    vector_db: VectorDatabaseConfig = field(default_factory=VectorDatabaseConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    response: ResponseConfig = field(default_factory=ResponseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    services: ServiceConfig = field(default_factory=ServiceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    production: ProductionConfig = field(default_factory=ProductionConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)

class ConfigManager:
    """Enhanced configuration manager for the RAG system."""
    
    def __init__(self, config_path: Optional[str] = None, env_file: Optional[str] = None):
        self.config_path = config_path
        self.env_file = env_file or ".env"
        self.config = EnhancedRAGConfig()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from environment variables and config files."""
        
        # Load from environment file if it exists
        self._load_from_env_file()
        
        # Load from environment variables
        self._load_from_env_vars()
        
        # Load from config file if specified
        if self.config_path:
            self._load_from_config_file()
        
        # Validate configuration
        self._validate_configuration()
        
        self.logger.info("Configuration loaded successfully")
    
    def _load_from_env_file(self):
        """Load configuration from .env file."""
        env_path = Path(self.env_file)
        
        if not env_path.exists():
            self.logger.warning(f"Environment file {self.env_file} not found")
            return
        
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            self.logger.info(f"Loaded environment from {self.env_file}")
        except ImportError:
            self.logger.warning("python-dotenv not installed, skipping .env file loading")
        except Exception as e:
            self.logger.error(f"Error loading .env file: {e}")
    
    def _load_from_env_vars(self):
        """Load configuration from environment variables."""
        
        # API Keys - Multiple LLM Providers
        self.config.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.config.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        
        # Gemini Configuration
        self.config.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.config.gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        
        # Anthropic Configuration
        self.config.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.config.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        
        # LLM Provider Fallback Configuration
        self.config.primary_llm_provider = os.getenv("PRIMARY_LLM_PROVIDER", "gemini")
        fallback_providers = os.getenv("FALLBACK_PROVIDERS", "openai,anthropic")
        self.config.fallback_providers = [provider.strip() for provider in fallback_providers.split(",")]
        self.config.enable_fallback = os.getenv("ENABLE_FALLBACK", "true").lower() == "true"
        self.config.fallback_timeout = float(os.getenv("FALLBACK_TIMEOUT", "5.0"))
        
        # Vector Database
        self.config.vector_db.type = os.getenv("VECTOR_DB_TYPE", "chroma").lower()
        self.config.vector_db.chroma_persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma")
        self.config.vector_db.chroma_host = os.getenv("CHROMA_HOST", "localhost")
        self.config.vector_db.chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        
        self.config.vector_db.pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
        self.config.vector_db.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "")
        self.config.vector_db.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "rag-system-enhanced")
        
        self.config.vector_db.weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        self.config.vector_db.weaviate_api_key = os.getenv("WEAVIATE_API_KEY", "")
        
        # Embeddings
        self.config.embeddings.primary_model = os.getenv("PRIMARY_EMBEDDING_MODEL", "all-mpnet-base-v2")
        self.config.embeddings.use_openai_embeddings = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
        self.config.embeddings.openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        self.config.embeddings.cross_encoder_model = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
        self.config.embeddings.spacy_model = os.getenv("SPACY_MODEL", "en_core_web_sm")
        
        # Retrieval
        self.config.retrieval.initial_retrieval_size = int(os.getenv("INITIAL_RETRIEVAL_SIZE", "50"))
        self.config.retrieval.rerank_size = int(os.getenv("RERANK_SIZE", "20"))
        self.config.retrieval.diversity_selection_size = int(os.getenv("DIVERSITY_SELECTION_SIZE", "10"))
        self.config.retrieval.final_results_size = int(os.getenv("FINAL_RESULTS_SIZE", "5"))
        
        self.config.retrieval.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))
        self.config.retrieval.rerank_threshold = float(os.getenv("RERANK_THRESHOLD", "0.5"))
        self.config.retrieval.diversity_threshold = float(os.getenv("DIVERSITY_THRESHOLD", "0.7"))
        
        self.config.retrieval.query_expansion_limit = int(os.getenv("QUERY_EXPANSION_LIMIT", "3"))
        self.config.retrieval.enable_query_expansion = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
        
        # Chunking
        self.config.chunking.strategy = os.getenv("CHUNKING_STRATEGY", "semantic_structure")
        self.config.chunking.max_chunk_size = int(os.getenv("MAX_CHUNK_SIZE", "1024"))
        # PHASE 1.2: Update chunking configuration for better accuracy
        self.config.chunking.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "256"))
        self.config.chunking.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD_CHUNKING", "0.8"))
        
        # Response
        self.config.response.use_structured_templates = os.getenv("USE_STRUCTURED_TEMPLATES", "true").lower() == "true"
        self.config.response.default_template_type = os.getenv("DEFAULT_TEMPLATE_TYPE", "analytical")
        self.config.response.synthesis_strategy = os.getenv("SYNTHESIS_STRATEGY", "auto")
        self.config.response.enable_multi_source_synthesis = os.getenv("ENABLE_MULTI_SOURCE_SYNTHESIS", "true").lower() == "true"
        self.config.response.max_sources_for_synthesis = int(os.getenv("MAX_SOURCES_FOR_SYNTHESIS", "10"))
        
        self.config.response.quality_threshold = float(os.getenv("QUALITY_THRESHOLD", "3.5"))
        self.config.response.enable_auto_improvement = os.getenv("ENABLE_AUTO_IMPROVEMENT", "true").lower() == "true"
        self.config.response.max_improvement_iterations = int(os.getenv("MAX_IMPROVEMENT_ITERATIONS", "2"))
        
        # Cache
        self.config.cache.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.config.cache.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.config.cache.redis_password = os.getenv("REDIS_PASSWORD", "")
        self.config.cache.redis_db = int(os.getenv("REDIS_DB", "0"))
        
        self.config.cache.enable_query_cache = os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true"
        self.config.cache.enable_embedding_cache = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
        self.config.cache.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
        
        # Services
        self.config.services.knowledge_base_host = os.getenv("KNOWLEDGE_BASE_SERVICE_HOST", "localhost")
        self.config.services.knowledge_base_port = int(os.getenv("KNOWLEDGE_BASE_SERVICE_PORT", "8002"))
        self.config.services.conversation_host = os.getenv("CONVERSATION_SERVICE_HOST", "localhost")
        self.config.services.conversation_port = int(os.getenv("CONVERSATION_SERVICE_PORT", "8001"))
        self.config.services.analytics_host = os.getenv("ANALYTICS_SERVICE_HOST", "localhost")
        self.config.services.analytics_port = int(os.getenv("ANALYTICS_SERVICE_PORT", "8005"))
        
        # Monitoring
        self.config.monitoring.enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        self.config.monitoring.metrics_port = int(os.getenv("METRICS_PORT", "9090"))
        self.config.monitoring.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.config.monitoring.log_format = os.getenv("LOG_FORMAT", "json")
        self.config.monitoring.enable_structured_logging = os.getenv("ENABLE_STRUCTURED_LOGGING", "true").lower() == "true"
        self.config.monitoring.enable_performance_tracking = os.getenv("ENABLE_PERFORMANCE_TRACKING", "true").lower() == "true"
        self.config.monitoring.benchmark_mode = os.getenv("BENCHMARK_MODE", "false").lower() == "true"
        
        # Production
        self.config.production.environment = os.getenv("ENVIRONMENT", "development")
        self.config.production.debug = os.getenv("DEBUG", "true").lower() == "true"
        self.config.production.enable_rate_limiting = os.getenv("ENABLE_RATE_LIMITING", "false").lower() == "true"
        self.config.production.requests_per_minute = int(os.getenv("REQUESTS_PER_MINUTE", "60"))
        self.config.production.enable_cors = os.getenv("ENABLE_CORS", "true").lower() == "true"
        
        allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501")
        self.config.production.allowed_origins = [origin.strip() for origin in allowed_origins.split(",")]
        
        self.config.production.database_url = os.getenv("DATABASE_URL", "")
        self.config.production.enable_database_logging = os.getenv("ENABLE_DATABASE_LOGGING", "false").lower() == "true"
        
        # Feature Flags
        self.config.features.enable_advanced_chunking = os.getenv("ENABLE_ADVANCED_CHUNKING", "true").lower() == "true"
        self.config.features.enable_multi_stage_retrieval = os.getenv("ENABLE_MULTI_STAGE_RETRIEVAL", "true").lower() == "true"
        self.config.features.enable_cross_encoder_reranking = os.getenv("ENABLE_CROSS_ENCODER_RERANKING", "true").lower() == "true"
        self.config.features.enable_diversity_selection = os.getenv("ENABLE_DIVERSITY_SELECTION", "true").lower() == "true"
        self.config.features.enable_response_templates = os.getenv("ENABLE_RESPONSE_TEMPLATES", "true").lower() == "true"
        self.config.features.enable_multi_source_synthesis = os.getenv("ENABLE_MULTI_SOURCE_SYNTHESIS", "true").lower() == "true"
        
        self.config.features.enable_ab_testing = os.getenv("ENABLE_AB_TESTING", "false").lower() == "true"
        self.config.features.ab_test_traffic_split = float(os.getenv("AB_TEST_TRAFFIC_SPLIT", "0.5"))
        
        self.config.features.enable_experimental_features = os.getenv("ENABLE_EXPERIMENTAL_FEATURES", "false").lower() == "true"
        self.config.features.experimental_embedding_model = os.getenv("EXPERIMENTAL_EMBEDDING_MODEL", "")
        self.config.features.experimental_retrieval_strategy = os.getenv("EXPERIMENTAL_RETRIEVAL_STRATEGY", "")
        
        # Development
        self.config.development.enable_hot_reload = os.getenv("ENABLE_HOT_RELOAD", "true").lower() == "true"
        self.config.development.enable_debug_endpoints = os.getenv("ENABLE_DEBUG_ENDPOINTS", "true").lower() == "true"
        self.config.development.enable_admin_interface = os.getenv("ENABLE_ADMIN_INTERFACE", "false").lower() == "true"
        self.config.development.load_sample_data = os.getenv("LOAD_SAMPLE_DATA", "true").lower() == "true"
        self.config.development.sample_data_path = os.getenv("SAMPLE_DATA_PATH", "./data/sample_documents.json")
        self.config.development.enable_mock_services = os.getenv("ENABLE_MOCK_SERVICES", "false").lower() == "true"
        self.config.development.test_data_path = os.getenv("TEST_DATA_PATH", "./tests/data/")
    
    def _load_from_config_file(self):
        """Load configuration from JSON config file."""
        config_path = Path(self.config_path)
        
        if not config_path.exists():
            self.logger.warning(f"Config file {self.config_path} not found")
            return
        
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            
            # Update configuration with values from file
            self._update_config_from_dict(file_config)
            
            self.logger.info(f"Loaded configuration from {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading config file: {e}")
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        # This would recursively update the config object
        # Implementation would depend on specific needs
        pass
    
    def _validate_configuration(self):
        """Validate the loaded configuration."""
        errors = []
        
        # Check required API keys
        if not self.config.openai_api_key and self.config.embeddings.use_openai_embeddings:
            errors.append("OpenAI API key is required when using OpenAI embeddings")
        
        # Validate vector database configuration
        if self.config.vector_db.type == "pinecone":
            if not self.config.vector_db.pinecone_api_key:
                errors.append("Pinecone API key is required when using Pinecone")
            if not self.config.vector_db.pinecone_environment:
                errors.append("Pinecone environment is required when using Pinecone")
        
        if self.config.vector_db.type == "weaviate":
            if not self.config.vector_db.weaviate_url:
                errors.append("Weaviate URL is required when using Weaviate")
        
        # Validate thresholds
        if not 0.0 <= self.config.retrieval.similarity_threshold <= 1.0:
            errors.append("Similarity threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.config.retrieval.rerank_threshold <= 1.0:
            errors.append("Rerank threshold must be between 0.0 and 1.0")
        
        if not 1.0 <= self.config.response.quality_threshold <= 5.0:
            errors.append("Quality threshold must be between 1.0 and 5.0")
        
        # Validate sizes
        if self.config.retrieval.final_results_size > self.config.retrieval.diversity_selection_size:
            errors.append("Final results size cannot be larger than diversity selection size")
        
        if self.config.retrieval.diversity_selection_size > self.config.retrieval.rerank_size:
            errors.append("Diversity selection size cannot be larger than rerank size")
        
        if self.config.retrieval.rerank_size > self.config.retrieval.initial_retrieval_size:
            errors.append("Rerank size cannot be larger than initial retrieval size")
        
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(f"- {error}" for error in errors)
            self.logger.error(error_msg)
            raise ValueError(error_msg)
    
    def get_vector_db_config(self) -> Dict[str, Any]:
        """Get vector database configuration as dictionary."""
        config = {
            "type": self.config.vector_db.type
        }
        
        if self.config.vector_db.type == "chroma":
            config.update({
                "persist_directory": self.config.vector_db.chroma_persist_directory,
                "host": self.config.vector_db.chroma_host,
                "port": self.config.vector_db.chroma_port
            })
        elif self.config.vector_db.type == "pinecone":
            config.update({
                "api_key": self.config.vector_db.pinecone_api_key,
                "environment": self.config.vector_db.pinecone_environment,
                "index_name": self.config.vector_db.pinecone_index_name
            })
        elif self.config.vector_db.type == "weaviate":
            config.update({
                "url": self.config.vector_db.weaviate_url,
                "auth": {"api_key": self.config.vector_db.weaviate_api_key} if self.config.vector_db.weaviate_api_key else {}
            })
        
        return config
    
    def get_service_url(self, service_name: str) -> str:
        """Get service URL by service name."""
        if service_name == "knowledge_base":
            return f"http://{self.config.services.knowledge_base_host}:{self.config.services.knowledge_base_port}"
        elif service_name == "conversation":
            return f"http://{self.config.services.conversation_host}:{self.config.services.conversation_port}"
        elif service_name == "analytics":
            return f"http://{self.config.services.analytics_host}:{self.config.services.analytics_port}"
        else:
            raise ValueError(f"Unknown service: {service_name}")
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.config.production.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.config.production.environment.lower() == "development"
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get all feature flags as dictionary."""
        return {
            "enable_advanced_chunking": self.config.features.enable_advanced_chunking,
            "enable_multi_stage_retrieval": self.config.features.enable_multi_stage_retrieval,
            "enable_cross_encoder_reranking": self.config.features.enable_cross_encoder_reranking,
            "enable_diversity_selection": self.config.features.enable_diversity_selection,
            "enable_response_templates": self.config.features.enable_response_templates,
            "enable_multi_source_synthesis": self.config.features.enable_multi_source_synthesis,
            "enable_ab_testing": self.config.features.enable_ab_testing,
            "enable_experimental_features": self.config.features.enable_experimental_features
        }
    
    def save_config(self, output_path: str):
        """Save current configuration to file."""
        try:
            config_dict = self._config_to_dict()
            
            with open(output_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            self.logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        # This would serialize the config dataclass to dict
        # Implementation would depend on specific serialization needs
        import dataclasses
        return dataclasses.asdict(self.config)
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        try:
            self._update_config_from_dict(updates)
            self._validate_configuration()
            self.logger.info("Configuration updated successfully")
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            raise
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            "level": self.config.monitoring.log_level,
            "format": self.config.monitoring.log_format,
            "structured": self.config.monitoring.enable_structured_logging
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return (
            f"EnhancedRAGConfig(\n"
            f"  Environment: {self.config.production.environment}\n"
            f"  Vector DB: {self.config.vector_db.type}\n"
            f"  Embedding Model: {self.config.embeddings.primary_model}\n"
            f"  OpenAI Embeddings: {self.config.embeddings.use_openai_embeddings}\n"
            f"  Multi-stage Retrieval: {self.config.features.enable_multi_stage_retrieval}\n"
            f"  Advanced Chunking: {self.config.features.enable_advanced_chunking}\n"
            f"  Response Templates: {self.config.features.enable_response_templates}\n"
            f"  Multi-source Synthesis: {self.config.features.enable_multi_source_synthesis}\n"
            f")"
        )

# Global configuration instance
_config_manager: Optional[ConfigManager] = None

def get_config_manager(config_path: Optional[str] = None, env_file: Optional[str] = None) -> ConfigManager:
    """Get or create global configuration manager instance."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_path=config_path, env_file=env_file)
    
    return _config_manager

def get_config() -> EnhancedRAGConfig:
    """Get the configuration object."""
    return get_config_manager().config
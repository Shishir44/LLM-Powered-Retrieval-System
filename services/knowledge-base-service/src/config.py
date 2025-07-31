"""
Configuration management for Knowledge Base Service
"""

import os
from typing import List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models - UNIFIED STRATEGY."""
    # Embedding provider choice
    embedding_provider: str = "gemini"  # "openai", "gemini", "local"
    
    # OpenAI embeddings (temporarily disabled due to quota limits)
    use_openai_embeddings: bool = False
    openai_model: str = "text-embedding-3-large"
    
    # Gemini embeddings
    use_gemini_embeddings: bool = True
    gemini_model: str = "models/embedding-001"
    
    # Dimension is provider-dependent
    embedding_dimension: int = 768  # Default for Gemini embedding-001
    
    # Cross-encoder for reranking only
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    
    # Processing parameters
    max_sequence_length: int = 8192
    batch_size: int = 16

@dataclass
class RetrievalConfig:
    """Configuration for retrieval pipeline."""
    initial_retrieval_size: int = 50
    rerank_size: int = 20
    diversity_selection_size: int = 10
    final_results_size: int = 5
    similarity_threshold: float = 0.8
    rerank_threshold: float = 0.5
    diversity_threshold: float = 0.7
    enable_query_expansion: bool = True
    query_expansion_limit: int = 3

@dataclass
class RerankingConfig:
    """Configuration for reranking models and strategies."""
    primary_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    fallback_model: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    max_sequence_length: int = 512
    batch_size: int = 32
    enable_diversity: bool = True
    diversity_threshold: float = 0.7
    min_relevance_score: float = 0.1
    max_rerank_documents: int = 100
    enable_cross_encoder_reranking: bool = True
    rerank_top_k: int = 50
    final_retrieval_top_k: int = 10

@dataclass
class ChunkingConfig:
    """Configuration for document chunking - OPTIMIZED FOR PERFORMANCE."""
    strategy: str = "semantic_structure"
    # PERFORMANCE OPTIMIZATION: Smaller chunks for better processing speed
    max_chunk_size: int = 512   # Reduced from 1024 for better performance
    chunk_overlap: int = 128    # Reduced from 256 for better performance
    similarity_threshold: float = 0.8
    spacy_model: str = "en_core_web_sm"

@dataclass
class CacheConfig:
    """Configuration for caching."""
    enable_query_cache: bool = True
    enable_embedding_cache: bool = True
    cache_ttl: int = 3600
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0

@dataclass
class VectorDBConfig:
    """Configuration for vector database."""
    type: str = "chroma"
    chroma_persist_directory: str = "./data/chroma"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    pinecone_api_key: str = ""
    pinecone_environment: str = ""
    pinecone_index_name: str = "rag-system-enhanced"
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: str = ""

@dataclass
class ServiceConfig:
    """Main service configuration."""
    service_name: str = "knowledge-base-service"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8002
    debug: bool = True
    log_level: str = "INFO"
    enable_cors: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])

@dataclass
class KnowledgeBaseConfig:
    """Complete configuration for Knowledge Base Service."""
    service: ServiceConfig = field(default_factory=ServiceConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = field(default_factory=RerankingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)

class ConfigManager:
    """Configuration manager that loads settings from environment variables."""
    
    def __init__(self, env_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self._load_env_file(env_file)
        self.config = self._create_config()
    
    def _load_env_file(self, env_file: Optional[str] = None):
        """Load environment variables from .env file."""
        try:
            from dotenv import load_dotenv
            
            if env_file:
                env_path = Path(env_file)
            else:
                # Look for .env files in common locations
                possible_paths = [
                    Path(".env"),
                    Path("setup/.env"),
                    Path("../.env"),
                    Path("../../setup/.env")
                ]
                env_path = None
                for path in possible_paths:
                    if path.exists():
                        env_path = path
                        break
            
            if env_path and env_path.exists():
                load_dotenv(env_path)
                self.logger.info(f"Loaded environment variables from {env_path}")
            else:
                self.logger.warning("No .env file found, using system environment variables")
                
        except ImportError:
            self.logger.warning("python-dotenv not available, using system environment variables")
    
    def _create_config(self) -> KnowledgeBaseConfig:
        """Create configuration from environment variables."""
        
        # Service configuration
        service_config = ServiceConfig(
            service_name=os.getenv("SERVICE_NAME", "knowledge-base-service"),
            version=os.getenv("SERVICE_VERSION", "1.0.0"),
            host=os.getenv("KNOWLEDGE_BASE_SERVICE_HOST", "0.0.0.0"),
            port=int(os.getenv("KNOWLEDGE_BASE_SERVICE_PORT", "8002")),
            debug=os.getenv("DEBUG", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            enable_cors=os.getenv("ENABLE_CORS", "true").lower() == "true",
            allowed_origins=os.getenv("ALLOWED_ORIGINS", "*").split(",")
        )
        
        # Embedding configuration
        embedding_config = EmbeddingConfig(
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "gemini"),
            use_openai_embeddings=os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true",
            openai_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            use_gemini_embeddings=os.getenv("USE_GEMINI_EMBEDDINGS", "true").lower() == "true",
            gemini_model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001"),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
            cross_encoder_model=os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
            max_sequence_length=int(os.getenv("MAX_SEQUENCE_LENGTH", "8192")),
            batch_size=int(os.getenv("BATCH_SIZE", "16"))
        )
        
        # Retrieval configuration
        retrieval_config = RetrievalConfig(
            initial_retrieval_size=int(os.getenv("INITIAL_RETRIEVAL_SIZE", "50")),
            rerank_size=int(os.getenv("RERANK_SIZE", "20")),
            diversity_selection_size=int(os.getenv("DIVERSITY_SELECTION_SIZE", "10")),
            final_results_size=int(os.getenv("FINAL_RESULTS_SIZE", "5")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.8")),
            rerank_threshold=float(os.getenv("RERANK_THRESHOLD", "0.5")),
            diversity_threshold=float(os.getenv("DIVERSITY_THRESHOLD", "0.7")),
            enable_query_expansion=os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true",
            query_expansion_limit=int(os.getenv("QUERY_EXPANSION_LIMIT", "3"))
        )
        
        # Reranking configuration
        reranking_config = RerankingConfig(
            primary_model=os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
            enable_cross_encoder_reranking=os.getenv("ENABLE_CROSS_ENCODER_RERANKING", "true").lower() == "true",
            enable_diversity=os.getenv("ENABLE_DIVERSITY_SELECTION", "true").lower() == "true"
        )
        
        # Chunking configuration
        chunking_config = ChunkingConfig(
            strategy=os.getenv("CHUNKING_STRATEGY", "semantic_structure"),
            max_chunk_size=int(os.getenv("MAX_CHUNK_SIZE", "1024")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "256")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD_CHUNKING", "0.8")),
            spacy_model=os.getenv("SPACY_MODEL", "en_core_web_sm")
        )
        
        # Cache configuration
        cache_config = CacheConfig(
            enable_query_cache=os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true",
            enable_embedding_cache=os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true",
            cache_ttl=int(os.getenv("CACHE_TTL", "3600")),
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_password=os.getenv("REDIS_PASSWORD", ""),
            redis_db=int(os.getenv("REDIS_DB", "0"))
        )
        
        # Vector DB configuration
        vector_db_config = VectorDBConfig(
            type=os.getenv("VECTOR_DB_TYPE", "chroma"),
            chroma_persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma"),
            chroma_host=os.getenv("CHROMA_HOST", "localhost"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            pinecone_environment=os.getenv("PINECONE_ENVIRONMENT", ""),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "rag-system-enhanced"),
            weaviate_url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
            weaviate_api_key=os.getenv("WEAVIATE_API_KEY", "")
        )
        
        return KnowledgeBaseConfig(
            service=service_config,
            embedding=embedding_config,
            retrieval=retrieval_config,
            reranking=reranking_config,
            chunking=chunking_config,
            cache=cache_config,
            vector_db=vector_db_config
        )
    
    def get_config(self) -> KnowledgeBaseConfig:
        """Get the complete configuration."""
        return self.config
    
    def get_openai_api_key(self) -> str:
        """Get OpenAI API key."""
        return os.getenv("OPENAI_API_KEY", "")
    
    def get_gemini_api_key(self) -> str:
        """Get Gemini API key."""
        return os.getenv("GEMINI_API_KEY", "")
    
    def get_service_config(self) -> ServiceConfig:
        """Get service-specific configuration."""
        return self.config.service
    
    def get_embedding_config(self) -> EmbeddingConfig:
        """Get embedding configuration."""
        return self.config.embedding
    
    def get_reranking_config(self) -> RerankingConfig:
        """Get reranking configuration."""
        return self.config.reranking

# Global configuration instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config() -> KnowledgeBaseConfig:
    """Get the global configuration."""
    return get_config_manager().get_config()

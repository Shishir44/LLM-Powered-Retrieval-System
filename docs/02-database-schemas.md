# Database Schemas and Data Models

## Overview

The LLM-Powered Retrieval System uses a hybrid data architecture combining in-memory storage for rapid prototyping with external database integrations for production scalability. The system is built with Pydantic models for data validation and structured data management.

## Data Architecture Pattern

### Current Implementation
- **In-Memory Storage**: Primary data storage using Python dictionaries and lists
- **External Integration Ready**: Configuration for PostgreSQL, Redis, and Vector Databases
- **Pydantic Validation**: Strong typing and validation throughout the system
- **No ORM Dependencies**: Direct data structure management for performance

### Production Migration Path
- **PostgreSQL**: Persistent storage for documents and metadata
- **Redis**: Caching layer and session management
- **Vector Databases**: Pinecone/FAISS for embedding storage

## Configuration Models

### Base Configuration Schema

```python
# services/shared/config.py:15
class BaseConfig(BaseSettings):
    # Database Configuration
    database_url: str = "postgresql://user:password@localhost:5432/rag_db"
    redis_url: str = "redis://localhost:6379/0"
    
    # Vector Database
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-west1-gcp"
    
    # External APIs
    openai_api_key: str
    anthropic_api_key: Optional[str] = None
    
    # System Settings
    environment: str = Field(default="development", 
                           regex="^(development|staging|production)$")
    log_level: str = "INFO"
    cors_origins: List[str] = ["http://localhost:3000"]
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
```

### Service-Specific Configurations

#### API Gateway Configuration
```python
class APIGatewayConfig(BaseConfig):
    # Service URLs
    conversation_service_url: str = "http://localhost:8001"
    knowledge_base_service_url: str = "http://localhost:8002"
    analytics_service_url: str = "http://localhost:8005"
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    
    # Circuit Breaker
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: int = 60
```

## API Request/Response Models

### Document Management Models

#### Document Creation Schema
```python
# services/knowledge-base-service/src/api/routes.py:25
class DocumentCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=10)
    category: str = Field(..., min_length=1, max_length=100)
    subcategory: Optional[str] = Field(None, max_length=100)
    tags: List[str] = Field(default=[], max_items=20)
    metadata: Dict[str, Any] = Field(default={})
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Docker Containerization Guide",
                "content": "Docker is a platform for developing, shipping...",
                "category": "DevOps",
                "subcategory": "Containerization",
                "tags": ["docker", "containers", "deployment"],
                "metadata": {"author": "Dev Team", "version": "1.0"}
            }
        }
```

#### Document Search Schema
```python
class DocumentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    category: Optional[str] = Field(None, max_length=100)
    limit: int = Field(default=10, ge=1, le=50)
    include_content: bool = Field(default=False)
    
    # Search filters
    tags: Optional[List[str]] = Field(None, max_items=10)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
```

#### Document Response Schema
```python
class DocumentResponse(BaseModel):
    id: str
    title: str
    content: Optional[str] = None
    category: str
    subcategory: Optional[str] = None
    tags: List[str]
    score: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata: Dict[str, Any]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
```

### Conversation Models

#### Conversation Request Schema
```python
# services/conversation-service/src/api/routes.py:20
class ConversationRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    conversation_id: Optional[str] = None
    context: Dict[str, Any] = Field(default={})
    
    # Advanced options
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=4000)
    stream: bool = Field(default=False)
```

#### Conversation Response Schema
```python
class ConversationResponse(BaseModel):
    response: str
    conversation_id: str
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    
    # Quality metrics
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    processing_time: float
    sources: List[Dict[str, Any]] = Field(default=[])
```

## Core Data Structures

### Query Processing Models

#### Query Analysis Schema
```python
# services/conversation-service/src/core/advanced_query_processor.py:15
@dataclass
class QueryAnalysis:
    # Core query information
    original_query: str
    query_type: QueryType  # FACTUAL, PROCEDURAL, ANALYTICAL, etc.
    complexity: QueryComplexity  # SIMPLE, MODERATE, COMPLEX, MULTI_STEP
    intent: str
    sentiment: str
    urgency: str
    
    # Extracted information
    entities: List[str]
    topics: List[str]
    keywords: List[str]
    
    # Processing metadata
    expanded_queries: List[str]
    confidence: float
    context_needed: bool
    estimated_response_length: int
    
    # Quality indicators
    clarity_score: float
    specificity_score: float
    completeness_score: float
```

#### Query Types and Complexity
```python
class QueryType(Enum):
    FACTUAL = "factual"              # "What is Docker?"
    PROCEDURAL = "procedural"        # "How do I deploy with Docker?"
    ANALYTICAL = "analytical"        # "Compare Docker vs Kubernetes"
    CONVERSATIONAL = "conversational" # "Tell me more about..."
    CLARIFICATION = "clarification"  # "What do you mean by..."
    MULTI_HOP = "multi_hop"         # Complex multi-step questions

class QueryComplexity(Enum):
    SIMPLE = "simple"        # Single concept, direct answer
    MODERATE = "moderate"    # Multiple concepts, structured answer
    COMPLEX = "complex"      # Multiple domains, detailed analysis
    MULTI_STEP = "multi_step" # Requires multiple retrieval rounds
```

### Context Management Models

#### Contextual Information Schema
```python
# services/conversation-service/src/core/advanced_context_manager.py:20
@dataclass
class ContextualInformation:
    # Context layers
    primary_context: str
    supporting_context: List[str]
    conversation_history: str
    user_profile: Dict[str, Any]
    
    # Temporal and domain context
    temporal_context: str
    domain_context: str
    
    # Quality metrics
    confidence_score: float
    relevance_scores: List[float]
    context_freshness: float
    
    # Metadata
    sources: List[str]
    last_updated: datetime
    retrieval_method: str
```

#### Conversation Turn Schema
```python
@dataclass
class ConversationTurn:
    # Turn information
    user_message: str
    assistant_response: str
    timestamp: datetime
    turn_id: str
    
    # Context and retrieval
    context_used: List[str]
    documents_retrieved: List[str]
    retrieval_quality: float
    
    # Feedback and quality
    user_feedback: Optional[float]
    quality_score: Optional[float]
    metadata: Dict[str, Any]
```

### Document Processing Models

#### Document Chunk Schema
```python
# services/knowledge-base-service/src/core/chunking.py:10
@dataclass
class DocumentChunk:
    content: str
    chunk_id: int
    document_id: str
    
    # Position information
    start_pos: int
    end_pos: int
    
    # Metadata
    metadata: Dict[str, Any]
    chunk_type: str = "text"  # text, code, table, etc.
    
    # Computed properties
    @property
    def hash(self) -> str:
        """Generate MD5 hash for deduplication"""
        return hashlib.md5(self.content.encode()).hexdigest()
    
    @property
    def word_count(self) -> int:
        return len(self.content.split())
```

#### Semantic Document Schema
```python
# services/knowledge-base-service/src/core/semantic_retriever.py:15
@dataclass
class SemanticDocument:
    # Core document information
    id: str
    content: str
    title: str
    metadata: Dict[str, Any]
    
    # Embeddings and features
    embedding: Optional[np.ndarray] = None
    dense_embedding: Optional[np.ndarray] = None
    sparse_features: Optional[Dict[str, float]] = None
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    # Computed properties
    @property
    def content_length(self) -> int:
        return len(self.content)
    
    @property
    def embedding_dimension(self) -> Optional[int]:
        return self.embedding.shape[0] if self.embedding is not None else None
```

### Retrieval and Ranking Models

#### Retrieval Result Schema
```python
# services/knowledge-base-service/src/core/retrieval.py:25
@dataclass
class RetrievalResult:
    # Core result
    chunk: DocumentChunk
    score: float
    retrieval_method: str
    
    # Detailed scoring
    bm25_score: Optional[float] = None
    tfidf_score: Optional[float] = None
    semantic_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    
    # Quality metrics
    relevance_explanation: str = ""
    confidence: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_timestamp: datetime = field(default_factory=datetime.now)
```

#### Enhanced Retrieval Result Schema
```python
# services/knowledge-base-service/src/core/semantic_retriever.py:45
@dataclass
class RetrievalResult:
    document: SemanticDocument
    
    # Multi-score ranking
    semantic_score: float
    keyword_score: float
    hybrid_score: float
    rerank_score: Optional[float] = None
    
    # Explanation and confidence
    relevance_explanation: str
    retrieval_method: str
    confidence: float
    
    # Performance metrics
    retrieval_time: float = 0.0
    rerank_time: Optional[float] = None
```

### Quality and Analytics Models

#### Quality Metrics Schema
```python
# services/conversation-service/src/core/response_quality_manager.py:10
@dataclass
class QualityMetrics:
    # Core quality dimensions
    accuracy: float = Field(..., ge=0.0, le=1.0)
    completeness: float = Field(..., ge=0.0, le=1.0)
    relevance: float = Field(..., ge=0.0, le=1.0)
    clarity: float = Field(..., ge=0.0, le=1.0)
    appropriateness: float = Field(..., ge=0.0, le=1.0)
    
    # Aggregated metrics
    overall_score: float = Field(..., ge=0.0, le=1.0)
    
    # Improvement suggestions
    suggestions: List[str] = field(default_factory=list)
    requires_revision: bool = False
    confidence_level: str = "medium"  # low, medium, high
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    evaluator: str = "automated"
```

#### RAG Response Schema
```python
# services/conversation-service/src/core/adaptive_rag_pipeline.py:30
@dataclass
class RAGResponse:
    # Core response
    response: str
    processing_time: float
    confidence_score: float
    
    # Analysis components
    query_analysis: QueryAnalysis
    contextual_info: ContextualInformation
    quality_metrics: QualityMetrics
    
    # Retrieval information
    retrieval_strategy: Dict[str, Any]
    sources: List[Dict[str, Any]]
    
    # System metadata
    metadata: Dict[str, Any]
    model_used: str
    tokens_used: int
    cost_estimate: float
```

### Caching Models

#### Cache Entry Schema
```python
# services/knowledge-base-service/src/core/cache.py:10
@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: float
    ttl: float  # Time to live in seconds
    
    # Usage tracking
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    # Cache metadata
    size_bytes: Optional[int] = None
    compression_used: bool = False
    
    @property
    def is_expired(self) -> bool:
        return time.time() > (self.created_at + self.ttl)
    
    def access(self) -> None:
        self.access_count += 1
        self.last_accessed = time.time()
```

## Storage Patterns and Data Flow

### In-Memory Storage Structure

```python
# Current storage implementation
document_storage: Dict[str, Dict[str, Any]] = {}
search_index: List[DocumentChunk] = []
conversations: Dict[str, Deque[ConversationTurn]] = {}
user_profiles: Dict[str, Dict[str, Any]] = {}
performance_history: List[Dict[str, Any]] = []

# Cache structures
vector_cache: Dict[str, np.ndarray] = {}
response_cache: Dict[str, CacheEntry] = {}
query_cache: Dict[str, List[RetrievalResult]] = {}
```

### Data Relationships

```mermaid
erDiagram
    Document ||--o{ DocumentChunk : "chunked_into"
    DocumentChunk ||--o{ RetrievalResult : "retrieved_as"
    Query ||--|| QueryAnalysis : "analyzed_to"
    QueryAnalysis ||--|| ContextualInformation : "enriched_with"
    ContextualInformation ||--|| RAGResponse : "generates"
    RAGResponse ||--o{ QualityMetrics : "evaluated_by"
    ConversationTurn ||--o{ ContextualInformation : "contributes_to"
    
    Document {
        string id PK
        string title
        string content
        string category
        string subcategory
        json tags
        json metadata
        datetime created_at
        datetime updated_at
    }
    
    DocumentChunk {
        string chunk_id PK
        string document_id FK
        string content
        int start_pos
        int end_pos
        json metadata
    }
    
    Query {
        string query_id PK
        string content
        string user_id
        datetime timestamp
    }
    
    QueryAnalysis {
        string query_id FK
        string query_type
        string complexity
        json entities
        json topics
        float confidence
    }
```

## Validation Rules and Constraints

### Field Validation
```python
# String length constraints
title: str = Field(..., min_length=1, max_length=500)
content: str = Field(..., min_length=10)
query: str = Field(..., min_length=1, max_length=1000)

# Numeric constraints
limit: int = Field(default=10, ge=1, le=50)
temperature: float = Field(default=0.7, ge=0.0, le=2.0)
satisfaction_score: float = Field(..., ge=0.0, le=1.0)

# List constraints
tags: List[str] = Field(default=[], max_items=20)
cors_origins: List[str] = Field(default=[])

# Enum constraints
environment: str = Field(regex="^(development|staging|production)$")
```

### Data Integrity
- **UUID Generation**: Document IDs using `uuid.uuid4()`
- **Content Hashing**: MD5 hashing for chunk deduplication
- **Timestamp Tracking**: Automatic datetime stamps for audit trails
- **Score Normalization**: All scores normalized to 0.0-1.0 range

## Migration to Production Database

### PostgreSQL Schema (Planned)
```sql
-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(100) NOT NULL,
    subcategory VARCHAR(100),
    tags JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Document chunks table
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    start_pos INTEGER NOT NULL,
    end_pos INTEGER NOT NULL,
    chunk_order INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Conversations table
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255),
    conversation_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Quality metrics table
CREATE TABLE quality_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID,
    query_text TEXT NOT NULL,
    response_text TEXT NOT NULL,
    metrics JSONB NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Redis Schema (Current Integration)
```python
# Cache keys
DOCUMENT_CACHE_KEY = "doc:{document_id}"
QUERY_CACHE_KEY = "query:{query_hash}"
CONVERSATION_CACHE_KEY = "conv:{conversation_id}"
USER_PROFILE_KEY = "user:{user_id}"

# TTL settings
DOCUMENT_TTL = 3600  # 1 hour
QUERY_TTL = 1800     # 30 minutes
CONVERSATION_TTL = 7200  # 2 hours
```

## Performance Considerations

### Memory Management
- **LRU Cache**: Size-limited caches with automatic eviction
- **Deque Structures**: Fixed-size conversation history (maxlen=100)
- **Lazy Loading**: Content loaded on-demand for large documents
- **Compression**: Optional compression for large cache entries

### Indexing Strategy
- **FAISS Indices**: Vector similarity search optimization
- **BM25 Indexing**: Keyword search optimization
- **Hash Indexing**: Fast exact-match lookups
- **Composite Indexing**: Multi-field search optimization

This data architecture provides a solid foundation for both development and production deployments, with clear migration paths to persistent storage solutions.
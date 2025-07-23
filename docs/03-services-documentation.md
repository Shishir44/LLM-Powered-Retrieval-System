# Services Documentation

## Overview

The LLM-Powered Retrieval System consists of four core microservices, each with specific responsibilities in the RAG pipeline. This document provides detailed technical documentation for each service's architecture, APIs, and implementation details.

## Service Architecture Map

```mermaid
graph TB
    subgraph "External Clients"
        WEB[Web UI]
        API[API Clients]
    end
    
    subgraph "Gateway Layer"
        AG[API Gateway<br/>Port 8080]
    end
    
    subgraph "Application Services"
        CS[Conversation Service<br/>Port 8001]
        KB[Knowledge Base Service<br/>Port 8002]  
        AS[Analytics Service<br/>Port 8005]
    end
    
    subgraph "Data Layer"
        REDIS[Redis Cache]
        VECTOR[Vector Store]
    end
    
    subgraph "Monitoring"
        PROM[Prometheus]
        GRAF[Grafana]
    end
    
    WEB --> AG
    API --> AG
    AG --> CS
    AG --> KB
    AG --> AS
    CS --> KB
    CS --> AS
    KB --> REDIS
    KB --> VECTOR
    AS --> PROM
    PROM --> GRAF
```

## 1. API Gateway Service

### Service Overview

**Purpose**: Central entry point providing unified API interface, security, and operational concerns.

**Port**: 8080  
**Base Path**: `/`  
**Health Check**: `/health`

### Technical Architecture

#### Core Responsibilities
- **Request Routing**: Intelligent proxying to backend services
- **Rate Limiting**: 100 requests/minute per IP address
- **Circuit Breaker**: Prevents cascading failures with automatic recovery
- **Health Monitoring**: Continuous backend service health assessment
- **Metrics Collection**: Prometheus metrics for operational visibility

#### Directory Structure
```
services/api-gateway/
├── Dockerfile                 # Container configuration
├── requirements.txt          # Python dependencies
├── src/
│   ├── __init__.py
│   └── main.py              # FastAPI application
└── tests/                   # Unit tests
```

### API Endpoints

| Endpoint | Method | Purpose | Proxy Target |
|----------|---------|---------|--------------|
| `/health` | GET | Gateway health + backend status | - |
| `/ready` | GET | Kubernetes readiness probe | - |
| `/metrics` | GET | Prometheus metrics | - |
| `/conversation/*` | ALL | Conversation service proxy | `:8001` |
| `/knowledge/*` | ALL | Knowledge base service proxy | `:8002` |
| `/analytics/*` | ALL | Analytics service proxy | `:8005` |
| `/` | GET | Service overview | - |

### Implementation Details

#### Enhanced Proxy Logic
```python
# services/api-gateway/src/main.py:45
async def proxy_request(service_url: str, path: str, request: Request):
    """Enhanced proxy with retry logic and circuit breaker"""
    retry_attempts = 3
    backoff_factor = 1.5
    
    for attempt in range(retry_attempts):
        try:
            # Circuit breaker check
            if not await is_service_healthy(service_url):
                raise HTTPException(503, "Service unavailable")
            
            # Forward request with timeout
            response = await httpx_client.request(
                method=request.method,
                url=f"{service_url}/{path}",
                headers=dict(request.headers),
                content=await request.body(),
                timeout=30.0
            )
            return response
            
        except httpx.TimeoutException:
            if attempt == retry_attempts - 1:
                raise HTTPException(504, "Gateway timeout")
            await asyncio.sleep(backoff_factor ** attempt)
```

#### Health Monitoring System
```python
# Periodic health checks every 30 seconds
health_status = {
    "conversation": {"healthy": True, "last_check": datetime},
    "knowledge_base": {"healthy": True, "last_check": datetime},
    "analytics": {"healthy": True, "last_check": datetime}
}
```

### Dependencies
```toml
fastapi = "0.104.1"
httpx = "0.25.0"
slowapi = "0.1.9"           # Rate limiting
prometheus-client = "0.19.0"
structlog = "23.2.0"
```

### Configuration
- **Service URLs**: Configurable backend service endpoints
- **Rate Limits**: Adjustable per-client request limits
- **Timeouts**: Connection (10s) and request (30s) timeouts
- **Circuit Breaker**: 5 failures trigger circuit opening, 60s recovery time

---

## 2. Conversation Service

### Service Overview

**Purpose**: Core RAG engine handling query processing, context management, and response generation.

**Port**: 8001  
**Base Path**: `/api/v1`  
**Health Check**: `/health`

### Technical Architecture

#### Core Components
1. **Adaptive RAG Pipeline**: `src/core/adaptive_rag_pipeline.py`
2. **Query Processor**: `src/core/advanced_query_processor.py`
3. **Context Manager**: `src/core/advanced_context_manager.py`
4. **Quality Manager**: `src/core/response_quality_manager.py`
5. **Streaming Engine**: `src/core/streaming.py`

#### Directory Structure
```
services/conversation-service/
├── Dockerfile
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # API endpoints
│   └── core/                # Business logic
│       ├── adaptive_rag_pipeline.py     # Main RAG engine
│       ├── advanced_query_processor.py  # Query analysis
│       ├── advanced_context_manager.py  # Context handling
│       ├── prompts.py                   # Prompt templates
│       ├── response_quality_manager.py  # Quality assessment
│       └── streaming.py                 # Streaming responses
└── tests/
```

### API Endpoints

| Endpoint | Method | Purpose | Response Format |
|----------|---------|---------|-----------------|
| `/chat` | POST | Process conversation message | JSON |
| `/chat/stream` | POST | Streaming conversation response | SSE |
| `/conversations/{id}` | GET | Retrieve conversation history | JSON |
| `/conversations/{id}` | DELETE | Delete conversation | JSON |
| `/pipeline/stats` | GET | RAG pipeline performance | JSON |
| `/pipeline/optimize` | POST | Optimize pipeline parameters | JSON |

### Core Algorithms

#### 1. Adaptive RAG Pipeline

The `EnhancedRAGPipeline` implements sophisticated query processing with dynamic strategy selection:

```python
# services/conversation-service/src/core/adaptive_rag_pipeline.py:30
class EnhancedRAGPipeline:
    async def process_query(self, query: str, conversation_id: str) -> RAGResponse:
        # 1. Query Analysis
        analysis = await self.query_processor.analyze_query(query)
        
        # 2. Strategy Selection
        strategy = self._select_strategy(analysis)
        
        # 3. Context Building
        context = await self.context_manager.build_context(
            query, conversation_id, analysis
        )
        
        # 4. Document Retrieval
        documents = await self._retrieve_documents(query, strategy)
        
        # 5. Response Generation
        response = await self._generate_response(query, context, documents)
        
        # 6. Quality Assessment
        quality = await self.quality_manager.evaluate_response(
            query, response, documents
        )
        
        return RAGResponse(
            response=response,
            query_analysis=analysis,
            contextual_info=context,
            quality_metrics=quality,
            sources=documents
        )
```

#### 2. Query Analysis Engine

**Query Type Classification**:
```python
class QueryType(Enum):
    FACTUAL = "factual"          # "What is Docker?"
    PROCEDURAL = "procedural"    # "How do I deploy with Docker?"
    ANALYTICAL = "analytical"    # "Compare Docker vs Kubernetes"
    CONVERSATIONAL = "conversational"  # "Tell me more about..."
    CLARIFICATION = "clarification"    # "What do you mean by..."
    MULTI_HOP = "multi_hop"           # Complex multi-step questions
```

**Dynamic Parameter Configuration**:
```python
QUERY_STRATEGIES = {
    QueryType.PROCEDURAL: {
        "max_documents": 8,
        "context_window": 7,
        "multi_round": False,
        "prioritize_recent": True
    },
    QueryType.ANALYTICAL: {
        "max_documents": 15,
        "context_window": 10,
        "multi_round": True,
        "comparative_mode": True
    },
    QueryType.MULTI_HOP: {
        "max_documents": 20,
        "context_window": 12,
        "multi_round": True,
        "retrieval_rounds": 3,
        "reasoning_chain": True
    }
}
```

#### 3. Advanced Context Management

**Context Layers**:
```python
@dataclass
class ContextualInformation:
    # Primary context from current query
    primary_context: str
    
    # Supporting context from conversation history
    supporting_context: List[str]
    
    # Full conversation history (last 10 turns)
    conversation_history: str
    
    # User profile and preferences
    user_profile: Dict[str, Any]
    
    # Temporal context (recent activity)
    temporal_context: str
    
    # Domain-specific context
    domain_context: str
    
    # Quality metrics
    confidence_score: float
    relevance_scores: List[float]
```

**Context Building Algorithm**:
```python
async def build_context(self, query: str, conversation_id: str, 
                       analysis: QueryAnalysis) -> ContextualInformation:
    # 1. Retrieve conversation history
    history = await self._get_conversation_history(conversation_id)
    
    # 2. Extract user profile
    profile = await self._build_user_profile(conversation_id)
    
    # 3. Determine temporal context
    temporal = await self._analyze_temporal_context(query, history)
    
    # 4. Build domain context
    domain = await self._extract_domain_context(analysis.topics)
    
    # 5. Calculate confidence scores
    confidence = self._calculate_context_confidence(history, profile)
    
    return ContextualInformation(
        primary_context=query,
        supporting_context=self._extract_supporting_context(history),
        conversation_history=self._format_history(history),
        user_profile=profile,
        temporal_context=temporal,
        domain_context=domain,
        confidence_score=confidence
    )
```

#### 4. Response Quality Management

**Quality Metrics**:
```python
@dataclass
class QualityMetrics:
    accuracy: float      # Factual correctness
    completeness: float  # Addresses all parts of query
    relevance: float     # Relevance to query
    clarity: float       # Clear and understandable
    appropriateness: float  # Tone and style appropriate
    overall_score: float    # Weighted average
    
    suggestions: List[str]     # Improvement suggestions
    requires_revision: bool    # Should response be revised?
    confidence_level: str      # "low", "medium", "high"
```

### Dependencies

#### Core Framework
```toml
fastapi = "0.104.1"
uvicorn = "0.24.0"
pydantic = "2.5.0"
```

#### LangChain Ecosystem
```toml
langchain = "0.0.340"
langchain-openai = "0.0.2"
langgraph = "0.0.20"
langchain-community = "0.0.3"
```

#### Utilities
```toml
httpx = "0.25.2"
structlog = "23.2.0"
tenacity = "8.2.3"    # Retry logic
asyncio-throttle = "1.0.2"
```

### Performance Features

#### Streaming Support
```python
# Server-Sent Events for real-time responses
async def stream_conversation(request: StreamingConversationRequest):
    async def generate():
        response_chunks = await rag_pipeline.stream_process_query(
            request.message, request.conversation_id
        )
        
        async for chunk in response_chunks:
            yield f"data: {json.dumps(chunk)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

#### Async Processing
- Non-blocking I/O for all external service calls
- Concurrent document retrieval and processing  
- Asynchronous context building and quality assessment

---

## 3. Knowledge Base Service

### Service Overview

**Purpose**: Document storage, indexing, retrieval, and vector operations management.

**Port**: 8002  
**Base Path**: `/api/v1`  
**Health Check**: `/health`

### Technical Architecture

#### Core Components
1. **Document Management**: CRUD operations with metadata
2. **Advanced Chunking**: Hierarchical text splitting strategies
3. **Multi-Modal Retrieval**: BM25, TF-IDF, semantic search
4. **Vector Store Integration**: Multiple backend support
5. **Caching Layer**: Redis-based performance optimization

#### Directory Structure
```
services/knowledge-base-service/
├── Dockerfile
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # API endpoints
│   └── core/                # Business logic
│       ├── cache.py         # Caching layer
│       ├── chunking.py      # Document chunking
│       ├── retrieval.py     # Search algorithms
│       └── semantic_retriever.py  # Vector search
└── tests/
```

### API Endpoints

| Endpoint | Method | Purpose | Parameters |
|----------|---------|---------|------------|
| `/documents` | POST | Create document | title, content, category, tags |
| `/documents` | GET | List documents | limit, offset, category filter |
| `/documents/{id}` | GET | Get document | include_content flag |
| `/documents/{id}` | DELETE | Delete document | - |
| `/documents/bulk` | POST | Bulk create (max 100) | documents array |
| `/search` | GET | Keyword search | query, category, limit |
| `/search/semantic` | GET | Semantic search | query, threshold, limit |
| `/stats` | GET | Knowledge base stats | - |

### Core Algorithms

#### 1. Advanced Document Chunking

**Hierarchical Text Splitting**:
```python
# services/knowledge-base-service/src/core/chunking.py:15
class DocumentChunker:
    SEPARATORS = [
        r'\n\n+',           # Multiple newlines (paragraphs)
        r'\n',              # Single newlines
        r'\. ',             # Sentence endings
        r'[!?] ',           # Exclamation/question marks
        r'; ',              # Semicolons
        r', ',              # Commas
        r' '                # Spaces
    ]
    
    def chunk_document(self, content: str, chunk_size: int = 1000, 
                      overlap: int = 200) -> List[DocumentChunk]:
        # 1. Try each separator in order of preference
        for separator in self.SEPARATORS:
            chunks = self._split_by_separator(content, separator, chunk_size)
            if self._chunks_are_reasonable(chunks):
                break
        
        # 2. Add overlap between chunks
        overlapped_chunks = self._add_overlap(chunks, overlap)
        
        # 3. Create DocumentChunk objects with metadata
        return [
            DocumentChunk(
                content=chunk_content,
                chunk_id=i,
                document_id=document_id,
                start_pos=start,
                end_pos=end,
                metadata={"chunk_method": separator}
            )
            for i, (chunk_content, start, end) in enumerate(overlapped_chunks)
        ]
```

#### 2. Advanced RAG Retrieval System

**Multi-Algorithm Scoring**:
```python
# services/knowledge-base-service/src/core/retrieval.py:30
class AdvancedRAGRetriever:
    async def search_documents(self, query: str, limit: int = 10) -> List[RetrievalResult]:
        # 1. BM25 Scoring
        bm25_results = await self._bm25_search(query, limit * 2)
        
        # 2. TF-IDF Scoring  
        tfidf_results = await self._tfidf_search(query, limit * 2)
        
        # 3. Enhanced Text Search
        text_results = await self._enhanced_text_search(query, limit * 2)
        
        # 4. Combine and rank results
        combined_results = self._hybrid_ranking(
            bm25_results, tfidf_results, text_results
        )
        
        return combined_results[:limit]
```

**BM25 Implementation**:
```python
def _calculate_bm25_score(self, query_terms: List[str], document: str) -> float:
    """
    BM25 scoring: score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_len)))
    """
    k1, b = 1.5, 0.75
    doc_terms = self._tokenize(document.lower())
    doc_length = len(doc_terms)
    
    score = 0.0
    for term in query_terms:
        tf = doc_terms.count(term)  # Term frequency
        if tf == 0:
            continue
            
        # Inverse document frequency
        idf = math.log((self.total_documents - self.term_doc_freq[term] + 0.5) / 
                      (self.term_doc_freq[term] + 0.5))
        
        # BM25 calculation
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / self.avg_doc_length)))
    
    return score
```

**Enhanced Text Search with Multi-Factor Scoring**:
```python
def _enhanced_text_search(self, query: str, limit: int) -> List[RetrievalResult]:
    query_terms = self._extract_query_terms(query)
    results = []
    
    for doc_id, document in self.document_storage.items():
        score = 0.0
        matched_terms = 0
        
        # Exact phrase matching (highest weight)
        if query.lower() in document['content'].lower():
            score += 3.0
        
        # Individual term matching with positional weighting
        for term in query_terms:
            term_lower = term.lower()
            
            # Title matching (high weight)
            if term_lower in document['title'].lower():
                score += 1.5
                matched_terms += 1
            
            # Content matching (base weight)  
            if term_lower in document['content'].lower():
                score += 1.0
                matched_terms += 1
                
            # Tag matching (medium weight)
            if any(term_lower in tag.lower() for tag in document.get('tags', [])):
                score += 1.2
                matched_terms += 1
        
        # Multiple term bonus
        if matched_terms > 1:
            score += 0.3 * matched_terms
            
        if score > 0:
            results.append(RetrievalResult(
                chunk=self._create_chunk(document),
                score=score,
                retrieval_method="enhanced_text",
                metadata={
                    "matched_terms": matched_terms,
                    "term_coverage": matched_terms / len(query_terms)
                }
            ))
    
    return sorted(results, key=lambda x: x.score, reverse=True)[:limit]
```

#### 3. Vector Store Integration

**Multi-Backend Support**:
```python
# Support for multiple vector database backends
VECTOR_STORES = {
    "pinecone": PineconeVectorStore,
    "weaviate": WeaviateVectorStore, 
    "chromadb": ChromaDBVectorStore,
    "faiss": FAISSVectorStore
}

class SemanticRetriever:
    def __init__(self, vector_store_type: str = "faiss"):
        self.vector_store = VECTOR_STORES[vector_store_type]()
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def semantic_search(self, query: str, k: int = 10) -> List[RetrievalResult]:
        # 1. Generate query embedding
        query_embedding = self.embeddings_model.encode([query])[0]
        
        # 2. Search vector store
        similar_docs = await self.vector_store.similarity_search(
            query_embedding, k=k
        )
        
        # 3. Re-rank with cross-encoder (if available)
        if self.cross_encoder:
            similar_docs = await self._rerank_documents(query, similar_docs)
            
        return similar_docs
```

### Dependencies

#### Core Framework
```toml
fastapi = "0.104.1"
uvicorn = "0.24.0"
pydantic = "2.5.0"
```

#### Vector & ML Libraries
```toml
sentence-transformers = "2.2.2"
faiss-cpu = "1.7.4"
numpy = "1.24.3"
pinecone-client = "2.2.4"
weaviate-client = "3.25.3"
chromadb = "0.4.18"
```

#### Database & Caching
```toml
sqlalchemy = "2.0.23"
psycopg2-binary = "2.9.9"
redis = "5.0.1"
```

### Performance Optimizations

#### Caching Strategy
```python
# services/knowledge-base-service/src/core/cache.py:10
class CacheManager:
    def __init__(self, redis_client, default_ttl: int = 3600):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.local_cache = {}  # L1 cache
        
    async def get_or_compute(self, key: str, compute_func, ttl: int = None):
        # 1. Check local cache first (L1)
        if key in self.local_cache:
            entry = self.local_cache[key]
            if not entry.is_expired:
                entry.access()
                return entry.value
        
        # 2. Check Redis cache (L2)
        cached_value = await self.redis.get(key)
        if cached_value:
            return json.loads(cached_value)
        
        # 3. Compute and cache
        value = await compute_func()
        await self._cache_value(key, value, ttl or self.default_ttl)
        return value
```

#### Bulk Operations
```python
@router.post("/documents/bulk")
async def create_documents_bulk(documents: List[DocumentCreateRequest]):
    """Create up to 100 documents in a single request"""
    if len(documents) > 100:
        raise HTTPException(400, "Maximum 100 documents per batch")
    
    results = []
    
    # Process in chunks of 10 for memory efficiency
    for chunk in chunked(documents, 10):
        chunk_results = await asyncio.gather(*[
            create_single_document(doc) for doc in chunk
        ])
        results.extend(chunk_results)
    
    return {"created": len(results), "documents": results}
```

---

## 4. Analytics Service

### Service Overview

**Purpose**: RAG quality evaluation, performance monitoring, and system observability.

**Port**: 8005  
**Base Path**: `/api/v1`  
**Health Check**: `/health`

### Technical Architecture

#### Core Components
1. **Quality Evaluation**: Automated RAG response assessment
2. **Performance Monitoring**: System metrics and alerting
3. **User Feedback**: Satisfaction tracking and analysis
4. **Prometheus Integration**: Metrics export for monitoring
5. **Health Assessment**: System status evaluation

#### Directory Structure
```
services/analytics-service/
├── Dockerfile
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # API endpoints
│   └── core/
│       ├── __init__.py
│       └── rag_metrics.py   # Quality evaluation
└── tests/
```

### API Endpoints

| Endpoint | Method | Purpose | Input |
|----------|---------|---------|-------|
| `/evaluate` | POST | Evaluate RAG response quality | query, context, response |
| `/metrics` | GET | Get current system metrics | - |
| `/feedback` | POST | Record user feedback | satisfaction_score, conversation_id |
| `/health-metrics` | GET | Service health indicators | - |
| `/metrics` (Prometheus) | GET | Prometheus metrics endpoint | - |

### Core Algorithms

#### 1. RAG Quality Evaluation

**Multi-Metric Assessment**:
```python
# services/analytics-service/src/core/rag_metrics.py:15
class RAGQualityEvaluator:
    def evaluate_response(self, query: str, context: str, 
                         response: str) -> Dict[str, float]:
        """Comprehensive quality assessment using multiple metrics"""
        
        # Tokenize inputs
        query_terms = set(self._tokenize(query.lower()))
        context_terms = set(self._tokenize(context.lower()))
        response_terms = set(self._tokenize(response.lower()))
        
        # Calculate individual metrics
        metrics = {
            "retrieval_precision": self._calculate_retrieval_precision(
                query_terms, context_terms
            ),
            "response_relevance": self._calculate_response_relevance(
                query_terms, response_terms
            ),
            "context_utilization": self._calculate_context_utilization(
                context_terms, response_terms
            ),
            "response_completeness": self._calculate_completeness(
                query, response
            ),
            "factual_consistency": self._calculate_consistency(
                context, response
            )
        }
        
        # Calculate overall score (weighted average)
        weights = {
            "retrieval_precision": 0.20,
            "response_relevance": 0.25,
            "context_utilization": 0.20,
            "response_completeness": 0.20,
            "factual_consistency": 0.15
        }
        
        overall_score = sum(
            metrics[metric] * weight 
            for metric, weight in weights.items()
        )
        
        metrics["overall_quality"] = overall_score
        return metrics
```

**Individual Metric Calculations**:
```python
def _calculate_retrieval_precision(self, query_terms: Set[str], 
                                  context_terms: Set[str]) -> float:
    """Measure how relevant retrieved documents are to the query"""
    if not query_terms:
        return 0.0
    
    relevant_terms = query_terms.intersection(context_terms)
    precision = len(relevant_terms) / len(query_terms)
    return min(precision, 1.0)

def _calculate_response_relevance(self, query_terms: Set[str], 
                                response_terms: Set[str]) -> float:
    """Measure how well the response addresses the query"""
    if not query_terms:
        return 0.0
    
    relevant_terms = query_terms.intersection(response_terms)
    relevance = len(relevant_terms) / len(query_terms)
    return min(relevance, 1.0)

def _calculate_context_utilization(self, context_terms: Set[str], 
                                 response_terms: Set[str]) -> float:
    """Measure how effectively the retrieved context is used"""
    if not context_terms:
        return 0.0
    
    utilized_terms = context_terms.intersection(response_terms)
    utilization = len(utilized_terms) / len(context_terms)
    return min(utilization, 1.0)
```

#### 2. System Health Monitoring

**Health Status Algorithm**:
```python
def calculate_system_health(self) -> Dict[str, Any]:
    """Calculate overall system health based on multiple indicators"""
    
    # Get recent performance data (last hour)
    recent_metrics = self._get_recent_metrics(hours=1)
    
    if not recent_metrics:
        return {"status": "unknown", "message": "No recent data"}
    
    # Calculate key indicators
    total_queries = len(recent_metrics)
    failed_queries = sum(1 for m in recent_metrics if m.get("error"))
    success_rate = (total_queries - failed_queries) / total_queries
    
    avg_response_time = sum(m.get("response_time", 0) for m in recent_metrics) / total_queries
    avg_quality_score = sum(m.get("quality_score", 0) for m in recent_metrics) / total_queries
    
    # Determine health status
    if success_rate >= 0.95 and avg_response_time < 2.0 and avg_quality_score >= 0.7:
        status = "healthy"
    elif success_rate >= 0.90 and avg_response_time < 5.0 and avg_quality_score >= 0.6:
        status = "degraded"
    else:
        status = "unhealthy"
    
    return {
        "status": status,
        "success_rate": success_rate,
        "avg_response_time": avg_response_time,
        "avg_quality_score": avg_quality_score,
        "total_queries": total_queries,
        "timestamp": datetime.utcnow().isoformat()
    }
```

#### 3. Prometheus Metrics Integration

**Custom Metrics Collection**:
```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
query_counter = Counter('rag_queries_total', 'Total RAG queries', ['service', 'status'])
response_time_histogram = Histogram('rag_response_time_seconds', 'Response time distribution')
quality_score_histogram = Histogram('rag_quality_score', 'Quality score distribution')
active_conversations_gauge = Gauge('rag_active_conversations', 'Number of active conversations')

class MetricsCollector:
    def record_query(self, service: str, status: str, response_time: float, 
                    quality_score: float):
        """Record query metrics"""
        query_counter.labels(service=service, status=status).inc()
        response_time_histogram.observe(response_time)
        quality_score_histogram.observe(quality_score)
        
    def update_active_conversations(self, count: int):
        """Update active conversation count"""
        active_conversations_gauge.set(count)
```

### Dependencies

#### Core Framework
```toml
fastapi = "0.104.1"
uvicorn = "0.24.0"
pydantic = "2.5.0"
```

#### Monitoring & Metrics
```toml
prometheus-client = "0.19.0"
structlog = "23.2.0"
```

#### Utilities
```toml
httpx = "0.25.2"
numpy = "1.24.3"
```

### Integration Points

#### Conversation Service Integration
```python
# Called after each response generation
async def evaluate_conversation_response(conversation_id: str, 
                                       query: str, response: str):
    # Send evaluation request to analytics service
    evaluation_request = {
        "query": query,
        "context": retrieved_context,
        "response": response,
        "conversation_id": conversation_id
    }
    
    async with httpx.AsyncClient() as client:
        evaluation_response = await client.post(
            f"{ANALYTICS_SERVICE_URL}/api/v1/evaluate",
            json=evaluation_request
        )
    
    return evaluation_response.json()
```

#### Prometheus Integration
```python
# Expose metrics endpoint for Prometheus scraping
@router.get("/metrics")
async def get_prometheus_metrics():
    """Prometheus metrics endpoint"""
    return Response(
        generate_latest(REGISTRY),
        media_type="text/plain"
    )
```

---

## Shared Infrastructure

### Configuration Management

All services use centralized configuration through `/services/shared/config.py`:

```python
class BaseConfig(BaseSettings):
    # Database connections
    database_url: str = "postgresql://user:password@localhost:5432/rag_db"
    redis_url: str = "redis://localhost:6379/0"
    
    # External APIs
    openai_api_key: str
    anthropic_api_key: Optional[str] = None
    
    # Vector databases
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-west1-gcp"
    
    # System settings
    environment: str = Field(default="development", 
                           regex="^(development|staging|production)$")
    log_level: str = "INFO"
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    class Config:
        env_file = ".env"
        case_sensitive = False
```

### Inter-Service Communication

**Async HTTP Client Pattern**:
```python
import httpx

class ServiceClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def post(self, endpoint: str, data: dict) -> dict:
        response = await self.client.post(
            f"{self.base_url}{endpoint}",
            json=data
        )
        response.raise_for_status()
        return response.json()
```

### Health Check Implementation

**Kubernetes-Compatible Health Checks**:
```python
@router.get("/health")
async def health_check():
    """Liveness probe - service is running"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@router.get("/ready")  
async def readiness_check():
    """Readiness probe - service is ready to handle requests"""
    # Check dependencies (database, external services)
    checks = await perform_dependency_checks()
    
    if all(check["healthy"] for check in checks.values()):
        return {"status": "ready", "checks": checks}
    else:
        raise HTTPException(503, {"status": "not_ready", "checks": checks})
```

This comprehensive service architecture provides a robust, scalable foundation for the LLM-Powered Retrieval System with clear separation of concerns, comprehensive monitoring, and production-ready operational features.
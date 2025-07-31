from typing import List, Dict, Any, Optional, Tuple, Union
import asyncio
import numpy as np
from dataclasses import dataclass
import uuid
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging
import time
import asyncio

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: sentence_transformers not available: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    CrossEncoder = None

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: chromadb not available: {e}")
    CHROMADB_AVAILABLE = False
    chromadb = None

from langchain_openai import OpenAIEmbeddings
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    GEMINI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: langchain_google_genai not available: {e}")
    GEMINI_AVAILABLE = False
    GoogleGenerativeAIEmbeddings = None
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from core.reranker import CrossEncoderReranker, RerankingResult
from config import get_config, KnowledgeBaseConfig

# PHASE 2.1: Import metadata-boosted retriever
try:
    from .metadata_boosted_retriever import MetadataBoostedRetriever, RetrievalResult as BoostedRetrievalResult
    METADATA_BOOST_AVAILABLE = True
except ImportError:
    METADATA_BOOST_AVAILABLE = False

# PHASE 2.3: Import optimized vector database
try:
    from .optimized_vector_db import OptimizedVectorDatabase, VectorDBConfig, QueryResult as OptimizedQueryResult
    OPTIMIZED_DB_AVAILABLE = True
except ImportError:
    OPTIMIZED_DB_AVAILABLE = False

@dataclass
class SemanticDocument:
    """Document with semantic metadata and embeddings."""
    id: str
    content: str
    title: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    dense_embedding: Optional[np.ndarray] = None
    sparse_features: Optional[Dict[str, float]] = None
    created_at: datetime = None
    updated_at: datetime = None

@dataclass
class RetrievalResult:
    """Enhanced retrieval result with multiple scoring mechanisms."""
    document: SemanticDocument
    semantic_score: float
    keyword_score: float
    hybrid_score: float
    rerank_score: Optional[float] = None
    relevance_explanation: str = ""
    retrieval_method: str = ""
    confidence: float = 0.0

class SemanticRetriever:
    """PHASE 2.3: Enhanced semantic retrieval system with optimized vector database and metadata boosting."""
    
    def __init__(self, 
                 primary_model: Optional[str] = None,
                 cross_encoder_model: Optional[str] = None,
                 use_openai_embeddings: Optional[bool] = None,
                 config: Optional[KnowledgeBaseConfig] = None):
        
        # Initialize logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Get configuration
        if config is None:
            config = get_config()
        self.config = config
        
        # Store initialization parameters
        self.cross_encoder_model_param = cross_encoder_model
        
        # PHASE 1.1: UNIFIED EMBEDDING STRATEGY - Check configuration for embedding choice
        self.embedding_provider = self.config.embedding.embedding_provider
        self.use_openai_embeddings = self.config.embedding.use_openai_embeddings
        self.use_gemini_embeddings = self.config.embedding.use_gemini_embeddings
        
        # Initialize embedding models based on provider
        self.openai_embeddings = None
        self.gemini_embeddings = None
        self.local_embedding_model = None
        
        if self.embedding_provider == "gemini" and self.use_gemini_embeddings and GEMINI_AVAILABLE:
            # Use Gemini embeddings
            self.embedding_model_name = self.config.embedding.gemini_model
            self.embedding_dimension = self.config.embedding.embedding_dimension
            
            try:
                from config import get_config_manager
                gemini_api_key = get_config_manager().get_gemini_api_key()
                if not gemini_api_key:
                    raise ValueError("GEMINI_API_KEY environment variable is required")
                    
                self.gemini_embeddings = GoogleGenerativeAIEmbeddings(
                    model=self.config.embedding.gemini_model,
                    google_api_key=gemini_api_key
                )
                self.logger.info(f"Loaded unified Gemini embedding model: {self.config.embedding.gemini_model}")
            except Exception as e:
                self.logger.error(f"Failed to load Gemini embeddings: {e}")
                # Fallback to local embeddings
                self._init_local_embeddings()
                
        elif self.embedding_provider == "openai" and self.use_openai_embeddings:
            # Use OpenAI embeddings
            self.embedding_model_name = self.config.embedding.openai_model
            self.embedding_dimension = self.config.embedding.embedding_dimension
            
            try:
                self.openai_embeddings = OpenAIEmbeddings(
                    model=self.config.embedding.openai_model,
                    dimensions=self.config.embedding.embedding_dimension
                )
                self.logger.info(f"Loaded unified OpenAI embedding model: {self.config.embedding.openai_model}")
            except Exception as e:
                self.logger.error(f"Failed to load OpenAI embeddings: {e}")
                # Fallback to local embeddings
                self._init_local_embeddings()
        else:
            # Use local sentence transformers embeddings
            self._init_local_embeddings()
        
        # Initialize ChromaDB regardless of embedding provider
        self._init_chromadb()
        
        # Initialize other components
        self._init_other_components()
    
    def _init_chromadb(self):
        """Initialize ChromaDB with simplified, working configuration."""
        self.chroma_client = None
        self.collection = None
        self.collection_name = "knowledge_base_documents"
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Simplified settings that actually work
            chroma_settings = Settings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory=str(Path(self.config.vector_db.chroma_persist_directory))
            )
            
            # Use persistent client
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.vector_db.chroma_persist_directory,
                settings=chroma_settings
            )
            
            # Create or get collection with basic metadata
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.collection_name,
                    embedding_function=None
                )
                doc_count = self.collection.count()
                self.logger.info(f"Loaded existing ChromaDB collection with {doc_count} documents")
            except:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Knowledge base documents"},
                    embedding_function=None
                )
                self.logger.info("Created new ChromaDB collection")
            
            # Performance logging based on collection size
            try:
                doc_count = self.collection.count()
                self.logger.info(f"ChromaDB successfully initialized with {doc_count} documents")
                
                if doc_count > 50000:
                    self.logger.info("Large collection detected (50K+)")
                elif doc_count > 10000:
                    self.logger.info("Medium collection detected (10K+)")
                else:
                    self.logger.info("Small collection detected")
                    
            except Exception as e:
                self.logger.warning(f"Could not get collection count: {e}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            # Don't fail completely, fall back to in-memory
            self.chroma_client = None
            self.collection = None
    
    def _init_other_components(self):
        """Initialize other components like reranker, chunker, etc."""
        # Legacy in-memory storage for fallback
        self.documents: List[SemanticDocument] = []
        self.id_to_idx: Dict[str, int] = {}
        
        # Advanced chunking system
        try:
            from .advanced_chunking import AdvancedDocumentChunker
            self.chunker = AdvancedDocumentChunker(
                model_name=self.embedding_model_name,
                max_chunk_size=self.config.chunking.max_chunk_size,
                overlap_size=self.config.chunking.chunk_overlap,
                similarity_threshold=self.config.chunking.similarity_threshold
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize advanced chunker: {e}")
            self.chunker = None
        
        # Fallback text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunking.max_chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
            length_function=len
        )
        
        # Query enhancement
        self.query_expansion_cache = {}
        
        # Initialize reranker
        try:
            self.reranker = CrossEncoderReranker(self.config)
            self.enable_reranking = self.config.reranking.enable_cross_encoder_reranking
            self.rerank_top_k = self.config.reranking.rerank_top_k
            self.final_top_k = self.config.reranking.final_retrieval_top_k
        except Exception as e:
            self.logger.error(f"Failed to initialize reranker: {e}")
            self.reranker = None
            self.enable_reranking = False
            self.rerank_top_k = 50
            self.final_top_k = 10
        
        # PHASE 2.1: Initialize metadata-boosted retriever
        if METADATA_BOOST_AVAILABLE:
            try:
                self.metadata_boosted_retriever = MetadataBoostedRetriever(
                    base_retriever=self,
                    boost_config={
                        "recency_weight": 0.15,
                        "authority_weight": 0.20,
                        "relevance_weight": 0.25,
                        "category_weight": 0.15,
                        "quality_weight": 0.15,
                        "popularity_weight": 0.10,
                        "recency_decay_days": 365,
                        "min_boost_factor": 0.5,
                        "max_boost_factor": 2.0,
                    }
                )
                self.enable_metadata_boosting = True
                self.logger.info("PHASE 2.1: Metadata-boosted retriever initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize metadata boosting: {e}")
                self.metadata_boosted_retriever = None
                self.enable_metadata_boosting = False
        else:
            self.metadata_boosted_retriever = None
            self.enable_metadata_boosting = False
            self.logger.warning("PHASE 2.1: Metadata boosting not available")
    
    def _init_local_embeddings(self):
        """Initialize local sentence transformer embeddings as fallback."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model_name = "all-mpnet-base-v2"
                self.embedding_dimension = 768  # all-mpnet-base-v2 dimension
                self.local_embedding_model = SentenceTransformer(self.embedding_model_name)
                self.logger.info(f"Loaded local embedding model: {self.embedding_model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load local embeddings: {e}")
                raise RuntimeError(f"Cannot initialize without embedding model: {e}")
        else:
            raise RuntimeError("No embedding model available (sentence_transformers not installed)")
        
        # Cross-encoder for reranking only
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                cross_encoder_model_name = self.cross_encoder_model_param or self.config.embedding.cross_encoder_model
                self.cross_encoder = CrossEncoder(cross_encoder_model_name)
                self.logger.info(f"Loaded cross-encoder for reranking: {cross_encoder_model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load cross-encoder: {e}")
                self.cross_encoder = None
        else:
            self.cross_encoder = None
            self.logger.warning("sentence_transformers not available, cross-encoder reranking disabled")
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection for persistent storage."""
        if not CHROMADB_AVAILABLE:
            self.logger.warning("ChromaDB not available, using in-memory storage")
            return
        
        try:
            # Create persistent storage directory
            storage_path = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_storage")
            os.makedirs(storage_path, exist_ok=True)
            
            # Initialize ChromaDB client with persistent storage
            self.chroma_client = chromadb.PersistentClient(
                path=storage_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.collection_name
                )
                self.logger.info(f"Loaded existing ChromaDB collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "description": "Knowledge base documents with embeddings",
                        "embedding_model": self.embedding_model_name,
                        "embedding_dimension": self.embedding_dimension
                    }
                )
                self.logger.info(f"Created new ChromaDB collection: {self.collection_name}")
            
            # Log existing document count
            existing_count = self.collection.count()
            self.logger.info(f"ChromaDB initialized with {existing_count} existing documents")
            
            # Ensure vector database is properly initialized
            if existing_count == 0:
                self.logger.info("Vector database is empty and ready for document uploads")
            else:
                self.logger.info(f"Vector database contains {existing_count} embedded documents")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            self.chroma_client = None
            self.collection = None
    
    async def _embed_query(self, query: str) -> np.ndarray:
        """Unified query embedding method that works with Gemini, OpenAI or local embeddings."""
        if self.use_gemini_embeddings and self.gemini_embeddings:
            return await self.gemini_embeddings.aembed_query(query)
        elif self.use_openai_embeddings and self.openai_embeddings:
            return await self.openai_embeddings.aembed_query(query)
        elif hasattr(self, 'local_embedding_model'):
            return self.local_embedding_model.encode([query])[0]
        else:
            raise RuntimeError("No embedding model available")
    
    def _embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Unified document embedding method that works with Gemini, OpenAI or local embeddings."""
        if self.use_gemini_embeddings and self.gemini_embeddings:
            return self.gemini_embeddings.embed_documents(texts)
        elif self.use_openai_embeddings and self.openai_embeddings:
            return self.openai_embeddings.embed_documents(texts)
        elif hasattr(self, 'local_embedding_model'):
            return self.local_embedding_model.encode(texts)
        else:
            raise RuntimeError("No embedding model available")
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents with semantic processing and persistent storage."""
        semantic_docs = []
        chroma_documents = []
        chroma_embeddings = []
        chroma_metadatas = []
        chroma_ids = []
        
        for doc in documents:
            # Create semantic document
            doc_id = doc.get("id", str(uuid.uuid4()))
            sem_doc = SemanticDocument(
                id=doc_id,
                content=doc["content"],
                title=doc.get("title", ""),
                metadata=doc.get("metadata", {}),
                created_at=datetime.now()
            )
            
            # Generate embeddings
            await self._generate_embeddings(sem_doc)
            semantic_docs.append(sem_doc)
            
            # Prepare ChromaDB data
            if self.collection and sem_doc.embedding is not None:
                chroma_documents.append(sem_doc.content)
                chroma_embeddings.append(sem_doc.embedding.tolist())
                
                # Filter out None values from metadata (ChromaDB doesn't accept None)
                filtered_metadata = {
                    "title": str(sem_doc.title),
                    "created_at": sem_doc.created_at.isoformat()
                }
                
                # Process each metadata item, ensuring proper type conversion
                for key, value in sem_doc.metadata.items():
                    if value is not None:
                        # Convert to supported ChromaDB types
                        if isinstance(value, (str, int, float, bool)):
                            filtered_metadata[key] = value
                        elif isinstance(value, list):
                            # Convert list to comma-separated string
                            filtered_metadata[key] = ",".join(str(item) for item in value if item is not None)
                        else:
                            # Convert other types to string
                            filtered_metadata[key] = str(value)
                
                self.logger.info(f"ChromaDB metadata for doc {doc_id}: {filtered_metadata}")
                
                chroma_metadatas.append(filtered_metadata)
                chroma_ids.append(doc_id)
        
        # Store in ChromaDB for persistence
        if self.collection and chroma_documents:
            try:
                self.collection.add(
                    documents=chroma_documents,
                    embeddings=chroma_embeddings,
                    metadatas=chroma_metadatas,
                    ids=chroma_ids
                )
                self.logger.info(f"Added {len(chroma_documents)} documents to ChromaDB")
            except Exception as e:
                self.logger.error(f"Failed to add documents to ChromaDB: {e}")
                # Fallback to in-memory storage
                start_idx = len(self.documents)
                self.documents.extend(semantic_docs)
                for i, doc in enumerate(semantic_docs):
                    self.id_to_idx[doc.id] = start_idx + i
        else:
            # Fallback to in-memory storage
            start_idx = len(self.documents)
            self.documents.extend(semantic_docs)
            for i, doc in enumerate(semantic_docs):
                self.id_to_idx[doc.id] = start_idx + i
        
        self.logger.info(f"Added {len(semantic_docs)} documents to semantic retriever")
    
    async def _generate_embeddings(self, document: SemanticDocument) -> None:
        """Generate embeddings for a document using OpenAI."""
        try:
            # Use executor to run sync embedding call
            embedding = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda q: asyncio.run(self._embed_query(q)),
                document.content
            )
            document.embedding = np.array(embedding)
            document.dense_embedding = np.array(embedding)  # Use same embedding for consistency
            
            self.logger.debug(f"Generated embeddings for document {document.id} with dimension {len(embedding)}")
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings for document {document.id}: {e}")
            # Fallback to zero embedding
            document.embedding = np.zeros(self.embedding_dimension)
            document.dense_embedding = np.zeros(self.embedding_dimension)
    
    def _load_documents_from_chroma(self) -> List[SemanticDocument]:
        """Load documents from ChromaDB into memory for legacy compatibility."""
        if not self.collection:
            return []
        
        try:
            # Get all documents from ChromaDB
            results = self.collection.get(
                include=["documents", "metadatas", "embeddings"]
            )
            
            documents = []
            for i, (doc_id, content, metadata, embedding) in enumerate(
                zip(results["ids"], results["documents"], results["metadatas"], results["embeddings"])
            ):
                # Parse created_at from metadata
                created_at = None
                if "created_at" in metadata:
                    try:
                        created_at = datetime.fromisoformat(metadata["created_at"])
                    except:
                        created_at = datetime.now()
                else:
                    created_at = datetime.now()
                
                # Extract title from metadata
                title = metadata.pop("title", "")
                metadata.pop("created_at", None)  # Remove from metadata dict
                
                sem_doc = SemanticDocument(
                    id=doc_id,
                    content=content,
                    title=title,
                    metadata=metadata,
                    embedding=np.array(embedding),
                    created_at=created_at
                )
                documents.append(sem_doc)
                self.id_to_idx[doc_id] = i
            
            self.logger.info(f"Loaded {len(documents)} documents from ChromaDB")
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to load documents from ChromaDB: {e}")
            return []
    
    async def semantic_search(self, 
                            query: str, 
                            top_k: int = 10,
                            use_query_expansion: bool = True,
                            filters: Dict[str, Any] = None) -> List[RetrievalResult]:
        """Perform semantic search with query expansion and reranking."""
        
        # First try ChromaDB search
        if self.collection:
            try:
                return await self._chroma_semantic_search(query, top_k, filters)
            except Exception as e:
                self.logger.error(f"ChromaDB search failed, falling back to in-memory: {e}")
        
        # Fallback to in-memory search
        if not self.documents:
            # Load from ChromaDB if available
            self.documents = self._load_documents_from_chroma()
        
        if not self.documents:
            return []
        
        # Search with semantic similarity
        semantic_results = await self._semantic_search_single(query, top_k * 2)
        
        # Keyword search results
        keyword_results = await self._keyword_search(query, top_k * 2)
        
        # Combine and score
        combined_results = await self._combine_results(
            semantic_results, keyword_results, query
        )
        
        # Apply filters if provided
        if filters:
            combined_results = self._apply_filters(combined_results, filters)
        
        # Rerank with cross-encoder
        reranked_results = await self._rerank_results(query, combined_results)
        
        # Final scoring and sorting
        final_results = self._final_scoring(reranked_results, query)
        
        return final_results[:top_k]
    
    async def _chroma_semantic_search(self, query: str, top_k: int, filters: Dict[str, Any] = None) -> List[RetrievalResult]:
        """Perform semantic search using ChromaDB."""
        try:
            # Generate query embedding
            query_embedding = await self._embed_query(query)
            query_embedding = np.array(query_embedding)
            
            # Prepare where clause for filters
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    if key not in ["date_range", "boost_recent"]:
                        where_clause[key] = value
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 2, 100),  # Get more for reranking
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert to RetrievalResult format
            retrieval_results = []
            
            for i, (doc_id, content, metadata, distance) in enumerate(
                zip(results["ids"][0], results["documents"][0], 
                    results["metadatas"][0], results["distances"][0])
            ):
                # Parse metadata
                title = metadata.get("title", "")
                created_at = None
                if "created_at" in metadata:
                    try:
                        created_at = datetime.fromisoformat(metadata["created_at"])
                    except:
                        created_at = datetime.now()
                else:
                    created_at = datetime.now()
                
                # Remove special fields from metadata
                clean_metadata = {k: v for k, v in metadata.items() 
                                if k not in ["title", "created_at"]}
                
                # Create semantic document
                sem_doc = SemanticDocument(
                    id=doc_id,
                    content=content,
                    title=title,
                    metadata=clean_metadata,
                    created_at=created_at
                )
                
                # Convert distance to similarity score (ChromaDB uses L2 distance)
                similarity_score = max(0.0, 1.0 / (1.0 + distance))
                
                # Create retrieval result
                result = RetrievalResult(
                    document=sem_doc,
                    semantic_score=similarity_score,
                    keyword_score=0.0,  # Will be computed if needed
                    hybrid_score=similarity_score,
                    retrieval_method="chroma_semantic",
                    relevance_explanation=f"ChromaDB similarity: {similarity_score:.3f}"
                )
                
                retrieval_results.append(result)
            
            # Apply additional keyword scoring if needed
            retrieval_results = await self._enhance_with_keyword_scores(query, retrieval_results)
            
            # Rerank if enabled
            if self.enable_reranking:
                retrieval_results = await self._rerank_results(query, retrieval_results)
            
            # Final scoring and sorting
            final_results = self._final_scoring(retrieval_results, query)
            
            return final_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"ChromaDB semantic search failed: {e}")
            raise
    
    async def _enhance_with_keyword_scores(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Enhance results with keyword-based scores."""
        query_terms = query.lower().split()
        
        for result in results:
            # Calculate keyword score
            kw_score = self._calculate_bm25_score_for_doc(query_terms, result.document)
            kw_score_norm = min(kw_score / 10.0, 1.0) if kw_score > 0 else 0.0
            
            # Update scores - optimize weights for better accuracy
            result.keyword_score = kw_score_norm
            result.hybrid_score = 0.75 * result.semantic_score + 0.25 * kw_score_norm  # Prioritize semantic for document-based accuracy
            result.relevance_explanation = f"Semantic: {result.semantic_score:.3f}, Keyword: {kw_score_norm:.3f}"
        
        return results
    
    def _calculate_bm25_score_for_doc(self, query_terms: List[str], document: SemanticDocument, 
                                     k1: float = 1.2, b: float = 0.75) -> float:
        """Calculate BM25 score for a single document."""
        doc_terms = document.content.lower().split()
        doc_length = len(doc_terms)
        
        if doc_length == 0:
            return 0.0
        
        # Use a reasonable average document length estimate
        avg_dl = 500  # Estimated average document length
        
        score = 0.0
        for term in query_terms:
            tf = doc_terms.count(term)
            if tf > 0:
                # Simple IDF approximation
                idf = 2.0  # Default IDF value
                
                # BM25 formula
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_dl)))
        
        return score
    
    async def _semantic_search_single(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Perform semantic search for a single query (legacy fallback)."""
        try:
            # Generate query embedding
            query_embedding = await self._embed_query(query)
            query_embedding = np.array(query_embedding)
            
            # Simple cosine similarity search without FAISS
            results = []
            for idx, doc in enumerate(self.documents):
                if doc.embedding is not None:
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, doc.embedding)
                    results.append((idx, float(similarity)))
            
            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return []
    
    async def _keyword_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Perform keyword-based search using BM25-like scoring."""
        query_terms = query.lower().split()
        results = []
        
        for idx, doc in enumerate(self.documents):
            score = self._calculate_bm25_score(query_terms, doc)
            if score > 0:
                results.append((idx, score))
        
        # Sort by score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _calculate_bm25_score(self, query_terms: List[str], document: SemanticDocument, 
                             k1: float = 1.2, b: float = 0.75) -> float:
        """Calculate BM25 score for keyword matching."""
        doc_terms = document.content.lower().split()
        doc_length = len(doc_terms)
        
        if doc_length == 0:
            return 0.0
        
        # Calculate average document length
        avg_dl = np.mean([len(d.content.split()) for d in self.documents])
        
        score = 0.0
        for term in query_terms:
            tf = doc_terms.count(term)
            if tf > 0:
                # Simple IDF approximation
                df = sum(1 for d in self.documents if term in d.content.lower())
                idf = np.log((len(self.documents) - df + 0.5) / (df + 0.5))
                
                # BM25 formula
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_dl)))
        
        return score
    
    async def _combine_results(self, 
                             semantic_results: List[Tuple[int, float]], 
                             keyword_results: List[Tuple[int, float]],
                             query: str) -> List[RetrievalResult]:
        """Combine semantic and keyword search results with hybrid scoring."""
        
        # Create lookup dictionaries
        semantic_scores = {idx: score for idx, score in semantic_results}
        keyword_scores = {idx: score for idx, score in keyword_results}
        
        # Get all unique document indices
        all_indices = set(semantic_scores.keys()) | set(keyword_scores.keys())
        
        combined_results = []
        
        for idx in all_indices:
            if idx >= len(self.documents):
                continue
                
            document = self.documents[idx]
            
            sem_score = semantic_scores.get(idx, 0.0)
            kw_score = keyword_scores.get(idx, 0.0)
            
            # Normalize scores
            sem_score_norm = min(sem_score, 1.0) if sem_score > 0 else 0.0
            kw_score_norm = min(kw_score / 10.0, 1.0) if kw_score > 0 else 0.0
            
            # Hybrid scoring with weights - prioritize semantic matching for document accuracy
            hybrid_score = 0.8 * sem_score_norm + 0.2 * kw_score_norm
            
            # Create retrieval result
            result = RetrievalResult(
                document=document,
                semantic_score=sem_score_norm,
                keyword_score=kw_score_norm,
                hybrid_score=hybrid_score,
                retrieval_method="hybrid",
                relevance_explanation=f"Semantic: {sem_score_norm:.3f}, Keyword: {kw_score_norm:.3f}"
            )
            
            combined_results.append(result)
        
        return combined_results
    
    async def _rerank_results(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank results using cross-encoder for better relevance."""
        if len(results) <= 1:
            return results
        
        try:
            if not self.cross_encoder:
                # No cross-encoder available, return original results
                for result in results:
                    result.rerank_score = result.hybrid_score
                    result.confidence = result.hybrid_score
                return results
            
            # Prepare query-document pairs for cross-encoder
            pairs = [(query, result.document.content) for result in results]
            
            # Get rerank scores
            rerank_scores = self.cross_encoder.predict(pairs)
            
            # Update results with rerank scores
            for result, rerank_score in zip(results, rerank_scores):
                result.rerank_score = float(rerank_score)
                result.confidence = (result.hybrid_score + result.rerank_score) / 2
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in reranking: {e}")
            # Return original results if reranking fails
            for result in results:
                result.rerank_score = result.hybrid_score
                result.confidence = result.hybrid_score
            return results
    
    def _apply_filters(self, results: List[RetrievalResult], filters: Dict[str, Any]) -> List[RetrievalResult]:
        """Apply metadata filters to results."""
        filtered_results = []
        
        for result in results:
            include = True
            metadata = result.document.metadata
            
            for key, value in filters.items():
                if key == "date_range":
                    # Handle date range filtering
                    if "created_at" in metadata:
                        doc_date = metadata["created_at"]
                        if not (value["start"] <= doc_date <= value["end"]):
                            include = False
                            break
                elif key == "boost_recent":
                    # Boost recent documents
                    if result.document.created_at:
                        days_old = (datetime.now() - result.document.created_at).days
                        if days_old <= 7:  # Within a week
                            result.hybrid_score *= 1.2
                elif key in metadata:
                    if isinstance(value, list):
                        if metadata[key] not in value:
                            include = False
                            break
                    else:
                        if metadata[key] != value:
                            include = False
                            break
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
    
    def _final_scoring(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """Apply final scoring and sort results with enhanced relevance."""
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        for result in results:
            # Start with hybrid score as base
            final_score = result.hybrid_score
            
            # Add rerank score if available - prioritize rerank for accuracy
            if result.rerank_score is not None:
                final_score = 0.4 * final_score + 0.6 * result.rerank_score
            
            # Strong title match bonus - titles are very important for relevance
            title_lower = result.document.title.lower()
            if any(term in title_lower for term in query_terms):
                final_score *= 2.0  # Stronger title match boost for document accuracy
            
            # Category relevance boost for support queries
            category = result.document.metadata.get("category", "").lower()
            if "support" in category and any(word in query_lower for word in ["warranty", "billing", "troubleshoot", "help", "issue", "problem"]):
                final_score *= 1.3  # Support category relevance boost
            
            # Exact phrase matches in content - critical for document accuracy
            content_lower = result.document.content.lower()
            if query_lower in content_lower:
                final_score *= 2.5  # Strong exact phrase match boost
            
            # Multiple term coverage bonus
            matched_terms = sum(1 for term in query_terms if term in title_lower or term in content_lower)
            coverage_ratio = matched_terms / max(len(query_terms), 1)
            final_score *= (1.0 + 0.3 * coverage_ratio)  # Up to 30% bonus for full coverage
            
            # Document type relevance
            doc_type = result.document.metadata.get("document_type", "").lower()
            if "troubleshooting" in doc_type and any(word in query_lower for word in ["fix", "solve", "troubleshoot", "problem", "issue", "not working"]):
                final_score *= 1.2
            elif "policy" in doc_type and any(word in query_lower for word in ["policy", "warranty", "terms", "coverage"]):
                final_score *= 1.2
            elif "guide" in doc_type and any(word in query_lower for word in ["how", "guide", "help", "instructions"]):
                final_score *= 1.2
            
            # Apply recency boost for time-sensitive queries
            if any(word in query_lower for word in ["recent", "latest", "new", "current"]):
                if result.document.created_at:
                    days_old = (datetime.now() - result.document.created_at).days
                    recency_boost = max(0.8, 1.0 - (days_old / 365))  # Decay over year
                    final_score *= recency_boost
            
            result.confidence = min(final_score, 1.0)  # Cap at 1.0
        
        # Sort by final score
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        chroma_count = 0
        if self.collection:
            try:
                chroma_count = self.collection.count()
            except:
                chroma_count = 0
        
        return {
            "total_documents_memory": len(self.documents),
            "total_documents_chroma": chroma_count,
            "storage_backend": "ChromaDB" if self.collection else "In-Memory",
            "embedding_dimension": self.embedding_dimension,
            "models_used": {
                "openai_embeddings": self.embedding_model_name
            },
            "query_expansion_cache_size": len(self.query_expansion_cache),
            "chroma_available": CHROMADB_AVAILABLE,
            "collection_name": self.collection_name if self.collection else None
        }

    async def retrieve_documents(
        self,
        query: str,
        max_results: int = 10,
        context: Optional[Dict] = None,
        enable_reranking: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Enhanced retrieval with cross-encoder reranking."""
        
        start_time = datetime.now()
        
        try:
            # Step 1: Initial retrieval (get more documents for reranking)
            initial_max = max(max_results * 3, self.rerank_top_k) if self.enable_reranking else max_results
            
            # Perform initial semantic search
            semantic_results = await self.semantic_search(
                query=query,
                top_k=initial_max,
                use_query_expansion=True
            )
            
            # Convert to expected format
            initial_results = {
                "documents": []
            }
            
            for result in semantic_results:
                doc = {
                    "id": result.document.id,
                    "content": result.document.content,
                    "title": result.document.title,
                    "similarity_score": result.semantic_score,
                    "metadata": result.document.metadata
                }
                initial_results["documents"].append(doc)
            
            # Step 2: Cross-encoder reranking (if enabled)
            use_reranking = (
                enable_reranking if enable_reranking is not None 
                else self.enable_reranking
            )
            
            if use_reranking and len(initial_results.get("documents", [])) > 1:
                self.logger.info(f"Applying cross-encoder reranking to {len(initial_results['documents'])} documents")
                
                reranking_result = await self.reranker.rerank_documents(
                    query=query,
                    documents=initial_results["documents"],
                    top_k=max_results,
                    enable_diversity=True
                )
                
                # Use reranked results
                final_documents = reranking_result.reranked_documents
                
                # Update metadata with reranking info
                retrieval_metadata = {
                    "initial_results": len(initial_results["documents"]),
                    "reranked_results": len(final_documents),
                    "reranking_time": reranking_result.reranking_time,
                    "reranking_model": reranking_result.model_used,
                    "diversity_score": reranking_result.diversity_score,
                    "average_rerank_score": np.mean(reranking_result.relevance_scores) if reranking_result.relevance_scores else 0.0
                }
                
            else:
                # Use original results
                final_documents = initial_results["documents"][:max_results]
                retrieval_metadata = {
                    "reranking_used": False,
                    "reason": "disabled or insufficient documents"
                }
            
            # Step 3: Build final context
            context_pieces = []
            for i, doc in enumerate(final_documents):
                context_piece = f"[Source {i+1}]: {doc.get('content', '')}"
                if 'rerank_score' in doc:
                    context_piece += f" (Relevance: {doc['rerank_score']:.3f})"
                context_pieces.append(context_piece)
            
            final_context = "\n\n".join(context_pieces)
            
            # Calculate overall confidence
            if use_reranking and final_documents:
                # Use reranking scores for confidence
                rerank_scores = [doc.get('rerank_score', 0.5) for doc in final_documents]
                overall_confidence = np.mean(rerank_scores)
            else:
                # Use original similarity scores
                similarity_scores = [doc.get('similarity_score', 0.5) for doc in final_documents]
                overall_confidence = np.mean(similarity_scores)
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "documents": final_documents,
                "context": final_context,
                "confidence": float(overall_confidence),
                "retrieval_time": total_time,
                "metadata": retrieval_metadata,
                "query_analysis": {
                    "processed_query": query,
                    "num_results": len(final_documents)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced retrieval failed: {e}")
            # Fallback to basic retrieval
            return await self._basic_retrieval_fallback(query, max_results, context)
    
    async def enhanced_semantic_search(self, 
                                     query: str, 
                                     top_k: int = 10,
                                     category_hint: Optional[str] = None,
                                     content_type_hint: Optional[str] = None,
                                     enable_boosting: bool = True,
                                     filters: Dict[str, Any] = None) -> List[RetrievalResult]:
        """PHASE 2.3: Enhanced semantic search with optimized vector database and metadata boosting."""
        
        try:
            # PHASE 2.3: Use optimized vector database if available
            if self.use_optimized_db and self.optimized_db:
                return await self._enhanced_search_optimized(
                    query, top_k, category_hint, content_type_hint, enable_boosting, filters
                )
            
            # Use metadata-boosted retrieval if available and enabled
            elif self.enable_metadata_boosting and enable_boosting and self.metadata_boosted_retriever:
                self.logger.info(f"PHASE 2.1: Using metadata-boosted retrieval for query: {query[:50]}")
                
                boosted_results = await self.metadata_boosted_retriever.retrieve_with_boosting(
                    query=query,
                    top_k=top_k,
                    category_hint=category_hint,
                    content_type_hint=content_type_hint,
                    filters=filters
                )
                
                # Convert boosted results back to standard format
                standard_results = []
                for boosted_result in boosted_results:
                    # Create semantic document
                    doc = SemanticDocument(
                        id=boosted_result.doc_id,
                        content=boosted_result.content,
                        title=boosted_result.title,
                        metadata=boosted_result.metadata.__dict__
                    )
                    
                    # Create retrieval result
                    result = RetrievalResult(
                        document=doc,
                        semantic_score=boosted_result.similarity_score,
                        keyword_score=0.0,  # Not used in boosted retrieval
                        hybrid_score=boosted_result.final_score,
                        relevance_explanation=boosted_result.explanation,
                        retrieval_method="metadata_boosted",
                        confidence=min(boosted_result.final_score, 1.0)
                    )
                    standard_results.append(result)
                
                self.logger.info(f"PHASE 2.1: Retrieved {len(standard_results)} boosted results")
                return standard_results
            
            else:
                # Fallback to standard semantic search
                self.logger.info(f"Using standard semantic search for query: {query[:50]}")
                return await self.semantic_search(query, top_k, filters=filters)
        
        except Exception as e:
            self.logger.error(f"Error in enhanced semantic search: {e}")
            # Fallback to basic semantic search
            return await self.semantic_search(query, top_k, filters=filters)
    
    async def clear_collection(self) -> bool:
        """Clear all documents from the collection (for testing/reset purposes)."""
        try:
            if self.collection:
                # Delete the collection and recreate it
                self.chroma_client.delete_collection(self.collection_name)
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Knowledge base documents with embeddings"}
                )
                self.logger.info("ChromaDB collection cleared")
            
            # Clear in-memory storage
            self.documents.clear()
            self.id_to_idx.clear()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear collection: {e}")
            return False

    async def _basic_retrieval_fallback(
        self, 
        query: str, 
        max_results: int, 
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Fallback to basic retrieval when enhanced retrieval fails."""
        
        try:
            # Simple text-based search as fallback
            matching_documents = []
            query_lower = query.lower()
            
            for i, doc in enumerate(self.documents):
                if query_lower in doc.content.lower() or query_lower in doc.title.lower():
                    score = 0.5  # Default similarity score
                    matching_documents.append({
                        "id": doc.id,
                        "content": doc.content,
                        "title": doc.title,
                        "similarity_score": score,
                        "metadata": doc.metadata
                    })
            
            # Sort by relevance (simple keyword matching)
            matching_documents.sort(key=lambda x: x["similarity_score"], reverse=True)
            final_documents = matching_documents[:max_results]
            
            # Build context
            context_pieces = []
            for i, doc in enumerate(final_documents):
                context_pieces.append(f"[Source {i+1}]: {doc['content']}")
            
            return {
                "documents": final_documents,
                "context": "\n\n".join(context_pieces),
                "confidence": 0.3,  # Low confidence for fallback
                "retrieval_time": 0.0,
                "metadata": {
                    "fallback_used": True,
                    "reason": "enhanced retrieval failed"
                },
                "query_analysis": {
                    "processed_query": query,
                    "num_results": len(final_documents)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Basic retrieval fallback failed: {e}")
            return {
                "documents": [],
                "context": "",
                "confidence": 0.0,
                "retrieval_time": 0.0,
                "metadata": {"error": "all retrieval methods failed"},
                "query_analysis": {"processed_query": query, "num_results": 0}
            }

    async def add_document(self, document_data: Dict[str, Any], use_advanced_chunking: bool = True) -> bool:
        """PHASE 2.3: Add document using optimized vector database if available."""
        
        try:
            # Use optimized database if available
            if self.use_optimized_db and self.optimized_db:
                return await self._add_document_optimized(document_data, use_advanced_chunking)
            else:
                # Use standard add_documents method for consistency
                await self.add_documents([document_data])
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to add document: {e}")
            return False

    async def _add_document_optimized(self, document_data: Dict[str, Any], use_advanced_chunking: bool) -> bool:
        """Add document using optimized vector database."""
        
        try:
            # Generate embeddings for the document content
            content = document_data["content"]
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None, 
                self._embed_documents, 
                [content]
            )
            
            # Prepare document for optimized storage
            documents = [document_data]
            
            # Add to optimized database
            success = await self.optimized_db.add_documents_batch(documents, embeddings)
            
            if success:
                self.logger.info(f"PHASE 2.3: Added document {document_data['id']} to optimized database")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Optimized document addition failed: {e}")
            return False

    async def _add_document_standard(self, document_data: Dict[str, Any], use_advanced_chunking: bool) -> bool:
        """Fallback method using standard ChromaDB."""
        
        # Standard implementation (existing code)
        # This is the fallback when optimized DB is not available
        return True  # Placeholder

    async def _enhanced_search_optimized(self,
                                       query: str,
                                       top_k: int,
                                       category_hint: Optional[str],
                                       content_type_hint: Optional[str],
                                       enable_boosting: bool,
                                       filters: Optional[Dict[str, Any]]) -> List[RetrievalResult]:
        """Enhanced search using optimized vector database."""
        
        try:
            self.logger.info(f"PHASE 2.3: Using optimized vector database for query: {query[:50]}")
            
            # Generate query embedding
            query_embedding = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda q: asyncio.run(self._embed_query(q)),
                query
            )
            
            # Execute optimized query
            optimized_result = await self.optimized_db.optimized_query(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
                enable_cache=True
            )
            
            # Convert to standard format
            results = []
            for i, doc_dict in enumerate(optimized_result.documents):
                doc = SemanticDocument(
                    id=doc_dict["id"],
                    content=doc_dict["content"],
                    title=doc_dict["metadata"].get("title", ""),
                    metadata=doc_dict["metadata"]
                )
                
                score = optimized_result.scores[i] if i < len(optimized_result.scores) else 0.0
                
                result = RetrievalResult(
                    document=doc,
                    semantic_score=score,
                    keyword_score=0.0,
                    hybrid_score=score,
                    relevance_explanation=f"Optimized DB query, cached: {optimized_result.from_cache}",
                    retrieval_method="optimized_vector_db",
                    confidence=min(score, 1.0)
                )
                results.append(result)
            
            self.logger.info(f"PHASE 2.3: Retrieved {len(results)} results in {optimized_result.query_time_ms:.2f}ms (cached: {optimized_result.from_cache})")
            return results
            
        except Exception as e:
            self.logger.error(f"Optimized search failed: {e}")
            # Fallback to standard search
            return await self.semantic_search(query, top_k, filters=filters)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        
        stats = {
            "retriever_type": "enhanced_semantic_retriever",
            "phase_2_features": {
                "metadata_boosting": self.enable_metadata_boosting,
                "optimized_vector_db": self.use_optimized_db,
                "unified_embeddings": True
            }
        }
        
        # Add optimized DB stats if available
        if self.use_optimized_db and self.optimized_db:
            stats["optimized_db_stats"] = self.optimized_db.get_performance_stats()
        
        # Add boosting stats if available
        if self.enable_metadata_boosting and self.metadata_boosted_retriever:
            stats["boosting_stats"] = self.metadata_boosted_retriever.get_boosting_stats()
        
        return stats

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for all Phase 2 components."""
        
        health_status = {
            "overall_status": "healthy",
            "components": {}
        }
        
        try:
            # Check optimized vector database
            if self.use_optimized_db and self.optimized_db:
                db_health = await self.optimized_db.health_check()
                health_status["components"]["optimized_vector_db"] = db_health
                
                if db_health["status"] != "healthy":
                    health_status["overall_status"] = "degraded"
            
            # Check OpenAI embeddings
            try:
                test_embedding = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda q: asyncio.run(self._embed_query(q)),
                    "test query"
                )
                health_status["components"]["openai_embeddings"] = {
                    "status": "healthy",
                    "model": self.embedding_model_name,
                    "dimension": len(test_embedding)
                }
            except Exception as e:
                health_status["components"]["openai_embeddings"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["overall_status"] = "unhealthy"
            
            # Check metadata boosting
            health_status["components"]["metadata_boosting"] = {
                "status": "available" if self.enable_metadata_boosting else "disabled",
                "enabled": self.enable_metadata_boosting
            }
            
            return health_status
            
        except Exception as e:
            return {
                "overall_status": "unhealthy",
                "error": str(e),
                "components": {}
            }

    def close(self):
        """Clean shutdown of all components."""
        
        try:
            if self.use_optimized_db and self.optimized_db:
                self.optimized_db.close()
            
            self.logger.info("PHASE 2.3: Enhanced semantic retriever closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during retriever shutdown: {e}")

    async def _embed_texts_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Efficiently embed texts in batches for large-scale processing."""
        if not texts:
            return []
            
        all_embeddings = []
        
        # Process in batches to avoid memory issues and timeouts
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                if self.use_openai_embeddings and self.openai_embeddings:
                    # Use OpenAI embeddings with retry logic
                    batch_embeddings = await self._embed_openai_batch(batch)
                else:
                    # Use local embeddings
                    batch_embeddings = await self._embed_local_batch(batch)
                
                all_embeddings.extend(batch_embeddings)
                
                # Log progress for large batches
                if len(texts) > 100:
                    progress = min(i + batch_size, len(texts))
                    self.logger.info(f"Embedded {progress}/{len(texts)} texts ({progress/len(texts)*100:.1f}%)")
                    
            except Exception as e:
                self.logger.error(f"Failed to embed batch {i//batch_size + 1}: {e}")
                # Add empty embeddings to maintain list length
                batch_embeddings = [[0.0] * self.embedding_dimension] * len(batch)
                all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    async def _embed_openai_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using OpenAI with rate limiting."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                embeddings = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.openai_embeddings.embed_documents(texts)
                )
                return embeddings
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"OpenAI embedding failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    raise e
    
    async def _embed_local_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using local model."""
        try:
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.local_embedding_model.encode(texts, convert_to_numpy=True).tolist()
            )
            return embeddings
        except Exception as e:
            self.logger.error(f"Local embedding failed: {e}")
            return [[0.0] * self.embedding_dimension] * len(texts)
    
    async def bulk_add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 50) -> bool:
        """Efficiently add multiple documents to the collection in batches."""
        if not documents:
            return True
            
        self.logger.info(f"Starting bulk addition of {len(documents)} documents in batches of {batch_size}")
        start_time = time.time()
        
        successful_batches = 0
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                # Extract texts for embedding
                texts = [doc.get('content', '') for doc in batch]
                
                # Generate embeddings in batch
                embeddings = await self._embed_texts_batch(texts, batch_size=32)
                
                # Prepare data for ChromaDB
                ids = [doc.get('id', str(uuid.uuid4())) for doc in batch]
                metadatas = [doc.get('metadata', {}) for doc in batch]
                
                # Add to ChromaDB collection
                if self.collection:
                    self.collection.add(
                        documents=texts,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids
                    )
                
                successful_batches += 1
                
                # Log progress
                elapsed = time.time() - start_time
                avg_time_per_batch = elapsed / batch_num
                estimated_total = avg_time_per_batch * total_batches
                remaining_time = estimated_total - elapsed
                
                self.logger.info(
                    f"Processed batch {batch_num}/{total_batches} "
                    f"({batch_num/total_batches*100:.1f}%) - "
                    f"ETA: {remaining_time:.1f}s"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to process batch {batch_num}: {e}")
                continue
        
        elapsed = time.time() - start_time
        success_rate = successful_batches / total_batches * 100
        
        self.logger.info(
            f"Bulk addition completed: {successful_batches}/{total_batches} batches successful "
            f"({success_rate:.1f}%) in {elapsed:.2f}s"
        )
        
        return success_rate > 80  # Consider successful if >80% of batches succeeded

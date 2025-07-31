"""
Enhanced Hybrid Retriever
Advanced retrieval combining BM25, semantic search, and cross-encoder reranking
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import asyncio
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from collections import defaultdict
import math

# Core retrieval libraries
try:
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import faiss
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    RETRIEVAL_LIBS_AVAILABLE = True
except ImportError:
    RETRIEVAL_LIBS_AVAILABLE = False

@dataclass
class RetrievalResult:
    """Single retrieval result with comprehensive scoring."""
    
    document_id: str
    content: str
    title: str
    source_type: str
    
    # Scoring components
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    cross_encoder_score: float = 0.0
    final_score: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    relevance_explanation: str = ""
    
    # Quality indicators
    content_quality: float = 1.0
    freshness_score: float = 1.0
    authority_score: float = 1.0

@dataclass
class RetrievalStrategy:
    """Configuration for retrieval strategy."""
    
    # Weighting for different retrieval methods
    bm25_weight: float = 0.3
    semantic_weight: float = 0.4
    cross_encoder_weight: float = 0.3
    
    # Retrieval parameters
    max_candidates: int = 100
    final_results: int = 10
    min_score_threshold: float = 0.1
    
    # Query processing
    enable_query_expansion: bool = True
    enable_query_decomposition: bool = True
    
    # Reranking
    enable_cross_encoder: bool = True
    cross_encoder_top_k: int = 50
    
    # Filtering
    enable_content_filtering: bool = True
    min_content_length: int = 50
    max_content_length: int = 5000

class RetrievalMethod(Enum):
    """Available retrieval methods."""
    BM25 = "bm25"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    CROSS_ENCODER = "cross_encoder"
    TFIDF = "tfidf"

class EnhancedHybridRetriever:
    """Advanced hybrid retrieval system with unified embedding strategy."""
    
    def __init__(self, 
                 embedding_model: str = "text-embedding-3-large",
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        
        self.logger = logging.getLogger(__name__)
        
        # PHASE 1.1: UNIFIED EMBEDDING STRATEGY - Use only OpenAI embeddings
        self.embedding_model_name = embedding_model
        self.embedding_dimension = 3072  # text-embedding-3-large dimension
        
        # Initialize models
        self.cross_encoder = None
        self.bm25_index = None
        self.tfidf_vectorizer = None
        self.faiss_index = None
        
        # Document storage
        self.documents = {}  # doc_id -> document
        self.document_embeddings = {}  # doc_id -> embedding
        self.document_chunks = {}  # doc_id -> list of chunks
        
        # Index mappings
        self.faiss_id_to_doc_id = {}  # faiss_index -> doc_id
        self.doc_id_to_faiss_id = {}  # doc_id -> faiss_index
        
        # Performance tracking
        self.retrieval_stats = defaultdict(list)
        
        # Initialize OpenAI embeddings
        try:
            from langchain_openai import OpenAIEmbeddings
            self.openai_embeddings = OpenAIEmbeddings(
                model=embedding_model,
                dimensions=self.embedding_dimension
            )
            self.logger.info(f"Loaded unified OpenAI embedding model: {embedding_model}")
        except Exception as e:
            self.logger.error(f"Failed to load OpenAI embeddings: {e}")
            raise RuntimeError(f"Cannot initialize without OpenAI embeddings: {e}")
        
        # Initialize cross-encoder for reranking if available
        if RETRIEVAL_LIBS_AVAILABLE:
            try:
                self.cross_encoder = CrossEncoder(cross_encoder_model)
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=10000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                self.logger.info(f"Loaded cross-encoder for reranking: {cross_encoder_model}")
            except Exception as e:
                self.logger.error(f"Failed to initialize reranking models: {e}")
        else:
            self.logger.warning("Advanced retrieval libraries not available, basic functionality only")

    async def add_document(self, 
                          doc_id: str,
                          content: str,
                          title: str = "",
                          metadata: Optional[Dict[str, Any]] = None,
                          chunk_size: int = 4000,  # Increased from 512 to fix truncation
                          chunk_overlap: int = 50) -> bool:
        """Add document to all retrieval indices."""
        
        try:
            # Store document
            self.documents[doc_id] = {
                'content': content,
                'title': title,
                'metadata': metadata or {},
                'added_at': time.time()
            }
            
            # Create chunks
            chunks = self._create_chunks(content, chunk_size, chunk_overlap)
            self.document_chunks[doc_id] = chunks
            
            # Add to BM25 index
            await self._add_to_bm25(doc_id, chunks)
            
            # Generate and store embeddings
            if self.openai_embeddings:
                embeddings = await self._generate_embeddings(chunks)
                self.document_embeddings[doc_id] = embeddings
                await self._add_to_faiss(doc_id, embeddings)
            
            # Add to TF-IDF (rebuild required)
            await self._rebuild_tfidf_if_needed()
            
            self.logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add document {doc_id}: {e}")
            return False

    async def retrieve(self, 
                      query: str,
                      strategy: Optional[RetrievalStrategy] = None,
                      method: RetrievalMethod = RetrievalMethod.HYBRID) -> List[RetrievalResult]:
        """Main retrieval method with multiple strategies."""
        
        start_time = time.time()
        strategy = strategy or RetrievalStrategy()
        
        try:
            # Query preprocessing
            processed_query = await self._preprocess_query(query, strategy)
            
            # Retrieve candidates using different methods
            candidates = await self._retrieve_candidates(processed_query, strategy, method)
            
            # Apply cross-encoder reranking if enabled
            if strategy.enable_cross_encoder and self.cross_encoder and candidates:
                candidates = await self._apply_cross_encoder_reranking(
                    query, candidates, strategy.cross_encoder_top_k
                )
            
            # Final scoring and ranking
            final_results = await self._final_ranking(candidates, strategy)
            
            # Apply post-processing filters
            filtered_results = await self._apply_filters(final_results, strategy)
            
            # Limit to requested number of results
            final_results = filtered_results[:strategy.final_results]
            
            # Track performance
            retrieval_time = time.time() - start_time
            self.retrieval_stats['retrieval_times'].append(retrieval_time)
            self.retrieval_stats['result_counts'].append(len(final_results))
            
            self.logger.info(f"Retrieved {len(final_results)} results in {retrieval_time:.3f}s")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Retrieval failed for query '{query}': {e}")
            return []

    async def _preprocess_query(self, query: str, strategy: RetrievalStrategy) -> Dict[str, Any]:
        """Preprocess query with expansion and decomposition."""
        
        processed = {
            'original': query,
            'cleaned': query.strip().lower(),
            'expanded': [query],
            'decomposed': [query],
            'keywords': []
        }
        
        try:
            # Basic cleaning
            processed['cleaned'] = ' '.join(query.split())
            
            # Extract keywords
            keywords = [word for word in query.split() if len(word) > 2]
            processed['keywords'] = keywords
            
            # Query expansion (if enabled)
            if strategy.enable_query_expansion:
                expanded_queries = await self._expand_query(query)
                processed['expanded'] = expanded_queries
            
            # Query decomposition (if enabled)
            if strategy.enable_query_decomposition:
                decomposed_queries = await self._decompose_query(query)
                processed['decomposed'] = decomposed_queries
            
        except Exception as e:
            self.logger.warning(f"Query preprocessing failed: {e}")
        
        return processed

    async def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms."""
        
        expanded = [query]
        
        try:
            # Simple synonym expansion (can be enhanced with WordNet or custom thesaurus)
            synonym_map = {
                'error': ['bug', 'issue', 'problem', 'failure'],
                'login': ['sign in', 'authentication', 'access'],
                'payment': ['billing', 'charge', 'transaction'],
                'setup': ['configuration', 'installation', 'initialization'],
                'help': ['support', 'assistance', 'guidance'],
                'api': ['interface', 'endpoint', 'service'],
                'account': ['profile', 'user', 'registration']
            }
            
            query_lower = query.lower()
            for term, synonyms in synonym_map.items():
                if term in query_lower:
                    for synonym in synonyms:
                        expanded_query = query_lower.replace(term, synonym)
                        if expanded_query != query_lower:
                            expanded.append(expanded_query)
            
            # Add keyword variations
            words = query.split()
            if len(words) > 1:
                # Add individual keywords
                expanded.extend(words)
                
                # Add partial combinations
                for i in range(len(words) - 1):
                    expanded.append(' '.join(words[i:i+2]))
            
        except Exception as e:
            self.logger.warning(f"Query expansion failed: {e}")
        
        return list(set(expanded))  # Remove duplicates

    async def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex queries into simpler sub-queries."""
        
        decomposed = [query]
        
        try:
            # Split on common conjunctions
            conjunctions = ['and', 'or', 'but', 'also', 'plus', 'with']
            
            for conj in conjunctions:
                if f' {conj} ' in query.lower():
                    parts = query.lower().split(f' {conj} ')
                    decomposed.extend([part.strip() for part in parts if part.strip()])
            
            # Split on question words for multi-part questions
            question_words = ['how', 'what', 'when', 'where', 'why', 'which', 'who']
            
            for qword in question_words:
                if query.lower().count(qword) > 1:
                    # Multiple question words suggest multiple questions
                    sentences = query.split('?')
                    for sentence in sentences:
                        if sentence.strip():
                            decomposed.append(sentence.strip() + '?')
            
        except Exception as e:
            self.logger.warning(f"Query decomposition failed: {e}")
        
        return list(set(decomposed))  # Remove duplicates

    async def _retrieve_candidates(self, 
                                 processed_query: Dict[str, Any],
                                 strategy: RetrievalStrategy,
                                 method: RetrievalMethod) -> List[RetrievalResult]:
        """Retrieve candidates using specified method(s)."""
        
        all_candidates = []
        
        try:
            if method == RetrievalMethod.HYBRID or method == RetrievalMethod.BM25:
                # BM25 retrieval
                bm25_results = await self._bm25_retrieve(processed_query, strategy)
                all_candidates.extend(bm25_results)
            
            if method == RetrievalMethod.HYBRID or method == RetrievalMethod.SEMANTIC:
                # Semantic retrieval
                semantic_results = await self._semantic_retrieve(processed_query, strategy)
                all_candidates.extend(semantic_results)
            
            if method == RetrievalMethod.TFIDF:
                # TF-IDF retrieval
                tfidf_results = await self._tfidf_retrieve(processed_query, strategy)
                all_candidates.extend(tfidf_results)
            
            # Merge and deduplicate candidates
            merged_candidates = await self._merge_candidates(all_candidates, strategy)
            
            return merged_candidates
            
        except Exception as e:
            self.logger.error(f"Candidate retrieval failed: {e}")
            return []

    async def _bm25_retrieve(self, 
                           processed_query: Dict[str, Any],
                           strategy: RetrievalStrategy) -> List[RetrievalResult]:
        """Retrieve using BM25 algorithm."""
        
        if not self.bm25_index:
            return []
        
        results = []
        
        try:
            # Get all query variations
            queries = processed_query['expanded'] + processed_query['decomposed']
            
            for query_text in queries:
                query_tokens = query_text.lower().split()
                scores = self.bm25_index.get_scores(query_tokens)
                
                # Get top candidates
                top_indices = np.argsort(scores)[::-1][:strategy.max_candidates]
                
                for idx in top_indices:
                    if scores[idx] > strategy.min_score_threshold:
                        doc_id, chunk_idx = self._get_doc_from_bm25_index(idx)
                        if doc_id and doc_id in self.documents:
                            chunk_content = self.document_chunks[doc_id][chunk_idx]
                            
                            result = RetrievalResult(
                                document_id=doc_id,
                                content=chunk_content,
                                title=self.documents[doc_id]['title'],
                                source_type=self.documents[doc_id]['metadata'].get('source_type', 'unknown'),
                                bm25_score=float(scores[idx]),
                                chunk_index=chunk_idx,
                                metadata=self.documents[doc_id]['metadata']
                            )
                            results.append(result)
            
        except Exception as e:
            self.logger.error(f"BM25 retrieval failed: {e}")
        
        return results

    async def _semantic_retrieve(self, 
                               processed_query: Dict[str, Any],
                               strategy: RetrievalStrategy) -> List[RetrievalResult]:
        """Retrieve using semantic similarity."""
        
        if not self.openai_embeddings or not self.faiss_index:
            return []
        
        results = []
        
        try:
            # Generate query embedding
            query_embedding = self.openai_embeddings.embed_query(processed_query['original'])
            
            # Search FAISS index
            scores, indices = self.faiss_index.search(
                query_embedding.astype('float32'), 
                strategy.max_candidates
            )
            
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score > strategy.min_score_threshold:
                    doc_id = self.faiss_id_to_doc_id.get(idx)
                    if doc_id and doc_id in self.documents:
                        # Find the chunk index for this embedding
                        chunk_idx = self._get_chunk_index_from_faiss_id(doc_id, idx)
                        chunk_content = self.document_chunks[doc_id][chunk_idx]
                        
                        result = RetrievalResult(
                            document_id=doc_id,
                            content=chunk_content,
                            title=self.documents[doc_id]['title'],
                            source_type=self.documents[doc_id]['metadata'].get('source_type', 'unknown'),
                            semantic_score=float(score),
                            chunk_index=chunk_idx,
                            metadata=self.documents[doc_id]['metadata']
                        )
                        results.append(result)
            
        except Exception as e:
            self.logger.error(f"Semantic retrieval failed: {e}")
        
        return results

    async def _tfidf_retrieve(self, 
                            processed_query: Dict[str, Any],
                            strategy: RetrievalStrategy) -> List[RetrievalResult]:
        """Retrieve using TF-IDF similarity."""
        
        if not self.tfidf_vectorizer:
            return []
        
        results = []
        
        try:
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([processed_query['original']])
            
            # Calculate similarities (this requires document vectors to be stored)
            # For now, return empty list - can be implemented when TF-IDF matrix is stored
            pass
            
        except Exception as e:
            self.logger.error(f"TF-IDF retrieval failed: {e}")
        
        return results

    async def _merge_candidates(self, 
                              candidates: List[RetrievalResult],
                              strategy: RetrievalStrategy) -> List[RetrievalResult]:
        """Merge and deduplicate candidates from different retrieval methods."""
        
        # Group by document_id and chunk_index
        candidate_groups = defaultdict(list)
        
        for candidate in candidates:
            key = f"{candidate.document_id}_{candidate.chunk_index}"
            candidate_groups[key].append(candidate)
        
        # Merge scores for duplicate candidates
        merged_candidates = []
        
        for key, group in candidate_groups.items():
            if len(group) == 1:
                merged_candidates.append(group[0])
            else:
                # Merge multiple candidates for same content
                base_candidate = group[0]
                
                # Combine scores
                total_bm25 = sum(c.bm25_score for c in group)
                total_semantic = sum(c.semantic_score for c in group)
                
                base_candidate.bm25_score = total_bm25
                base_candidate.semantic_score = total_semantic
                
                merged_candidates.append(base_candidate)
        
        return merged_candidates

    async def _apply_cross_encoder_reranking(self, 
                                           query: str,
                                           candidates: List[RetrievalResult],
                                           top_k: int) -> List[RetrievalResult]:
        """Apply cross-encoder reranking to top candidates."""
        
        if not self.cross_encoder or not candidates:
            return candidates
        
        try:
            # Take top candidates for reranking
            top_candidates = sorted(
                candidates, 
                key=lambda x: x.bm25_score + x.semantic_score, 
                reverse=True
            )[:top_k]
            
            # Prepare query-document pairs
            pairs = [(query, candidate.content) for candidate in top_candidates]
            
            # Get cross-encoder scores
            cross_scores = self.cross_encoder.predict(pairs)
            
            # Update candidates with cross-encoder scores
            for candidate, score in zip(top_candidates, cross_scores):
                candidate.cross_encoder_score = float(score)
            
            # Add remaining candidates with zero cross-encoder score
            remaining_candidates = candidates[top_k:]
            for candidate in remaining_candidates:
                candidate.cross_encoder_score = 0.0
            
            return top_candidates + remaining_candidates
            
        except Exception as e:
            self.logger.error(f"Cross-encoder reranking failed: {e}")
            return candidates

    async def _final_ranking(self, 
                           candidates: List[RetrievalResult],
                           strategy: RetrievalStrategy) -> List[RetrievalResult]:
        """Apply final ranking with weighted scores."""
        
        for candidate in candidates:
            # Normalize scores to 0-1 range
            normalized_bm25 = self._normalize_score(candidate.bm25_score, 'bm25')
            normalized_semantic = self._normalize_score(candidate.semantic_score, 'semantic')
            normalized_cross = self._normalize_score(candidate.cross_encoder_score, 'cross_encoder')
            
            # Calculate weighted final score
            candidate.final_score = (
                normalized_bm25 * strategy.bm25_weight +
                normalized_semantic * strategy.semantic_weight +
                normalized_cross * strategy.cross_encoder_weight
            )
            
            # Apply quality multipliers
            candidate.final_score *= candidate.content_quality
            candidate.final_score *= candidate.freshness_score
            candidate.final_score *= candidate.authority_score
        
        # Sort by final score
        return sorted(candidates, key=lambda x: x.final_score, reverse=True)

    def _normalize_score(self, score: float, score_type: str) -> float:
        """Normalize scores to 0-1 range based on score type."""
        
        if score_type == 'bm25':
            # BM25 scores are typically 0-10+
            return min(score / 10.0, 1.0)
        elif score_type == 'semantic':
            # Cosine similarity scores are typically 0-1
            return max(0.0, min(score, 1.0))
        elif score_type == 'cross_encoder':
            # Cross-encoder scores can vary, apply sigmoid
            return 1.0 / (1.0 + math.exp(-score))
        else:
            return score

    async def _apply_filters(self, 
                           results: List[RetrievalResult],
                           strategy: RetrievalStrategy) -> List[RetrievalResult]:
        """Apply post-processing filters."""
        
        if not strategy.enable_content_filtering:
            return results
        
        filtered = []
        
        for result in results:
            content_length = len(result.content)
            
            # Length filters
            if content_length < strategy.min_content_length:
                continue
            if content_length > strategy.max_content_length:
                continue
            
            # Quality filters
            if result.content_quality < 0.3:
                continue
            
            filtered.append(result)
        
        return filtered

    # Helper methods for index management
    
    def _create_chunks(self, content: str, chunk_size: int, overlap: int) -> List[str]:
        """Create overlapping chunks from content."""
        
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            
            # Try to break at sentence boundary
            if end < len(content):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = content[start:start + break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(content):
                break
        
        return [chunk for chunk in chunks if chunk.strip()]

    async def _add_to_bm25(self, doc_id: str, chunks: List[str]):
        """Add document chunks to BM25 index."""
        
        try:
            # Tokenize all chunks
            tokenized_chunks = [chunk.lower().split() for chunk in chunks]
            
            if self.bm25_index is None:
                # Initialize BM25 with first document
                self.bm25_index = BM25Okapi(tokenized_chunks)
                self.bm25_doc_mapping = [(doc_id, i) for i in range(len(chunks))]
            else:
                # Add to existing index (requires rebuilding)
                all_docs = []
                new_mapping = []
                
                # Add existing documents
                for existing_doc_id, existing_chunks in self.document_chunks.items():
                    if existing_doc_id != doc_id:  # Skip current doc if updating
                        tokenized_existing = [chunk.lower().split() for chunk in existing_chunks]
                        all_docs.extend(tokenized_existing)
                        new_mapping.extend([(existing_doc_id, i) for i in range(len(existing_chunks))])
                
                # Add new document
                all_docs.extend(tokenized_chunks)
                new_mapping.extend([(doc_id, i) for i in range(len(chunks))])
                
                # Rebuild index
                self.bm25_index = BM25Okapi(all_docs)
                self.bm25_doc_mapping = new_mapping
                
        except Exception as e:
            self.logger.error(f"Failed to add to BM25 index: {e}")

    async def _generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for document chunks."""
        
        if not self.openai_embeddings:
            return np.array([])
        
        try:
            embeddings = self.openai_embeddings.embed_documents(chunks)
            return embeddings
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            return np.array([])

    async def _add_to_faiss(self, doc_id: str, embeddings: np.ndarray):
        """Add embeddings to FAISS index."""
        
        if embeddings.size == 0:
            return
        
        try:
            if self.faiss_index is None:
                # Initialize FAISS index
                dimension = embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings.astype('float32'))
            
            # Add to index
            start_id = self.faiss_index.ntotal
            self.faiss_index.add(embeddings.astype('float32'))
            
            # Update mappings
            for i, embedding in enumerate(embeddings):
                faiss_id = start_id + i
                self.faiss_id_to_doc_id[faiss_id] = doc_id
                if doc_id not in self.doc_id_to_faiss_id:
                    self.doc_id_to_faiss_id[doc_id] = []
                self.doc_id_to_faiss_id[doc_id].append(faiss_id)
                
        except Exception as e:
            self.logger.error(f"Failed to add to FAISS index: {e}")

    async def _rebuild_tfidf_if_needed(self):
        """Rebuild TF-IDF index when new documents are added."""
        
        if not self.tfidf_vectorizer:
            return
        
        try:
            # Collect all document content
            all_content = []
            for doc_id, chunks in self.document_chunks.items():
                all_content.extend(chunks)
            
            if all_content:
                # Fit TF-IDF vectorizer
                self.tfidf_vectorizer.fit(all_content)
                
        except Exception as e:
            self.logger.error(f"Failed to rebuild TF-IDF index: {e}")

    def _get_doc_from_bm25_index(self, bm25_idx: int) -> Tuple[Optional[str], int]:
        """Get document ID and chunk index from BM25 index position."""
        
        if hasattr(self, 'bm25_doc_mapping') and bm25_idx < len(self.bm25_doc_mapping):
            return self.bm25_doc_mapping[bm25_idx]
        return None, 0

    def _get_chunk_index_from_faiss_id(self, doc_id: str, faiss_id: int) -> int:
        """Get chunk index from FAISS ID."""
        
        if doc_id in self.doc_id_to_faiss_id:
            faiss_ids = self.doc_id_to_faiss_id[doc_id]
            if faiss_id in faiss_ids:
                return faiss_ids.index(faiss_id)
        return 0

    async def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval performance statistics."""
        
        stats = {
            'total_documents': len(self.documents),
            'total_chunks': sum(len(chunks) for chunks in self.document_chunks.values()),
            'index_status': {
                'bm25_ready': self.bm25_index is not None,
                'faiss_ready': self.faiss_index is not None,
                'tfidf_ready': self.tfidf_vectorizer is not None,
                'cross_encoder_ready': self.cross_encoder is not None
            }
        }
        
        if self.retrieval_stats['retrieval_times']:
            stats['performance'] = {
                'avg_retrieval_time': np.mean(self.retrieval_stats['retrieval_times']),
                'avg_result_count': np.mean(self.retrieval_stats['result_counts']),
                'total_retrievals': len(self.retrieval_stats['retrieval_times'])
            }
        
        return stats

    async def clear_indices(self):
        """Clear all indices and reset the retriever."""
        
        self.documents.clear()
        self.document_embeddings.clear()
        self.document_chunks.clear()
        self.faiss_id_to_doc_id.clear()
        self.doc_id_to_faiss_id.clear()
        
        self.bm25_index = None
        self.faiss_index = None
        
        if hasattr(self, 'bm25_doc_mapping'):
            delattr(self, 'bm25_doc_mapping')
        
        self.logger.info("All retrieval indices cleared")
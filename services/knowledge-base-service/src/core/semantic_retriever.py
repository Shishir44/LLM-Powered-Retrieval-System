from typing import List, Dict, Any, Optional, Tuple, Union
import asyncio
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import faiss
from datetime import datetime, timedelta
import logging

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
    """State-of-the-art semantic retrieval system with multiple embedding models and reranking."""
    
    def __init__(self, 
                 primary_model: str = "all-MiniLM-L6-v2",
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 use_openai_embeddings: bool = True):
        
        # Initialize embedding models
        self.sentence_transformer = SentenceTransformer(primary_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        
        if use_openai_embeddings:
            self.openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        else:
            self.openai_embeddings = None
        
        # FAISS indices for fast similarity search
        self.faiss_index = None
        self.openai_faiss_index = None
        
        # Document storage
        self.documents: List[SemanticDocument] = []
        self.id_to_idx: Dict[str, int] = {}
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100,
            length_function=len
        )
        
        # Query enhancement
        self.query_expansion_cache = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents with semantic processing."""
        semantic_docs = []
        
        for doc in documents:
            # Create semantic document
            sem_doc = SemanticDocument(
                id=doc.get("id", str(len(self.documents))),
                content=doc["content"],
                title=doc.get("title", ""),
                metadata=doc.get("metadata", {}),
                created_at=datetime.now()
            )
            
            # Generate embeddings
            await self._generate_embeddings(sem_doc)
            semantic_docs.append(sem_doc)
        
        # Add to storage
        start_idx = len(self.documents)
        self.documents.extend(semantic_docs)
        
        # Update indices
        for i, doc in enumerate(semantic_docs):
            self.id_to_idx[doc.id] = start_idx + i
        
        # Rebuild FAISS indices
        await self._rebuild_indices()
        
        self.logger.info(f"Added {len(semantic_docs)} documents to semantic retriever")
    
    async def _generate_embeddings(self, document: SemanticDocument) -> None:
        """Generate multiple types of embeddings for a document."""
        try:
            # Sentence transformer embedding
            document.embedding = self.sentence_transformer.encode(
                document.content,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # OpenAI embedding if available
            if self.openai_embeddings:
                document.dense_embedding = await self.openai_embeddings.aembed_query(
                    document.content
                )
                document.dense_embedding = np.array(document.dense_embedding)
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings for document {document.id}: {e}")
            # Fallback to zero embedding
            document.embedding = np.zeros(self.sentence_transformer.get_sentence_embedding_dimension())
            if self.openai_embeddings:
                document.dense_embedding = np.zeros(3072)  # text-embedding-3-large dimension
    
    async def _rebuild_indices(self) -> None:
        """Rebuild FAISS indices for fast similarity search."""
        if not self.documents:
            return
        
        # Build sentence transformer index
        embeddings = np.array([doc.embedding for doc in self.documents])
        dimension = embeddings.shape[1]
        
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for normalized embeddings
        self.faiss_index.add(embeddings.astype(np.float32))
        
        # Build OpenAI embeddings index if available
        if self.openai_embeddings and all(doc.dense_embedding is not None for doc in self.documents):
            dense_embeddings = np.array([doc.dense_embedding for doc in self.documents])
            dense_dimension = dense_embeddings.shape[1]
            
            self.openai_faiss_index = faiss.IndexFlatIP(dense_dimension)
            self.openai_faiss_index.add(dense_embeddings.astype(np.float32))
        
        self.logger.info(f"Rebuilt FAISS indices for {len(self.documents)} documents")
    
    async def semantic_search(self, 
                            query: str, 
                            top_k: int = 10,
                            use_query_expansion: bool = True,
                            filters: Dict[str, Any] = None) -> List[RetrievalResult]:
        """Perform semantic search with query expansion and reranking."""
        
        if not self.documents or self.faiss_index is None:
            return []
        
        # Expand query if needed
        expanded_queries = [query]
        if use_query_expansion:
            expanded_queries.extend(await self._expand_query(query))
        
        # Search with multiple query variations
        all_results = []
        
        for exp_query in expanded_queries:
            # Semantic search results
            semantic_results = await self._semantic_search_single(exp_query, top_k * 2)
            
            # Keyword search results
            keyword_results = await self._keyword_search(exp_query, top_k * 2)
            
            # Combine and score
            combined_results = await self._combine_results(
                semantic_results, keyword_results, exp_query
            )
            
            all_results.extend(combined_results)
        
        # Remove duplicates and apply filters
        unique_results = self._remove_duplicates(all_results)
        if filters:
            unique_results = self._apply_filters(unique_results, filters)
        
        # Rerank with cross-encoder
        reranked_results = await self._rerank_results(query, unique_results)
        
        # Final scoring and sorting
        final_results = self._final_scoring(reranked_results, query)
        
        return final_results[:top_k]
    
    async def _semantic_search_single(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Perform semantic search for a single query."""
        try:
            # Generate query embedding
            query_embedding = self.sentence_transformer.encode(
                query, convert_to_numpy=True, normalize_embeddings=True
            )
            
            # Search with FAISS
            scores, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype(np.float32), top_k
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx < len(self.documents):  # Valid index
                    results.append((idx, float(score)))
            
            return results
            
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
            
            # Normalize scores (simple min-max normalization)
            sem_score_norm = min(sem_score, 1.0) if sem_score > 0 else 0.0
            kw_score_norm = min(kw_score / 10.0, 1.0) if kw_score > 0 else 0.0
            
            # Hybrid scoring with weights
            hybrid_score = 0.7 * sem_score_norm + 0.3 * kw_score_norm
            
            # Create retrieval result
            result = RetrievalResult(
                document=document,
                semantic_score=sem_score_norm,
                keyword_score=kw_score_norm,
                hybrid_score=hybrid_score,
                retrieval_method=\"hybrid\",
                relevance_explanation=f\"Semantic: {sem_score_norm:.3f}, Keyword: {kw_score_norm:.3f}\"
            )\n            \n            combined_results.append(result)\n        \n        return combined_results\n    \n    async def _rerank_results(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:\n        \"\"\"Rerank results using cross-encoder for better relevance.\"\"\"\n        if len(results) <= 1:\n            return results\n        \n        try:\n            # Prepare query-document pairs for cross-encoder\n            pairs = [(query, result.document.content) for result in results]\n            \n            # Get rerank scores\n            rerank_scores = self.cross_encoder.predict(pairs)\n            \n            # Update results with rerank scores\n            for result, rerank_score in zip(results, rerank_scores):\n                result.rerank_score = float(rerank_score)\n                result.confidence = (result.hybrid_score + result.rerank_score) / 2\n            \n            return results\n            \n        except Exception as e:\n            self.logger.error(f\"Error in reranking: {e}\")\n            # Return original results if reranking fails\n            for result in results:\n                result.rerank_score = result.hybrid_score\n                result.confidence = result.hybrid_score\n            return results\n    \n    def _remove_duplicates(self, results: List[RetrievalResult]) -> List[RetrievalResult]:\n        \"\"\"Remove duplicate documents from results.\"\"\"\n        seen_ids = set()\n        unique_results = []\n        \n        for result in results:\n            if result.document.id not in seen_ids:\n                seen_ids.add(result.document.id)\n                unique_results.append(result)\n        \n        return unique_results\n    \n    def _apply_filters(self, results: List[RetrievalResult], filters: Dict[str, Any]) -> List[RetrievalResult]:\n        \"\"\"Apply metadata filters to results.\"\"\"\n        filtered_results = []\n        \n        for result in results:\n            include = True\n            metadata = result.document.metadata\n            \n            for key, value in filters.items():\n                if key == \"date_range\":\n                    # Handle date range filtering\n                    if \"created_at\" in metadata:\n                        doc_date = metadata[\"created_at\"]\n                        if not (value[\"start\"] <= doc_date <= value[\"end\"]):\n                            include = False\n                            break\n                elif key == \"boost_recent\":\n                    # Boost recent documents\n                    if result.document.created_at:\n                        days_old = (datetime.now() - result.document.created_at).days\n                        if days_old <= 7:  # Within a week\n                            result.hybrid_score *= 1.2\n                elif key in metadata:\n                    if isinstance(value, list):\n                        if metadata[key] not in value:\n                            include = False\n                            break\n                    else:\n                        if metadata[key] != value:\n                            include = False\n                            break\n            \n            if include:\n                filtered_results.append(result)\n        \n        return filtered_results\n    \n    def _final_scoring(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:\n        \"\"\"Apply final scoring and sort results.\"\"\"\n        for result in results:\n            # Combine all scores with weights\n            final_score = (\n                0.4 * result.hybrid_score +\n                0.4 * (result.rerank_score or 0.0) +\n                0.2 * result.confidence\n            )\n            \n            # Apply query-specific boosting\n            if query.lower() in result.document.title.lower():\n                final_score *= 1.3  # Title match boost\n            \n            # Apply recency boost for time-sensitive queries\n            if any(word in query.lower() for word in [\"recent\", \"latest\", \"new\", \"current\"]):\n                if result.document.created_at:\n                    days_old = (datetime.now() - result.document.created_at).days\n                    recency_boost = max(0.8, 1.0 - (days_old / 365))  # Decay over year\n                    final_score *= recency_boost\n            \n            result.confidence = final_score\n        \n        # Sort by final score\n        results.sort(key=lambda x: x.confidence, reverse=True)\n        return results\n    \n    async def _expand_query(self, query: str) -> List[str]:\n        \"\"\"Expand query with synonyms and related terms.\"\"\"\n        if query in self.query_expansion_cache:\n            return self.query_expansion_cache[query]\n        \n        # Simple query expansion (in practice, use more sophisticated methods)\n        expanded = []\n        \n        # Add key terms extraction\n        words = query.lower().split()\n        if len(words) > 1:\n            # Add individual important words\n            important_words = [w for w in words if len(w) > 3]\n            if important_words:\n                expanded.append(\" \".join(important_words))\n        \n        # Add question variations\n        if not query.lower().startswith((\"what\", \"how\", \"when\", \"where\", \"why\")):\n            expanded.append(f\"what is {query}\")\n            expanded.append(f\"how to {query}\")\n        \n        # Cache the result\n        self.query_expansion_cache[query] = expanded[:3]  # Limit to 3 expansions\n        return expanded[:3]\n    \n    async def get_similar_documents(self, document_id: str, top_k: int = 5) -> List[RetrievalResult]:\n        \"\"\"Find documents similar to a given document.\"\"\"\n        if document_id not in self.id_to_idx:\n            return []\n        \n        doc_idx = self.id_to_idx[document_id]\n        target_doc = self.documents[doc_idx]\n        \n        # Use the document content as query\n        return await self.semantic_search(\n            target_doc.content[:500],  # Use first 500 chars as query\n            top_k=top_k + 1  # +1 to exclude the source document\n        )\n    \n    def get_statistics(self) -> Dict[str, Any]:\n        \"\"\"Get retriever statistics.\"\"\"\n        return {\n            \"total_documents\": len(self.documents),\n            \"faiss_index_size\": self.faiss_index.ntotal if self.faiss_index else 0,\n            \"embedding_dimension\": self.sentence_transformer.get_sentence_embedding_dimension(),\n            \"models_used\": {\n                \"sentence_transformer\": self.sentence_transformer._modules[\"0\"].auto_model.name_or_path,\n                \"cross_encoder\": self.cross_encoder.model.name_or_path,\n                \"openai_embeddings\": \"text-embedding-3-large\" if self.openai_embeddings else None\n            },\n            \"query_expansion_cache_size\": len(self.query_expansion_cache)\n        }"
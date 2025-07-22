from typing import List, Dict, Any, Optional, Tuple
import math
from dataclasses import dataclass
from collections import Counter
import re
from .chunking import DocumentChunk

@dataclass
class RetrievalResult:
    """Represents a retrieval result with scoring."""
    chunk: DocumentChunk
    score: float
    retrieval_method: str
    metadata: Dict[str, Any]

class AdvancedRAGRetriever:
    """Advanced retrieval system with multiple ranking strategies."""
    
    def __init__(self, similarity_threshold: float = 0.7, max_results: int = 10):
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        
        # Document storage (in production this would be a vector database)
        self.chunks: List[DocumentChunk] = []
        self.term_frequencies: Dict[str, Dict[str, float]] = {}  # TF for BM25
        self.document_frequencies: Dict[str, int] = {}  # DF for BM25
        self.total_documents = 0
        
    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add chunks to the retrieval index."""
        self.chunks.extend(chunks)
        self._update_bm25_index(chunks)
        
    def _update_bm25_index(self, chunks: List[DocumentChunk]):
        """Update BM25 index with new chunks."""
        for chunk in chunks:
            doc_id = f"{chunk.document_id}_{chunk.chunk_id}"
            terms = self._tokenize(chunk.content)
            term_counts = Counter(terms)
            total_terms = len(terms)
            
            # Calculate term frequencies for this document
            self.term_frequencies[doc_id] = {}
            for term, count in term_counts.items():
                tf = count / total_terms
                self.term_frequencies[doc_id][term] = tf
                
                # Update document frequency
                if term not in self.document_frequencies:
                    self.document_frequencies[term] = 0
                self.document_frequencies[term] += 1
        
        self.total_documents = len(self.chunks)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Convert to lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return [word for word in words if len(word) > 2]  # Filter short words
    
    def _calculate_bm25_score(self, query_terms: List[str], chunk: DocumentChunk, k1: float = 1.5, b: float = 0.75) -> float:
        """Calculate BM25 score for a chunk given query terms."""
        doc_id = f"{chunk.document_id}_{chunk.chunk_id}"
        
        if doc_id not in self.term_frequencies:
            return 0.0
        
        score = 0.0
        doc_length = len(self._tokenize(chunk.content))
        
        # Average document length
        if self.total_documents > 0:
            avg_dl = sum(len(self._tokenize(c.content)) for c in self.chunks) / self.total_documents
        else:
            avg_dl = doc_length
        
        for term in query_terms:
            if term not in self.term_frequencies[doc_id]:
                continue
            
            tf = self.term_frequencies[doc_id][term]
            df = self.document_frequencies.get(term, 1)
            
            # BM25 formula
            idf = math.log((self.total_documents - df + 0.5) / (df + 0.5))
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / avg_dl))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def _calculate_tfidf_score(self, query_terms: List[str], chunk: DocumentChunk) -> float:
        """Calculate TF-IDF score for a chunk."""
        doc_terms = self._tokenize(chunk.content)
        doc_term_counts = Counter(doc_terms)
        doc_length = len(doc_terms)
        
        score = 0.0
        
        for term in query_terms:
            if term in doc_term_counts:
                tf = doc_term_counts[term] / doc_length
                df = self.document_frequencies.get(term, 1)
                idf = math.log(self.total_documents / df) if df > 0 else 0
                score += tf * idf
        
        return score
    
    def _calculate_semantic_score(self, query: str, chunk: DocumentChunk) -> float:
        """Simple semantic similarity based on keyword matching."""
        query_words = set(self._tokenize(query.lower()))
        chunk_words = set(self._tokenize(chunk.content.lower()))
        
        if not query_words or not chunk_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(query_words & chunk_words)
        union = len(query_words | chunk_words)
        
        return intersection / union if union > 0 else 0.0
    
    def search(self, query: str, filters: Dict[str, Any] = None, top_k: int = None) -> List[RetrievalResult]:
        """Search for relevant chunks using hybrid retrieval."""
        if not query.strip():
            return []
        
        top_k = top_k or self.max_results
        query_terms = self._tokenize(query)
        
        if not query_terms:
            return []
        
        results = []
        
        for chunk in self.chunks:
            # Apply filters
            if filters and not self._apply_filters(chunk, filters):
                continue
            
            # Calculate different scores
            bm25_score = self._calculate_bm25_score(query_terms, chunk)
            tfidf_score = self._calculate_tfidf_score(query_terms, chunk)
            semantic_score = self._calculate_semantic_score(query, chunk)
            
            # Combine scores with weights
            combined_score = (
                0.5 * bm25_score +
                0.3 * tfidf_score +
                0.2 * semantic_score
            )
            
            if combined_score >= self.similarity_threshold:
                results.append(RetrievalResult(
                    chunk=chunk,
                    score=combined_score,
                    retrieval_method="hybrid",
                    metadata={
                        "bm25_score": bm25_score,
                        "tfidf_score": tfidf_score,
                        "semantic_score": semantic_score,
                        "query_terms_matched": len(set(query_terms) & set(self._tokenize(chunk.content)))
                    }
                ))
        
        # Sort by score and return top results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _apply_filters(self, chunk: DocumentChunk, filters: Dict[str, Any]) -> bool:
        """Apply metadata filters to chunk."""
        metadata = chunk.metadata
        
        for key, value in filters.items():
            if key in metadata:
                if isinstance(value, list):
                    # Check if any filter value matches
                    chunk_value = metadata[key]
                    if isinstance(chunk_value, list):
                        if not any(v in chunk_value for v in value):
                            return False
                    else:
                        if chunk_value not in value:
                            return False
                else:
                    if metadata[key] != value:
                        return False
        
        return True
    
    def get_similar_chunks(self, chunk_id: str, top_k: int = 5) -> List[RetrievalResult]:
        """Find chunks similar to a given chunk."""
        target_chunk = None
        
        # Find the target chunk
        for chunk in self.chunks:
            if f"{chunk.document_id}_{chunk.chunk_id}" == chunk_id:
                target_chunk = chunk
                break
        
        if not target_chunk:
            return []
        
        # Use chunk content as query
        return self.search(target_chunk.content, top_k=top_k)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval system statistics."""
        return {
            "total_chunks": len(self.chunks),
            "total_unique_terms": len(self.document_frequencies),
            "avg_chunk_length": sum(len(chunk.content) for chunk in self.chunks) / max(len(self.chunks), 1),
            "similarity_threshold": self.similarity_threshold,
            "max_results": self.max_results
        }
    
    def rerank_results(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """Re-rank results using additional signals."""
        if len(results) <= 1:
            return results
        
        query_terms = set(self._tokenize(query))
        
        for result in results:
            chunk_terms = set(self._tokenize(result.chunk.content))
            
            # Boost score based on additional factors
            boost = 0.0
            
            # Recent documents get slight boost
            if 'created_at' in result.chunk.metadata:
                boost += 0.1
            
            # Exact phrase matches get boost
            if query.lower() in result.chunk.content.lower():
                boost += 0.2
            
            # Coverage boost (how many query terms are covered)
            coverage = len(query_terms & chunk_terms) / len(query_terms) if query_terms else 0
            boost += coverage * 0.1
            
            result.score += boost
            result.metadata['rerank_boost'] = boost
        
        # Re-sort
        results.sort(key=lambda x: x.score, reverse=True)
        return results
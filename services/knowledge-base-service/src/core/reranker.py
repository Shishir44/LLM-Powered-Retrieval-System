"""
Cross-Encoder Reranking System
Advanced reranking of retrieved documents using cross-encoder models for improved precision.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
import numpy as np
from datetime import datetime

try:
    from sentence_transformers import CrossEncoder
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: sentence_transformers or torch not available: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    CrossEncoder = None
    torch = None

@dataclass
class RerankingResult:
    """Result of document reranking."""
    original_documents: List[Dict[str, Any]]
    reranked_documents: List[Dict[str, Any]]
    relevance_scores: List[float]
    reranking_time: float
    model_used: str
    diversity_score: float

@dataclass
class RerankingConfig:
    """Configuration for reranking models and strategies."""
    primary_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    fallback_model: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    max_sequence_length: int = 512
    batch_size: int = 32
    enable_diversity: bool = True
    diversity_threshold: float = 0.7
    min_relevance_score: float = 0.3
    max_rerank_documents: int = 100

class CrossEncoderReranker:
    """Advanced document reranking using cross-encoder models."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Reranking configuration
        self.rerank_config = RerankingConfig()
        if hasattr(config, 'reranking'):
            reranking_config = config.reranking
            if hasattr(reranking_config, '__dict__'):
                # If it's a dataclass/object, access attributes
                for key, value in reranking_config.__dict__.items():
                    if hasattr(self.rerank_config, key):
                        setattr(self.rerank_config, key, value)
            elif isinstance(reranking_config, dict):
                # If it's a dictionary, access items
                for key, value in reranking_config.items():
                    if hasattr(self.rerank_config, key):
                        setattr(self.rerank_config, key, value)
        
        # Initialize models
        self.models = {}
        self.current_model = None
        self._initialize_models()
        
        # Performance tracking
        self.reranking_stats = {
            "total_requests": 0,
            "total_documents_reranked": 0,
            "average_reranking_time": 0.0,
            "model_usage": {}
        }

    def _initialize_models(self):
        """Initialize cross-encoder models."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE or CrossEncoder is None:
            self.logger.warning("sentence_transformers not available, cross-encoder reranking disabled")
            self.models = {}
            self.current_model = None
            return
            
        try:
            # Load primary model
            self.logger.info(f"Loading primary reranking model: {self.rerank_config.primary_model}")
            self.models["primary"] = CrossEncoder(
                self.rerank_config.primary_model,
                max_length=self.rerank_config.max_sequence_length
            )
            self.current_model = "primary"
            
            # Load fallback model
            self.logger.info(f"Loading fallback reranking model: {self.rerank_config.fallback_model}")
            self.models["fallback"] = CrossEncoder(
                self.rerank_config.fallback_model,
                max_length=self.rerank_config.max_sequence_length
            )
            
            self.logger.info("Cross-encoder models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load cross-encoder models: {e}")
            self.models = {}
            self.current_model = None

    async def rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        enable_diversity: Optional[bool] = None
    ) -> RerankingResult:
        """Rerank documents using cross-encoder models."""
        
        start_time = datetime.now()
        
        try:
            # Validate inputs
            if not documents or not self.models:
                return self._create_fallback_result(documents, start_time)
            
            # Limit documents for performance
            max_docs = min(len(documents), self.rerank_config.max_rerank_documents)
            documents_to_rerank = documents[:max_docs]
            
            # Prepare query-document pairs
            query_doc_pairs = []
            for doc in documents_to_rerank:
                doc_text = self._extract_document_text(doc)
                query_doc_pairs.append([query, doc_text])
            
            # Get relevance scores
            relevance_scores = await self._compute_relevance_scores(query_doc_pairs)
            
            # Apply minimum relevance threshold
            filtered_results = []
            for i, score in enumerate(relevance_scores):
                if score >= self.rerank_config.min_relevance_score:
                    doc_with_score = documents_to_rerank[i].copy()
                    doc_with_score["rerank_score"] = float(score)
                    doc_with_score["original_rank"] = i
                    filtered_results.append((doc_with_score, score))
            
            # Sort by relevance score
            filtered_results.sort(key=lambda x: x[1], reverse=True)
            
            # Apply diversity if enabled
            enable_div = enable_diversity if enable_diversity is not None else self.rerank_config.enable_diversity
            if enable_div and len(filtered_results) > 1:
                reranked_docs, diversity_score = await self._apply_diversity_reranking(
                    filtered_results, query
                )
            else:
                reranked_docs = [doc for doc, score in filtered_results]
                diversity_score = 0.0
            
            # Apply top_k limit
            if top_k:
                reranked_docs = reranked_docs[:top_k]
            
            # Calculate metrics
            reranking_time = (datetime.now() - start_time).total_seconds()
            final_scores = [doc.get("rerank_score", 0.0) for doc in reranked_docs]
            
            # Update stats
            self._update_stats(len(documents_to_rerank), reranking_time)
            
            return RerankingResult(
                original_documents=documents_to_rerank,
                reranked_documents=reranked_docs,
                relevance_scores=final_scores,
                reranking_time=reranking_time,
                model_used=self.current_model,
                diversity_score=diversity_score
            )
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            return self._create_fallback_result(documents, start_time)

    async def _compute_relevance_scores(self, query_doc_pairs: List[List[str]]) -> List[float]:
        """Compute relevance scores using cross-encoder model."""
        
        try:
            model = self.models.get(self.current_model)
            if not model:
                raise ValueError(f"Model {self.current_model} not available")
            
            # Process in batches for memory efficiency
            all_scores = []
            batch_size = self.rerank_config.batch_size
            
            for i in range(0, len(query_doc_pairs), batch_size):
                batch = query_doc_pairs[i:i + batch_size]
                
                # Run inference
                if torch is not None:
                    with torch.no_grad():
                        batch_scores = model.predict(batch)
                else:
                    batch_scores = model.predict(batch)
                
                # Convert to list and normalize
                if isinstance(batch_scores, np.ndarray):
                    batch_scores = batch_scores.tolist()
                elif torch is not None and isinstance(batch_scores, torch.Tensor):
                    batch_scores = batch_scores.cpu().numpy().tolist()
                
                all_scores.extend(batch_scores)
            
            # Normalize scores to 0-1 range
            if all_scores:
                min_score = min(all_scores)
                max_score = max(all_scores)
                if max_score > min_score:
                    all_scores = [(score - min_score) / (max_score - min_score) for score in all_scores]
            
            return all_scores
            
        except Exception as e:
            self.logger.error(f"Score computation failed with {self.current_model}: {e}")
            
            # Try fallback model
            if self.current_model == "primary" and "fallback" in self.models:
                self.logger.info("Switching to fallback model")
                self.current_model = "fallback"
                return await self._compute_relevance_scores(query_doc_pairs)
            
            # Return uniform scores as last resort
            return [0.5] * len(query_doc_pairs)

    async def _apply_diversity_reranking(
        self, 
        scored_documents: List[Tuple[Dict, float]], 
        query: str
    ) -> Tuple[List[Dict], float]:
        """Apply diversity-aware reranking to avoid redundant results."""
        
        if len(scored_documents) <= 1:
            return [doc for doc, score in scored_documents], 1.0
        
        try:
            # Extract document texts for similarity computation
            doc_texts = [self._extract_document_text(doc) for doc, score in scored_documents]
            
            # Simple diversity algorithm: MMR-like approach
            selected_docs = []
            remaining_docs = scored_documents.copy()
            
            # Always select the highest scoring document first
            best_doc, best_score = remaining_docs.pop(0)
            selected_docs.append(best_doc)
            selected_texts = [self._extract_document_text(best_doc)]
            
            # Select remaining documents balancing relevance and diversity
            while remaining_docs and len(selected_docs) < len(scored_documents):
                best_candidate = None
                best_mmr_score = -1
                best_idx = -1
                
                for idx, (doc, relevance_score) in enumerate(remaining_docs):
                    doc_text = self._extract_document_text(doc)
                    
                    # Calculate maximum similarity to already selected documents
                    max_similarity = 0.0
                    for selected_text in selected_texts:
                        similarity = self._calculate_text_similarity(doc_text, selected_text)
                        max_similarity = max(max_similarity, similarity)
                    
                    # MMR score: balance relevance and diversity
                    lambda_param = 0.7  # Weight for relevance vs diversity
                    mmr_score = lambda_param * relevance_score - (1 - lambda_param) * max_similarity
                    
                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_candidate = doc
                        best_idx = idx
                
                if best_candidate:
                    selected_docs.append(best_candidate)
                    selected_texts.append(self._extract_document_text(best_candidate))
                    remaining_docs.pop(best_idx)
                else:
                    break
            
            # Calculate diversity score
            if len(selected_docs) > 1:
                total_similarity = 0
                comparisons = 0
                for i in range(len(selected_texts)):
                    for j in range(i + 1, len(selected_texts)):
                        similarity = self._calculate_text_similarity(selected_texts[i], selected_texts[j])
                        total_similarity += similarity
                        comparisons += 1
                
                avg_similarity = total_similarity / comparisons if comparisons > 0 else 0
                diversity_score = 1.0 - avg_similarity
            else:
                diversity_score = 1.0
            
            return selected_docs, diversity_score
            
        except Exception as e:
            self.logger.error(f"Diversity reranking failed: {e}")
            # Fallback to relevance-only ranking
            return [doc for doc, score in scored_documents], 0.0

    def _extract_document_text(self, document: Dict[str, Any]) -> str:
        """Extract text content from document for processing."""
        
        # Try different possible text fields
        text_fields = ["content", "text", "body", "description", "summary"]
        
        for field in text_fields:
            if field in document and document[field]:
                text = str(document[field])
                # Truncate if too long
                max_length = self.rerank_config.max_sequence_length - 100  # Leave room for query
                if len(text) > max_length:
                    text = text[:max_length]  # Remove truncation ellipsis for reranking
                return text
        
        # Fallback: concatenate all string values
        text_parts = []
        for key, value in document.items():
            if isinstance(value, str) and len(value) > 10:
                text_parts.append(value)
        
        combined_text = " ".join(text_parts)
        if len(combined_text) > self.rerank_config.max_sequence_length - 100:
            combined_text = combined_text[:self.rerank_config.max_sequence_length - 100] + "..."
        
        return combined_text or "No content available"

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simple implementation)."""
        
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

    def _create_fallback_result(self, documents: List[Dict], start_time: datetime) -> RerankingResult:
        """Create fallback result when reranking fails."""
        
        reranking_time = (datetime.now() - start_time).total_seconds()
        
        return RerankingResult(
            original_documents=documents,
            reranked_documents=documents,
            relevance_scores=[0.5] * len(documents),
            reranking_time=reranking_time,
            model_used="fallback",
            diversity_score=0.0
        )

    def _update_stats(self, num_documents: int, reranking_time: float):
        """Update performance statistics."""
        
        self.reranking_stats["total_requests"] += 1
        self.reranking_stats["total_documents_reranked"] += num_documents
        
        # Update average reranking time
        total_requests = self.reranking_stats["total_requests"]
        current_avg = self.reranking_stats["average_reranking_time"]
        self.reranking_stats["average_reranking_time"] = (
            (current_avg * (total_requests - 1) + reranking_time) / total_requests
        )
        
        # Update model usage
        model_usage = self.reranking_stats["model_usage"]
        model_usage[self.current_model] = model_usage.get(self.current_model, 0) + 1

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get reranking performance statistics."""
        return self.reranking_stats.copy()

    async def health_check(self) -> Dict[str, Any]:
        """Check health of reranking system."""
        
        health_status = {
            "status": "healthy" if self.models else "unhealthy",
            "models_loaded": list(self.models.keys()),
            "current_model": self.current_model,
            "performance_stats": self.get_performance_stats()
        }
        
        # Test inference if models are available
        if self.models:
            try:
                test_pairs = [["test query", "test document"]]
                await self._compute_relevance_scores(test_pairs)
                health_status["inference_test"] = "passed"
            except Exception as e:
                health_status["inference_test"] = f"failed: {str(e)}"
                health_status["status"] = "degraded"
        
        return health_status
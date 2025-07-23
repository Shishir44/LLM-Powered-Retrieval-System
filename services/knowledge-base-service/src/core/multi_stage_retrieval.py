from typing import List, Dict, Any, Optional, Tuple, Set
import asyncio
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_openai import OpenAIEmbeddings
import logging
from datetime import datetime
import heapq
import random

@dataclass
class RetrievalCandidate:
    """Candidate document in the retrieval pipeline."""
    id: str
    content: str
    title: str
    metadata: Dict[str, Any]
    initial_score: float
    rerank_score: Optional[float] = None
    diversity_score: Optional[float] = None
    final_score: Optional[float] = None
    retrieval_stage: str = "initial"
    source_queries: List[str] = None

@dataclass 
class RetrievalContext:
    """Context for the retrieval pipeline."""
    original_query: str
    expanded_queries: List[str]
    entities: List[str]
    topics: List[str]
    query_type: str
    complexity: str
    user_preferences: Dict[str, Any]
    retrieval_strategy: Dict[str, Any]

class MultiStageRetrievalPipeline:
    """Advanced multi-stage retrieval pipeline with reranking and diversity selection."""
    
    def __init__(self,
                 vector_database,
                 primary_model: str = "all-mpnet-base-v2",
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
                 use_openai_embeddings: bool = True):
        
        self.vector_database = vector_database
        self.sentence_transformer = SentenceTransformer(primary_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        
        if use_openai_embeddings:
            self.openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        else:
            self.openai_embeddings = None
        
        # Pipeline configuration
        self.config = {
            "initial_retrieval_size": 50,
            "rerank_size": 20,
            "diversity_selection_size": 10,
            "final_results_size": 5,
            "diversity_threshold": 0.7,
            "rerank_threshold": 0.5,
            "query_expansion_limit": 3
        }
        
        # Performance tracking
        self.retrieval_stats = {
            "total_queries": 0,
            "avg_retrieval_time": 0.0,
            "stage_performance": {},
            "query_type_performance": {}
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def retrieve(self, 
                      context: RetrievalContext,
                      index_name: str = "default") -> List[RetrievalCandidate]:
        """Main retrieval pipeline with multiple stages."""
        
        start_time = datetime.now()
        self.logger.info(f"Starting multi-stage retrieval for query: {context.original_query[:50]}...")
        
        try:
            # Stage 1: Initial Retrieval (Cast wide net)
            initial_candidates = await self._stage_1_initial_retrieval(context, index_name)
            self.logger.info(f"Stage 1: Retrieved {len(initial_candidates)} initial candidates")
            
            # Stage 2: Cross-encoder Reranking
            reranked_candidates = await self._stage_2_reranking(context, initial_candidates)
            self.logger.info(f"Stage 2: Reranked to {len(reranked_candidates)} candidates")
            
            # Stage 3: Diversity Selection
            diverse_candidates = await self._stage_3_diversity_selection(context, reranked_candidates)
            self.logger.info(f"Stage 3: Selected {len(diverse_candidates)} diverse candidates")
            
            # Stage 4: Final Scoring and Context Assembly
            final_results = await self._stage_4_final_scoring(context, diverse_candidates)
            self.logger.info(f"Stage 4: Final {len(final_results)} results assembled")
            
            # Update performance statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_stats(context, processing_time, len(final_results))
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in multi-stage retrieval pipeline: {e}")
            return []
    
    async def _stage_1_initial_retrieval(self, 
                                       context: RetrievalContext,
                                       index_name: str) -> List[RetrievalCandidate]:
        """Stage 1: Initial broad retrieval from multiple query variants."""
        
        candidates = []
        seen_ids = set()
        
        # Prepare all queries for retrieval 
        all_queries = [context.original_query] + context.expanded_queries
        
        # Limit query expansion based on complexity
        if context.complexity == "simple":
            all_queries = all_queries[:2]
        elif context.complexity == "moderate":
            all_queries = all_queries[:3]
        # For complex queries, use all available queries
        
        retrieval_tasks = []
        
        # Generate embeddings for all queries
        for query in all_queries:
            task = self._retrieve_for_query(query, index_name, context)
            retrieval_tasks.append(task)
        
        # Execute retrievals in parallel
        query_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
        
        # Aggregate results
        for i, results in enumerate(query_results):
            if isinstance(results, Exception):
                self.logger.warning(f"Query retrieval failed for query {i}: {results}")
                continue
            
            query = all_queries[i] if i < len(all_queries) else context.original_query
            
            for result in results:
                if result.id not in seen_ids:
                    candidate = RetrievalCandidate(
                        id=result.id,
                        content=result.document.content,
                        title=result.document.title,
                        metadata=result.document.metadata,
                        initial_score=result.score,
                        retrieval_stage="initial",
                        source_queries=[query]
                    )
                    candidates.append(candidate)
                    seen_ids.add(result.id)
                else:
                    # Update existing candidate with additional query source
                    for candidate in candidates:
                        if candidate.id == result.id:
                            candidate.source_queries.append(query)
                            # Boost score for multi-query matches
                            candidate.initial_score = max(candidate.initial_score, result.score * 1.1)
                            break
        
        # Sort by initial score and take top candidates
        candidates.sort(key=lambda x: x.initial_score, reverse=True)
        return candidates[:self.config["initial_retrieval_size"]]
    
    async def _retrieve_for_query(self, 
                                query: str, 
                                index_name: str,
                                context: RetrievalContext):
        """Retrieve documents for a single query."""
        
        try:
            # Generate query embedding
            if self.openai_embeddings and not context.retrieval_strategy.get("fast_mode", False):
                query_embedding = await self.openai_embeddings.aembed_query(query)
                query_vector = np.array(query_embedding)
            else:
                # Use sentence transformer for faster retrieval
                query_vector = self.sentence_transformer.encode(
                    query, convert_to_numpy=True, normalize_embeddings=True
                )
            
            # Build filters based on context
            filters = self._build_retrieval_filters(context)
            
            # Retrieve from vector database
            results = await self.vector_database.search(
                index_name=index_name,
                query_vector=query_vector,
                top_k=self.config["initial_retrieval_size"] // len(context.expanded_queries + [context.original_query]),
                filters=filters
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving for query '{query}': {e}")
            return []
    
    def _build_retrieval_filters(self, context: RetrievalContext) -> Dict[str, Any]:
        """Build filters for retrieval based on context."""
        
        filters = {}
        
        # Apply user preferences
        if context.user_preferences:
            if "preferred_sources" in context.user_preferences:
                filters["source"] = context.user_preferences["preferred_sources"]
            
            if "exclude_categories" in context.user_preferences:
                # This would need custom filter logic in the vector database
                pass
        
        # Apply temporal filters for time-sensitive queries
        if any(word in context.original_query.lower() for word in ["recent", "latest", "new", "current"]):
            # Boost recent documents (implementation depends on vector database)
            filters["boost_recent"] = True
        
        # Apply domain-specific filters based on entities
        if context.entities:
            # This could be used to filter by document categories/domains
            pass
        
        return filters
    
    async def _stage_2_reranking(self, 
                               context: RetrievalContext,
                               candidates: List[RetrievalCandidate]) -> List[RetrievalCandidate]:
        """Stage 2: Cross-encoder reranking for better relevance."""
        
        if len(candidates) <= 1:
            return candidates
        
        try:
            # Prepare query-document pairs for cross-encoder
            pairs = []
            for candidate in candidates:
                # Use the most relevant source query for reranking
                source_query = context.original_query
                if candidate.source_queries:
                    # Choose the query that likely produced the highest score
                    source_query = candidate.source_queries[0]
                
                pairs.append((source_query, candidate.content[:1000]))  # Limit content length
            
            # Get rerank scores in batches to handle large numbers of candidates
            batch_size = 32
            rerank_scores = []
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_scores = self.cross_encoder.predict(batch_pairs)
                rerank_scores.extend(batch_scores)
            
            # Update candidates with rerank scores
            for candidate, rerank_score in zip(candidates, rerank_scores):
                candidate.rerank_score = float(rerank_score)
                candidate.retrieval_stage = "reranked"
            
            # Filter candidates based on rerank threshold
            filtered_candidates = [
                c for c in candidates 
                if c.rerank_score >= self.config["rerank_threshold"]
            ]
            
            # If too few candidates pass threshold, keep top ones anyway
            if len(filtered_candidates) < 5:
                filtered_candidates = sorted(
                    candidates, 
                    key=lambda x: x.rerank_score, 
                    reverse=True
                )[:self.config["rerank_size"]]
            else:
                # Sort by rerank score and take top candidates
                filtered_candidates.sort(key=lambda x: x.rerank_score, reverse=True)
                filtered_candidates = filtered_candidates[:self.config["rerank_size"]]
            
            return filtered_candidates
            
        except Exception as e:
            self.logger.error(f"Error in reranking stage: {e}")
            # Return original candidates if reranking fails
            for candidate in candidates:
                candidate.rerank_score = candidate.initial_score
                candidate.retrieval_stage = "rerank_failed"
            return candidates[:self.config["rerank_size"]]
    
    async def _stage_3_diversity_selection(self, 
                                         context: RetrievalContext,
                                         candidates: List[RetrievalCandidate]) -> List[RetrievalCandidate]:
        """Stage 3: Select diverse candidates to avoid redundancy."""
        
        if len(candidates) <= self.config["diversity_selection_size"]:
            return candidates
        
        try:
            # Generate embeddings for content comparison
            content_embeddings = []
            for candidate in candidates:
                embedding = self.sentence_transformer.encode(
                    candidate.content[:500],  # Use first 500 chars for diversity calculation
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                content_embeddings.append(embedding)
            
            # Select diverse candidates using MMR (Maximal Marginal Relevance)
            selected_candidates = self._maximal_marginal_relevance_selection(
                candidates, content_embeddings, 
                lambda_param=0.7,  # Balance between relevance and diversity
                k=self.config["diversity_selection_size"]
            )
            
            # Update diversity scores
            for candidate in selected_candidates:
                candidate.diversity_score = self._calculate_diversity_score(
                    candidate, selected_candidates, content_embeddings, candidates
                )
                candidate.retrieval_stage = "diversified"
            
            return selected_candidates
            
        except Exception as e:
            self.logger.error(f"Error in diversity selection stage: {e}")
            return candidates[:self.config["diversity_selection_size"]]
    
    def _maximal_marginal_relevance_selection(self, 
                                            candidates: List[RetrievalCandidate],
                                            embeddings: List[np.ndarray],
                                            lambda_param: float = 0.7,
                                            k: int = 10) -> List[RetrievalCandidate]:
        """Select candidates using Maximal Marginal Relevance (MMR)."""
        
        if len(candidates) <= k:
            return candidates
        
        selected_indices = []
        remaining_indices = list(range(len(candidates)))
        
        # Start with the highest-scoring candidate
        best_idx = 0
        for i, candidate in enumerate(candidates):
            if candidate.rerank_score > candidates[best_idx].rerank_score:
                best_idx = i
        
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
        # Iteratively select candidates that maximize MMR
        while len(selected_indices) < k and remaining_indices:
            best_mmr_score = float('-inf')
            best_candidate_idx = None
            
            for idx in remaining_indices:
                # Relevance score (normalized)
                relevance = candidates[idx].rerank_score
                
                # Calculate max similarity to already selected candidates
                max_similarity = 0.0
                for selected_idx in selected_indices:
                    similarity = np.dot(embeddings[idx], embeddings[selected_idx])
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate_idx = idx
            
            if best_candidate_idx is not None:
                selected_indices.append(best_candidate_idx)
                remaining_indices.remove(best_candidate_idx)
            else:
                break
        
        return [candidates[i] for i in selected_indices]
    
    def _calculate_diversity_score(self, 
                                 candidate: RetrievalCandidate,
                                 selected_candidates: List[RetrievalCandidate],
                                 embeddings: List[np.ndarray],
                                 all_candidates: List[RetrievalCandidate]) -> float:
        """Calculate diversity score for a candidate."""
        
        candidate_idx = None
        for i, c in enumerate(all_candidates):
            if c.id == candidate.id:
                candidate_idx = i
                break
        
        if candidate_idx is None:
            return 0.0
        
        # Calculate average similarity to other selected candidates
        similarities = []
        for other in selected_candidates:
            if other.id != candidate.id:
                other_idx = None
                for i, c in enumerate(all_candidates):
                    if c.id == other.id:
                        other_idx = i
                        break
                
                if other_idx is not None:
                    similarity = np.dot(embeddings[candidate_idx], embeddings[other_idx])
                    similarities.append(similarity)
        
        if similarities:
            # Diversity score is inverse of average similarity
            avg_similarity = sum(similarities) / len(similarities)
            diversity_score = 1.0 - avg_similarity
        else:
            diversity_score = 1.0  # Maximum diversity if no other candidates
        
        return max(0.0, min(1.0, diversity_score))
    
    async def _stage_4_final_scoring(self, 
                                   context: RetrievalContext,
                                   candidates: List[RetrievalCandidate]) -> List[RetrievalCandidate]:
        """Stage 4: Final scoring and intelligent ordering."""
        
        for candidate in candidates:
            # Combine all scores with weights
            relevance_weight = 0.5
            rerank_weight = 0.3
            diversity_weight = 0.2
            
            # Adjust weights based on query type
            if context.query_type == "analytical":
                diversity_weight = 0.3  # More emphasis on diverse perspectives
                relevance_weight = 0.4
                rerank_weight = 0.3
            elif context.query_type == "factual":
                relevance_weight = 0.6  # More emphasis on relevance
                rerank_weight = 0.3
                diversity_weight = 0.1
            
            # Calculate final score
            relevance_score = candidate.initial_score or 0.0
            rerank_score = candidate.rerank_score or 0.0
            diversity_score = candidate.diversity_score or 0.5
            
            final_score = (
                relevance_weight * relevance_score +
                rerank_weight * rerank_score +
                diversity_weight * diversity_score
            )
            
            # Apply query-specific boosting
            final_score = self._apply_query_specific_boosting(
                candidate, context, final_score
            )
            
            candidate.final_score = final_score
            candidate.retrieval_stage = "final"
        
        # Sort by final score and return top results
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        final_results = candidates[:self.config["final_results_size"]]
        
        # Intelligent reordering based on context flow
        final_results = self._intelligent_context_ordering(final_results, context)
        
        return final_results
    
    def _apply_query_specific_boosting(self, 
                                     candidate: RetrievalCandidate,
                                     context: RetrievalContext,
                                     base_score: float) -> float:
        """Apply query-specific boosting to the final score."""
        
        boosted_score = base_score
        
        # Title relevance boost
        if any(entity.lower() in candidate.title.lower() for entity in context.entities):
            boosted_score *= 1.2
        
        # Multi-query match boost
        if candidate.source_queries and len(candidate.source_queries) > 1:
            boosted_score *= 1.15
        
        # Content length penalty for very short or very long content
        content_length = len(candidate.content.split())
        if content_length < 50:
            boosted_score *= 0.9  # Penalize very short content
        elif content_length > 2000:
            boosted_score *= 0.95  # Slight penalty for very long content
        
        # Recency boost for time-sensitive queries
        if any(word in context.original_query.lower() for word in ["recent", "latest", "new", "current"]):
            if "created_at" in candidate.metadata:
                try:
                    created_date = datetime.fromisoformat(candidate.metadata["created_at"])
                    days_old = (datetime.now() - created_date).days
                    if days_old <= 30:  # Within a month
                        boosted_score *= 1.3
                    elif days_old <= 90:  # Within 3 months
                        boosted_score *= 1.1
                except (ValueError, TypeError):
                    pass
        
        # Category/domain boost
        if context.topics:
            candidate_category = candidate.metadata.get("category", "").lower()
            if any(topic.lower() in candidate_category for topic in context.topics):
                boosted_score *= 1.1
        
        return boosted_score
    
    def _intelligent_context_ordering(self, 
                                    candidates: List[RetrievalCandidate],
                                    context: RetrievalContext) -> List[RetrievalCandidate]:
        """Intelligently reorder candidates for better context flow."""
        
        if len(candidates) <= 2:
            return candidates
        
        # For analytical queries, ensure diverse perspectives are well-distributed
        if context.query_type == "analytical" and len(candidates) >= 3:
            # Keep highest scoring at top, but ensure diversity in positions 2-4
            top_candidate = candidates[0]
            remaining = candidates[1:]
            
            # Sort remaining by diversity score
            remaining.sort(key=lambda x: x.diversity_score or 0.0, reverse=True)
            
            reordered = [top_candidate] + remaining
            return reordered
        
        # For procedural queries, try to order by logical flow
        if context.query_type == "procedural":
            # Look for step indicators in content
            step_candidates = []
            non_step_candidates = []
            
            for candidate in candidates:
                content_lower = candidate.content.lower()
                if any(word in content_lower for word in ["step", "first", "then", "next", "finally"]):
                    step_candidates.append(candidate)
                else:
                    non_step_candidates.append(candidate)
            
            # Order step candidates by likely sequence, others by score
            step_candidates.sort(key=lambda x: self._extract_step_order(x.content))
            non_step_candidates.sort(key=lambda x: x.final_score, reverse=True)
            
            return step_candidates + non_step_candidates
        
        # Default: return as-is (already sorted by final score)
        return candidates
    
    def _extract_step_order(self, content: str) -> int:
        """Extract likely step order from content."""
        content_lower = content.lower()
        
        # Look for explicit step numbers
        import re
        step_match = re.search(r'step\s*(\d+)', content_lower)
        if step_match:
            return int(step_match.group(1))
        
        # Look for sequence words
        sequence_words = {
            "first": 1, "initially": 1, "begin": 1, "start": 1,
            "second": 2, "then": 3, "next": 4, "after": 5,
            "finally": 10, "lastly": 10, "end": 10, "conclude": 10
        }
        
        for word, order in sequence_words.items():
            if word in content_lower:
                return order
        
        return 5  # Default middle order
    
    def _update_performance_stats(self, 
                                context: RetrievalContext,
                                processing_time: float,
                                result_count: int):
        """Update performance statistics."""
        
        self.retrieval_stats["total_queries"] += 1
        
        # Update average processing time
        total_time = self.retrieval_stats["avg_retrieval_time"] * (self.retrieval_stats["total_queries"] - 1)
        self.retrieval_stats["avg_retrieval_time"] = (total_time + processing_time) / self.retrieval_stats["total_queries"]
        
        # Update query type performance
        query_type = context.query_type
        if query_type not in self.retrieval_stats["query_type_performance"]:
            self.retrieval_stats["query_type_performance"][query_type] = {
                "count": 0, "avg_time": 0.0, "avg_results": 0.0
            }
        
        type_stats = self.retrieval_stats["query_type_performance"][query_type]
        type_stats["count"] += 1
        
        # Update average time for this query type
        total_type_time = type_stats["avg_time"] * (type_stats["count"] - 1)
        type_stats["avg_time"] = (total_type_time + processing_time) / type_stats["count"]
        
        # Update average results for this query type
        total_type_results = type_stats["avg_results"] * (type_stats["count"] - 1)
        type_stats["avg_results"] = (total_type_results + result_count) / type_stats["count"]
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get retrieval pipeline performance statistics."""
        return {
            "pipeline_config": self.config,
            "performance_stats": self.retrieval_stats,
            "models_used": {
                "sentence_transformer": self.sentence_transformer.model_name,
                "cross_encoder": self.cross_encoder.model_name,
                "openai_embeddings": "text-embedding-3-large" if self.openai_embeddings else None
            }
        }
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """Update pipeline configuration."""
        self.config.update(new_config)
        self.logger.info(f"Updated pipeline configuration: {new_config}")
    
    async def benchmark_pipeline(self, 
                               test_queries: List[RetrievalContext],
                               index_name: str = "default") -> Dict[str, Any]:
        """Benchmark the retrieval pipeline with test queries."""
        
        results = {
            "total_queries": len(test_queries),
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "average_processing_time": 0.0,
            "stage_performance": {},
            "query_type_breakdown": {}
        }
        
        processing_times = []
        
        for context in test_queries:
            try:
                start_time = datetime.now()
                retrieved_candidates = await self.retrieve(context, index_name)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                if retrieved_candidates:
                    results["successful_retrievals"] += 1
                    processing_times.append(processing_time)
                    
                    # Track by query type
                    query_type = context.query_type
                    if query_type not in results["query_type_breakdown"]:
                        results["query_type_breakdown"][query_type] = {
                            "count": 0, "success_rate": 0.0, "avg_time": 0.0
                        }
                    
                    type_stats = results["query_type_breakdown"][query_type]
                    type_stats["count"] += 1
                else:
                    results["failed_retrievals"] += 1
                    
            except Exception as e:
                self.logger.error(f"Benchmark failed for query: {context.original_query[:50]}... Error: {e}")
                results["failed_retrievals"] += 1
        
        # Calculate overall statistics
        if processing_times:
            results["average_processing_time"] = sum(processing_times) / len(processing_times)
        
        # Calculate success rates by query type
        for query_type, stats in results["query_type_breakdown"].items():
            if stats["count"] > 0:
                stats["success_rate"] = stats["count"] / len([q for q in test_queries if q.query_type == query_type])
        
        return results
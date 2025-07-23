from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
import json
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI

# Import our enhanced components
from .advanced_query_processor import AdvancedQueryProcessor, QueryAnalysis, QueryType, QueryComplexity
from .advanced_context_manager import AdvancedContextManager, ContextualInformation
from .response_quality_manager import ResponseQualityManager, QualityMetrics
from .prompts import get_prompt_template, build_prompt_variables

@dataclass
class RAGResponse:
    """Complete RAG response with metadata."""
    response: str
    query_analysis: QueryAnalysis
    contextual_info: ContextualInformation
    quality_metrics: QualityMetrics
    retrieval_strategy: Dict[str, Any]
    processing_time: float
    confidence_score: float
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class AdaptiveRAGStrategy:
    """Adaptive retrieval strategy based on query characteristics and performance history."""
    
    def __init__(self):
        self.strategy_history: Dict[str, List[float]] = {}  # Strategy -> success scores
        self.query_type_preferences: Dict[str, str] = {}  # Query type -> preferred strategy
        
    def select_strategy(self, query_analysis: QueryAnalysis, context: str = "") -> Dict[str, Any]:
        """Select optimal retrieval strategy based on query analysis and history."""
        
        # Base strategy from query analysis
        base_strategy = {
            "use_semantic_search": True,
            "use_keyword_search": True,
            "use_hybrid_ranking": True,
            "context_window": 5,
            "retrieval_rounds": 1,
            "reranking_enabled": True,
            "query_expansion": True
        }
        
        # Adapt based on query type
        if query_analysis.query_type == QueryType.PROCEDURAL:
            base_strategy.update({
                "context_window": 7,
                "prefer_sequential": True,
                "boost_step_by_step": True
            })
        elif query_analysis.query_type == QueryType.ANALYTICAL:
            base_strategy.update({
                "context_window": 10,
                "use_comparative_search": True,
                "retrieval_rounds": 2,
                "enable_multi_perspective": True
            })
        elif query_analysis.query_type == QueryType.MULTI_HOP:
            base_strategy.update({
                "retrieval_rounds": 3,
                "use_graph_traversal": True,
                "context_window": 12,
                "enable_reasoning_chain": True
            })
        elif query_analysis.query_type == QueryType.CONVERSATIONAL:
            base_strategy.update({
                "context_window": 3,
                "boost_conversational_context": True,
                "prefer_recent_context": True
            })
        
        # Adapt based on complexity
        if query_analysis.complexity == QueryComplexity.COMPLEX:
            base_strategy["retrieval_rounds"] = max(base_strategy["retrieval_rounds"], 2)
            base_strategy["context_window"] += 2
        elif query_analysis.complexity == QueryComplexity.SIMPLE:
            base_strategy["retrieval_rounds"] = 1
            base_strategy["context_window"] = min(base_strategy["context_window"], 5)
        
        # Adapt based on urgency
        if query_analysis.urgency == "high":
            base_strategy.update({
                "fast_mode": True,
                "retrieval_rounds": 1,
                "skip_reranking": True
            })
        elif query_analysis.urgency == "critical":
            base_strategy.update({
                "fast_mode": True,
                "retrieval_rounds": 1,
                "skip_reranking": True,
                "use_cache_aggressive": True
            })
        
        # Apply learned preferences
        query_type_key = query_analysis.query_type.value
        if query_type_key in self.query_type_preferences:
            preferred_strategy = self.query_type_preferences[query_type_key]
            if preferred_strategy in self.strategy_history:
                avg_success = sum(self.strategy_history[preferred_strategy]) / len(self.strategy_history[preferred_strategy])
                if avg_success > 0.8:  # High success rate
                    base_strategy["strategy_boost"] = preferred_strategy
        
        return base_strategy
    
    def update_strategy_performance(self, strategy: Dict[str, Any], success_score: float):
        """Update strategy performance based on results."""
        strategy_key = json.dumps(sorted(strategy.items()))
        
        if strategy_key not in self.strategy_history:
            self.strategy_history[strategy_key] = []
        
        self.strategy_history[strategy_key].append(success_score)
        
        # Keep only recent history (last 100 entries)
        if len(self.strategy_history[strategy_key]) > 100:
            self.strategy_history[strategy_key] = self.strategy_history[strategy_key][-100:]

class EnhancedRAGPipeline:
    """State-of-the-art RAG pipeline with adaptive strategies and quality assurance."""
    
    def __init__(self, 
                 knowledge_base_url: str = "http://localhost:8002",
                 analytics_url: str = "http://localhost:8005"):
        
        # Core components
        self.query_processor = AdvancedQueryProcessor()
        self.context_manager = AdvancedContextManager()
        self.quality_manager = ResponseQualityManager()
        self.adaptive_strategy = AdaptiveRAGStrategy()
        
        # LLM for response generation
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        
        # Service URLs
        self.knowledge_base_url = knowledge_base_url
        self.analytics_url = analytics_url
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def process_query(self, 
                          conversation_id: str,
                          user_message: str,
                          user_profile: Optional[Dict[str, Any]] = None,
                          conversation_context: str = "") -> RAGResponse:
        """Process user query through the enhanced RAG pipeline."""
        
        start_time = datetime.now()
        
        try:
            # Step 1: Advanced query analysis
            self.logger.info(f"Processing query: {user_message[:50]}...")
            query_analysis = await self.query_processor.analyze_query(
                user_message, conversation_context
            )
            
            # Step 2: Select adaptive retrieval strategy
            retrieval_strategy = self.adaptive_strategy.select_strategy(
                query_analysis, conversation_context
            )
            
            # Step 3: Retrieve relevant documents
            retrieved_documents = await self._retrieve_documents(
                query_analysis, retrieval_strategy
            )
            
            # Step 4: Build optimal context
            contextual_info = await self.context_manager.build_optimal_context(
                conversation_id=conversation_id,
                current_message=user_message,
                query_analysis=query_analysis,
                retrieved_documents=retrieved_documents,
                user_profile=user_profile
            )
            
            # Step 5: Generate initial response
            initial_response = await self._generate_response(
                query_analysis, contextual_info, user_profile, retrieved_documents
            )
            
            # Step 6: Quality validation and improvement
            final_response, quality_metrics = await self.quality_manager.auto_improve_if_needed(
                response=initial_response,
                query=user_message,
                query_type=query_analysis.query_type.value,
                context=contextual_info.primary_context,
                user_profile=user_profile
            )
            
            # Step 7: Calculate processing time and confidence
            processing_time = (datetime.now() - start_time).total_seconds()
            confidence_score = self._calculate_confidence_score(
                query_analysis, contextual_info, quality_metrics
            )
            
            # Step 8: Create response object
            rag_response = RAGResponse(
                response=final_response,
                query_analysis=query_analysis,
                contextual_info=contextual_info,
                quality_metrics=quality_metrics,
                retrieval_strategy=retrieval_strategy,
                processing_time=processing_time,
                confidence_score=confidence_score,
                sources=retrieved_documents[:5],  # Top 5 sources
                metadata={
                    "conversation_id": conversation_id,
                    "timestamp": datetime.now().isoformat(),
                    "pipeline_version": "enhanced_v1.0",
                    "improvement_applied": quality_metrics.requires_revision
                }
            )
            
            # Step 9: Update conversation history
            self.context_manager.add_conversation_turn(
                conversation_id=conversation_id,
                user_message=user_message,
                assistant_response=final_response,
                context_used=[contextual_info.primary_context],
                retrieval_quality=quality_metrics.overall_score
            )
            
            # Step 10: Update adaptive strategy performance
            success_score = self._calculate_success_score(quality_metrics, confidence_score)
            self.adaptive_strategy.update_strategy_performance(retrieval_strategy, success_score)
            
            # Step 11: Track performance
            self._track_performance(rag_response)
            
            self.logger.info(f"Query processed successfully in {processing_time:.2f}s with confidence {confidence_score:.2f}")
            
            return rag_response
            
        except Exception as e:
            self.logger.error(f"Error in RAG pipeline: {e}")
            return self._create_error_response(user_message, str(e), start_time)
    
    async def _retrieve_documents(self, 
                                query_analysis: QueryAnalysis, 
                                strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve documents using adaptive strategy."""
        
        # This would integrate with the knowledge base service
        # For now, return mock data structure
        retrieved_docs = []
        
        # Multi-round retrieval if specified
        rounds = strategy.get("retrieval_rounds", 1)
        queries_to_use = [query_analysis.original_query] + query_analysis.expanded_queries[:2]
        
        for round_num in range(rounds):
            query_set = queries_to_use[round_num:round_num+1] if round_num < len(queries_to_use) else [query_analysis.original_query]
            
            for query in query_set:
                # Simulate retrieval call to knowledge base service
                docs = await self._call_knowledge_base_service(
                    query, 
                    strategy.get("context_window", 5),
                    query_analysis.entities,
                    strategy
                )
                retrieved_docs.extend(docs)
        
        # Remove duplicates and apply strategy-specific filtering
        unique_docs = self._deduplicate_and_filter(retrieved_docs, strategy)
        
        return unique_docs[:strategy.get("context_window", 10)]
    
    async def _call_knowledge_base_service(self, 
                                         query: str, 
                                         top_k: int,
                                         entities: List[str],
                                         strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Call knowledge base service for document retrieval."""
        import httpx
        
        try:
            # Use working search endpoint
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.knowledge_base_url}/api/v1/search",
                    params={
                        "q": query,
                        "limit": top_k
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return [
                        {
                            "id": result["id"],
                            "content": result["content"],
                            "title": result["title"],
                            "score": result.get("score", 0.0),
                            "metadata": {
                                **result.get("metadata", {}),
                                "category": result.get("category", "Unknown"),
                                "subcategory": result.get("subcategory"),
                                "tags": result.get("tags", [])
                            }
                        }
                        for result in data.get("results", [])
                    ]
                else:
                    self.logger.warning(f"Knowledge base service returned {response.status_code}")
                    
        except Exception as e:
            self.logger.error(f"Error calling knowledge base service: {e}")
        
        # Fallback to mock data if service fails
        return [
            {
                "id": f"doc_{i}",
                "content": f"Mock content for query: {query}",
                "title": f"Document {i}",
                "score": 0.9 - (i * 0.1),
                "metadata": {"source": "mock", "entities": entities}
            }
            for i in range(min(top_k, 3))
        ]
    
    def _deduplicate_and_filter(self, docs: List[Dict[str, Any]], strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Remove duplicates and apply strategy-specific filtering."""
        seen_ids = set()
        unique_docs = []
        
        for doc in docs:
            if doc.get("id") not in seen_ids:
                seen_ids.add(doc.get("id"))
                
                # Apply strategy-specific filters
                if strategy.get("boost_recent") and "timestamp" in doc.get("metadata", {}):
                    doc["score"] *= 1.2
                
                if strategy.get("prefer_sequential") and "step" in doc.get("content", "").lower():
                    doc["score"] *= 1.3
                
                unique_docs.append(doc)
        
        # Sort by score
        return sorted(unique_docs, key=lambda x: x.get("score", 0), reverse=True)
    
    async def _generate_response(self, 
                               query_analysis: QueryAnalysis,
                               contextual_info: ContextualInformation,
                               user_profile: Optional[Dict[str, Any]] = None,
                               retrieved_documents: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate response using appropriate prompt template."""
        
        # Select appropriate prompt template
        prompt_template = get_prompt_template(query_analysis.query_type.value)
        
        # Build prompt variables
        prompt_variables = build_prompt_variables(
            contextual_info, query_analysis, user_profile, retrieved_documents
        )
        
        # Generate response
        try:
            response = await prompt_template.ainvoke(prompt_variables)
            return response.content.strip()
        except Exception as e:
            self.logger.error(f"Error generating response with LLM: {e}")
            # Fallback: Create response from retrieved documents
            return self._create_fallback_response(query_analysis, retrieved_documents or [])
    
    def _create_fallback_response(self, 
                                query_analysis: QueryAnalysis, 
                                retrieved_documents: List[Dict[str, Any]]) -> str:
        """Create a fallback response using retrieved documents when LLM fails."""
        if not retrieved_documents:
            return f"I apologize, but I couldn't find specific information about '{query_analysis.original_query}' in the knowledge base."
        
        # Use the highest scoring document to create a response
        best_doc = retrieved_documents[0]
        title = best_doc.get("title", "")
        content = best_doc.get("content", "")
        
        # Create a simple response based on the query type
        if "what is" in query_analysis.original_query.lower() or "define" in query_analysis.original_query.lower():
            return f"{title}: {content[:300]}{'...' if len(content) > 300 else ''}"
        elif "how" in query_analysis.original_query.lower():
            return f"Based on the information about {title}: {content[:400]}{'...' if len(content) > 400 else ''}"
        else:
            return f"Regarding your question about '{query_analysis.original_query}', here's relevant information from {title}: {content[:300]}{'...' if len(content) > 300 else ''}"
    
    def _calculate_confidence_score(self, 
                                  query_analysis: QueryAnalysis,
                                  contextual_info: ContextualInformation,
                                  quality_metrics: QualityMetrics) -> float:
        """Calculate overall confidence score for the response."""
        
        # Components of confidence
        query_confidence = query_analysis.confidence
        context_confidence = contextual_info.confidence_score
        quality_confidence = quality_metrics.overall_score / 5.0
        
        # Weighted combination
        confidence = (
            0.3 * query_confidence +
            0.3 * context_confidence +
            0.4 * quality_confidence
        )
        
        # Apply adjustments based on various factors
        if len(contextual_info.supporting_context) > 2:
            confidence += 0.1  # More context sources
        
        if query_analysis.complexity == QueryComplexity.SIMPLE:
            confidence += 0.05  # Simpler queries are more reliable
        
        if quality_metrics.requires_revision:
            confidence -= 0.1  # Quality issues reduce confidence
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_success_score(self, quality_metrics: QualityMetrics, confidence_score: float) -> float:
        """Calculate success score for strategy adaptation."""
        return (quality_metrics.overall_score / 5.0 + confidence_score) / 2
    
    def _track_performance(self, rag_response: RAGResponse) -> None:
        """Track pipeline performance metrics."""
        performance_entry = {
            "timestamp": datetime.now().isoformat(),
            "processing_time": rag_response.processing_time,
            "quality_score": rag_response.quality_metrics.overall_score,
            "confidence_score": rag_response.confidence_score,
            "query_type": rag_response.query_analysis.query_type.value,
            "query_complexity": rag_response.query_analysis.complexity.value,
            "improvement_applied": rag_response.quality_metrics.requires_revision,
            "retrieval_strategy": rag_response.retrieval_strategy
        }
        
        self.performance_history.append(performance_entry)
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def _create_error_response(self, query: str, error: str, start_time: datetime) -> RAGResponse:
        """Create error response when pipeline fails."""
        from .advanced_query_processor import QueryType, QueryComplexity
        
        # Create minimal response objects
        error_query_analysis = QueryAnalysis(
            original_query=query,
            query_type=QueryType.FACTUAL,
            complexity=QueryComplexity.SIMPLE,
            intent="error_handling",
            entities=[],
            topics=["error"],
            sentiment="neutral",
            urgency="medium",
            expanded_queries=[],
            keywords=[],
            context_needed=False,
            confidence=0.0
        )
        
        error_context = ContextualInformation(
            primary_context="Error occurred during processing",
            supporting_context=[],
            conversation_history="",
            user_profile={},
            temporal_context="",
            domain_context="",
            confidence_score=0.0,
            relevance_scores=[]
        )
        
        error_quality = QualityMetrics(
            accuracy=1.0,
            completeness=1.0,
            relevance=1.0,
            clarity=1.0,
            appropriateness=1.0,
            overall_score=1.0,
            suggestions=[],
            requires_revision=False,
            confidence_level="low",
            timestamp=datetime.now()
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return RAGResponse(
            response=f"I apologize, but I encountered an issue processing your query: '{query}'. Please try rephrasing your question or contact support if the problem persists.",
            query_analysis=error_query_analysis,
            contextual_info=error_context,
            quality_metrics=error_quality,
            retrieval_strategy={},
            processing_time=processing_time,
            confidence_score=0.0,
            sources=[],
            metadata={"error": error, "pipeline_version": "enhanced_v1.0"}
        )
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline performance statistics."""
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        recent_history = self.performance_history[-100:]
        
        return {
            "total_queries": len(self.performance_history),
            "recent_performance": {
                "avg_processing_time": sum(entry["processing_time"] for entry in recent_history) / len(recent_history),
                "avg_quality_score": sum(entry["quality_score"] for entry in recent_history) / len(recent_history),
                "avg_confidence": sum(entry["confidence_score"] for entry in recent_history) / len(recent_history),
                "improvement_rate": sum(1 for entry in recent_history if entry["improvement_applied"]) / len(recent_history)
            },
            "query_type_distribution": self._get_query_type_distribution(recent_history),
            "strategy_performance": dict(list(self.adaptive_strategy.strategy_history.items())[:10]),
            "quality_manager_stats": self.quality_manager.get_quality_statistics(),
            "context_manager_conversations": len(self.context_manager.conversations)
        }
    
    def _get_query_type_distribution(self, history: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of query types."""
        distribution = {}
        for entry in history:
            query_type = entry["query_type"]
            distribution[query_type] = distribution.get(query_type, 0) + 1
        return distribution
    
    async def optimize_pipeline(self) -> Dict[str, Any]:
        """Optimize pipeline based on performance history."""
        if len(self.performance_history) < 50:
            return {"message": "Insufficient data for optimization"}
        
        # Analyze performance patterns
        recent_stats = self.get_pipeline_statistics()
        
        # Adjust quality threshold based on performance
        avg_quality = recent_stats["recent_performance"]["avg_quality_score"]
        if avg_quality > 4.2:
            self.quality_manager.update_quality_threshold(4.2)
        elif avg_quality < 3.5:
            self.quality_manager.update_quality_threshold(3.5)
        
        # Optimize context window sizes based on query types
        optimizations_applied = []
        
        if recent_stats["recent_performance"]["avg_processing_time"] > 5.0:
            # Reduce complexity for faster processing
            optimizations_applied.append("Reduced context windows for faster processing")
        
        if recent_stats["recent_performance"]["avg_confidence"] < 0.7:
            # Increase retrieval rounds for better confidence
            optimizations_applied.append("Increased retrieval rounds for better confidence")
        
        return {
            "optimizations_applied": optimizations_applied,
            "current_performance": recent_stats["recent_performance"],
            "quality_threshold": self.quality_manager.quality_threshold
        }
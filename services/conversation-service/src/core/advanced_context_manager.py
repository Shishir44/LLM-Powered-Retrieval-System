"""
Advanced Context Manager for RAG Pipeline

This module provides sophisticated context management capabilities including:
- User profile tracking
- Conversation history management
- Context optimization for different query types
"""

from typing import Dict, List, Any, Optional, Deque
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .advanced_query_processor import QueryAnalysis, QueryComplexity


@dataclass
class ContextualInformation:
    """Comprehensive contextual information for RAG responses."""
    primary_context: str
    supporting_context: List[str]
    conversation_history: str
    user_profile: Dict[str, Any]
    temporal_context: str
    domain_context: str
    confidence_score: float
    relevance_scores: List[float]


@dataclass
class ConversationTurn:
    """Represents a single conversation turn."""
    user_message: str
    assistant_response: str
    timestamp: datetime
    context_used: List[str]
    retrieval_quality: float
    user_feedback: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedContextManager:
    """Advanced context management with user profiling and optimization."""
    
    def __init__(self, max_conversation_length: int = 20, context_window_tokens: int = 4000):
        self.conversations: Dict[str, Deque[ConversationTurn]] = {}
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.context_cache: Dict[str, ContextualInformation] = {}
        self.max_conversation_length = max_conversation_length
        self.context_window_tokens = context_window_tokens
        
        # LLM for context optimization
        self.context_optimizer = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        # Context selection strategies
        self.context_selector = self._create_context_selector()
        self.conversation_summarizer = self._create_conversation_summarizer()
        self.relevance_scorer = self._create_relevance_scorer()
    
    def _create_context_selector(self):
        return ChatPromptTemplate.from_messages([
            ("system", """You are a context selection expert. Given retrieved documents and a query analysis, 
select and organize the most relevant context for answering the user's question.

Return a JSON object:
{
    "primary_context": "most relevant context",
    "supporting_context": ["additional context 1", "additional context 2"],
    "relevance_scores": [0.95, 0.87, 0.76],
    "reasoning": "explanation of context selection",
    "confidence": 0.9
}"""),
            ("human", """Query Analysis: {query_analysis}
            
Retrieved Documents:
{retrieved_documents}

Conversation Context:
{conversation_context}

Select optimal context:""")
        ])
    
    def _create_conversation_summarizer(self):
        return ChatPromptTemplate.from_messages([
            ("system", "Summarize the conversation history to provide relevant context for the current query."),
            ("human", "Current Query: {current_query}\n\nConversation History:\n{conversation_history}\n\nProvide summary:")
        ])
    
    def _create_relevance_scorer(self):
        return ChatPromptTemplate.from_messages([
            ("system", "Score the relevance of each context piece to the user's query on a scale of 0.0 to 1.0."),
            ("human", "Query: {query}\nContext Pieces:\n{context_pieces}\n\nScore relevance:")
        ])
    
    async def build_optimal_context(self, 
                                  conversation_id: str,
                                  current_message: str,
                                  query_analysis: QueryAnalysis,
                                  retrieved_documents: List[Dict[str, Any]],
                                  user_profile: Optional[Dict[str, Any]] = None) -> ContextualInformation:
        """Build optimal context using advanced selection strategies."""
        
        # Get conversation history
        conversation_history = self._get_conversation_summary(conversation_id, current_message)
        
        # Update user profile
        if user_profile:
            self.user_profiles[conversation_id] = user_profile
        elif conversation_id not in self.user_profiles:
            self.user_profiles[conversation_id] = self._infer_user_profile(conversation_id, query_analysis)
        
        # Select optimal context based on query analysis
        context_selection = await self._select_context(
            query_analysis=query_analysis,
            retrieved_documents=retrieved_documents,
            conversation_history=conversation_history
        )
        
        # Build temporal context if needed
        temporal_context = self._build_temporal_context(query_analysis)
        
        # Build domain context
        domain_context = self._build_domain_context(query_analysis, retrieved_documents)
        
        # Create final contextual information
        contextual_info = ContextualInformation(
            primary_context=context_selection["primary_context"],
            supporting_context=context_selection["supporting_context"],
            conversation_history=conversation_history,
            user_profile=self.user_profiles[conversation_id],
            temporal_context=temporal_context,
            domain_context=domain_context,
            confidence_score=context_selection["confidence"],
            relevance_scores=context_selection["relevance_scores"]
        )
        
        # Cache the context for potential reuse
        cache_key = f"{conversation_id}_{hash(current_message)}"
        self.context_cache[cache_key] = contextual_info
        
        return contextual_info
    
    async def _select_context(self, 
                            query_analysis: QueryAnalysis,
                            retrieved_documents: List[Dict[str, Any]],
                            conversation_history: str) -> Dict[str, Any]:
        """Select optimal context using LLM-based selection."""
        try:
            # Prepare documents for selection
            doc_summaries = []
            for i, doc in enumerate(retrieved_documents[:10]):  # Limit to top 10
                summary = {
                    "id": i,
                    "title": doc.get("title", "Untitled"),
                    "content": doc["content"][:500],  # Limit content length
                    "relevance_score": doc.get("score", 0.0),
                    "metadata": doc.get("metadata", {})
                }
                doc_summaries.append(summary)
            
            response = await self.context_selector.ainvoke({
                "query_analysis": json.dumps({
                    "query": query_analysis.original_query,
                    "type": query_analysis.query_type.value,
                    "complexity": query_analysis.complexity.value,
                    "intent": query_analysis.intent,
                    "keywords": query_analysis.keywords
                }),
                "retrieved_documents": json.dumps(doc_summaries, indent=2),
                "conversation_context": conversation_history[:1000]  # Limit context length
            })
            
            return json.loads(response.content)
            
        except Exception as e:
            # Fallback to rule-based selection
            return self._fallback_context_selection(retrieved_documents, query_analysis)
    
    def _fallback_context_selection(self, 
                                   retrieved_documents: List[Dict[str, Any]],
                                   query_analysis: QueryAnalysis) -> Dict[str, Any]:
        """Fallback context selection using rules."""
        if not retrieved_documents:
            return {
                "primary_context": "No relevant information found.",
                "supporting_context": [],
                "relevance_scores": [],
                "confidence": 0.0
            }
        
        # Simple selection based on scores
        sorted_docs = sorted(retrieved_documents, key=lambda x: x.get("score", 0), reverse=True)
        
        primary_context = sorted_docs[0]["content"][:1000]
        supporting_context = [doc["content"][:500] for doc in sorted_docs[1:3]]
        relevance_scores = [doc.get("score", 0.5) for doc in sorted_docs[:3]]
        
        return {
            "primary_context": primary_context,
            "supporting_context": supporting_context,
            "relevance_scores": relevance_scores,
            "confidence": 0.7
        }
    
    def _get_conversation_summary(self, conversation_id: str, current_message: str) -> str:
        """Get summarized conversation history."""
        if conversation_id not in self.conversations:
            return "No previous conversation history."
        
        history = list(self.conversations[conversation_id])
        if not history:
            return "No previous conversation history."
        
        # Create simple summary of recent turns
        recent_turns = history[-5:]  # Last 5 turns
        summary_parts = []
        
        for turn in recent_turns:
            summary_parts.append(f"User: {turn.user_message[:100]}...")
            summary_parts.append(f"Assistant: {turn.assistant_response[:100]}...")
        
        return "\n".join(summary_parts)
    
    def _infer_user_profile(self, conversation_id: str, query_analysis: QueryAnalysis) -> Dict[str, Any]:
        """Infer user profile from conversation and query analysis."""
        profile = {
            "expertise_level": "intermediate",
            "preferred_response_style": "detailed",
            "interests": query_analysis.topics,
            "conversation_start": datetime.now(),
            "query_patterns": [query_analysis.query_type.value],
            "sentiment_history": [query_analysis.sentiment],
            "urgency_patterns": [query_analysis.urgency]
        }
        
        # Analyze complexity patterns
        if query_analysis.complexity in [QueryComplexity.COMPLEX, QueryComplexity.MULTI_STEP]:
            profile["expertise_level"] = "advanced"
        elif query_analysis.complexity == QueryComplexity.SIMPLE:
            profile["expertise_level"] = "beginner"
        
        return profile
    
    def _build_temporal_context(self, query_analysis: QueryAnalysis) -> str:
        """Build temporal context if relevant."""
        now = datetime.now()
        temporal_indicators = ["recent", "latest", "current", "today", "now", "new"]
        
        if any(indicator in query_analysis.original_query.lower() for indicator in temporal_indicators):
            return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        
        return ""
    
    def _build_domain_context(self, 
                            query_analysis: QueryAnalysis,
                            retrieved_documents: List[Dict[str, Any]]) -> str:
        """Build domain-specific context."""
        # Extract common domains from documents
        domains = set()
        for doc in retrieved_documents[:5]:
            metadata = doc.get("metadata", {})
            if "domain" in metadata:
                domains.add(metadata["domain"])
            if "category" in metadata:
                domains.add(metadata["category"])
        
        if domains:
            return f"Domain context: {', '.join(domains)}"
        
        return ""
    
    def add_conversation_turn(self, 
                            conversation_id: str,
                            user_message: str,
                            assistant_response: str,
                            context_used: List[str],
                            retrieval_quality: float) -> None:
        """Add a conversation turn to history."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = deque(maxlen=self.max_conversation_length)
        
        turn = ConversationTurn(
            user_message=user_message,
            assistant_response=assistant_response,
            timestamp=datetime.now(),
            context_used=context_used,
            retrieval_quality=retrieval_quality
        )
        
        self.conversations[conversation_id].append(turn)
        
        # Update user profile based on interaction
        self._update_user_profile(conversation_id, turn)
    
    def _update_user_profile(self, conversation_id: str, turn: ConversationTurn) -> None:
        """Update user profile based on conversation turn."""
        if conversation_id not in self.user_profiles:
            return
        
        profile = self.user_profiles[conversation_id]
        
        # Update interaction patterns
        if "total_interactions" not in profile:
            profile["total_interactions"] = 0
        profile["total_interactions"] += 1
        
        # Track average retrieval quality
        if "avg_retrieval_quality" not in profile:
            profile["avg_retrieval_quality"] = turn.retrieval_quality
        else:
            current_avg = profile["avg_retrieval_quality"]
            total = profile["total_interactions"]
            profile["avg_retrieval_quality"] = (current_avg * (total - 1) + turn.retrieval_quality) / total
        
        # Update last interaction
        profile["last_interaction"] = turn.timestamp
    
    def clear_conversation(self, conversation_id: str) -> None:
        """Clear conversation history for a given ID."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
        if conversation_id in self.user_profiles:
            del self.user_profiles[conversation_id]
    
    def get_active_conversations(self) -> List[str]:
        """Get list of active conversation IDs."""
        # Consider conversations active if they have activity in last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        active = []
        
        for conv_id, turns in self.conversations.items():
            if turns and turns[-1].timestamp > cutoff:
                active.append(conv_id)
        
        return active
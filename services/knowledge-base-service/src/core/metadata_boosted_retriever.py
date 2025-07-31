"""
PHASE 2.1: Metadata-Boosted Retriever
Enhanced semantic search with intelligent metadata boosting for improved accuracy
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import math

@dataclass
class DocumentMetadata:
    """Enhanced document metadata for boosting calculations."""
    doc_id: str
    title: str
    category: str
    subcategory: Optional[str] = None
    tags: List[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    authority_score: float = 1.0
    quality_score: float = 1.0
    access_count: int = 0
    user_rating: float = 0.0
    content_type: str = "general"  # faq, policy, procedure, troubleshooting
    importance_level: str = "normal"  # critical, high, normal, low

@dataclass 
class RetrievalResult:
    """Enhanced retrieval result with metadata boosting."""
    doc_id: str
    content: str
    title: str
    similarity_score: float
    metadata: DocumentMetadata
    # Boosting components
    recency_boost: float = 1.0
    authority_boost: float = 1.0
    relevance_boost: float = 1.0
    category_boost: float = 1.0
    final_score: float = 0.0
    explanation: str = ""

class MetadataBoostedRetriever:
    """PHASE 2.1: Advanced retrieval with intelligent metadata boosting."""
    
    def __init__(self, base_retriever, boost_config: Optional[Dict[str, Any]] = None):
        self.base_retriever = base_retriever
        self.logger = logging.getLogger(__name__)
        
        # PHASE 2.1: Boosting configuration - tuned for accuracy
        self.boost_config = boost_config or {
            "recency_weight": 0.15,      # Boost recent documents
            "authority_weight": 0.20,    # Boost authoritative sources
            "relevance_weight": 0.25,    # Boost highly relevant content
            "category_weight": 0.15,     # Boost matching categories
            "quality_weight": 0.15,      # Boost high-quality content
            "popularity_weight": 0.10,   # Boost frequently accessed docs
            
            # Decay parameters
            "recency_decay_days": 365,   # How quickly recency boost decays
            "min_boost_factor": 0.5,     # Minimum boost multiplier
            "max_boost_factor": 2.0,     # Maximum boost multiplier
        }
        
        # Category relevance mapping for query-category matching
        self.category_relevance = {
            "billing": ["payment", "charge", "invoice", "billing", "cost", "price", "subscription"],
            "technical": ["error", "bug", "issue", "problem", "troubleshoot", "debug", "fix"],
            "account": ["login", "register", "profile", "password", "access", "authentication"],
            "policy": ["policy", "terms", "condition", "agreement", "rule", "guideline"],
            "procedure": ["how to", "step", "guide", "tutorial", "instruction", "process"],
            "faq": ["question", "answer", "faq", "frequently", "common", "help"],
            "product": ["feature", "specification", "model", "version", "product", "service"]
        }
        
        self.logger.info("PHASE 2.1: Initialized Metadata-Boosted Retriever")

    async def retrieve_with_boosting(self, 
                                   query: str, 
                                   top_k: int = 10,
                                   category_hint: Optional[str] = None,
                                   content_type_hint: Optional[str] = None,
                                   filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """Enhanced retrieval with metadata boosting."""
        
        try:
            # Step 1: Get initial results from base retriever (get more for boosting)
            initial_results = await self.base_retriever.semantic_search(
                query=query,
                top_k=min(top_k * 3, 50),  # Get 3x more for better boosting selection
                filters=filters
            )
            
            if not initial_results:
                self.logger.warning(f"No initial results for query: {query}")
                return []
            
            # Step 2: Convert to enhanced results with metadata
            enhanced_results = []
            for result in initial_results:
                enhanced_result = self._convert_to_enhanced_result(result, query)
                enhanced_results.append(enhanced_result)
            
            # Step 3: Apply metadata boosting
            boosted_results = await self._apply_metadata_boosting(
                enhanced_results, query, category_hint, content_type_hint
            )
            
            # Step 4: Final ranking and selection
            final_results = self._final_ranking_with_explanation(boosted_results)
            
            # Step 5: Return top K results
            top_results = final_results[:top_k]
            
            self.logger.info(f"PHASE 2.1: Retrieved {len(top_results)} boosted results for query: {query[:50]}")
            
            return top_results
            
        except Exception as e:
            self.logger.error(f"Error in metadata-boosted retrieval: {e}")
            # Fallback to base retriever
            return await self.base_retriever.semantic_search(query, top_k, filters=filters)

    def _convert_to_enhanced_result(self, base_result, query: str) -> RetrievalResult:
        """Convert base retrieval result to enhanced result with metadata."""
        
        # Extract metadata from base result
        metadata_dict = base_result.metadata if hasattr(base_result, 'metadata') else {}
        
        # Create enhanced metadata
        metadata = DocumentMetadata(
            doc_id=base_result.document.id if hasattr(base_result, 'document') else base_result.get('id', 'unknown'),
            title=base_result.document.title if hasattr(base_result, 'document') else base_result.get('title', ''),
            category=metadata_dict.get('category', 'general'),
            subcategory=metadata_dict.get('subcategory'),
            tags=metadata_dict.get('tags', []),
            created_at=self._parse_timestamp(metadata_dict.get('created_at')),
            updated_at=self._parse_timestamp(metadata_dict.get('updated_at')),
            authority_score=metadata_dict.get('authority_score', 1.0),
            quality_score=metadata_dict.get('quality_score', 1.0),
            access_count=metadata_dict.get('access_count', 0),
            user_rating=metadata_dict.get('user_rating', 0.0),
            content_type=metadata_dict.get('content_type', 'general'),
            importance_level=metadata_dict.get('importance_level', 'normal')
        )
        
        return RetrievalResult(
            doc_id=metadata.doc_id,
            content=base_result.document.content if hasattr(base_result, 'document') else base_result.get('content', ''),
            title=metadata.title,
            similarity_score=base_result.semantic_score if hasattr(base_result, 'semantic_score') else base_result.get('score', 0.0),
            metadata=metadata
        )

    async def _apply_metadata_boosting(self, 
                                     results: List[RetrievalResult], 
                                     query: str,
                                     category_hint: Optional[str] = None,
                                     content_type_hint: Optional[str] = None) -> List[RetrievalResult]:
        """Apply intelligent metadata boosting to results."""
        
        current_time = datetime.now()
        query_lower = query.lower()
        
        for result in results:
            # Calculate individual boost factors
            result.recency_boost = self._calculate_recency_boost(result.metadata, current_time)
            result.authority_boost = self._calculate_authority_boost(result.metadata)
            result.relevance_boost = self._calculate_relevance_boost(result.metadata, query_lower)
            result.category_boost = self._calculate_category_boost(result.metadata, query_lower, category_hint)
            
            # Calculate final boosted score
            base_score = result.similarity_score
            
            # Apply weighted boosting
            boosted_score = base_score * (
                1.0 +
                (result.recency_boost - 1.0) * self.boost_config["recency_weight"] +
                (result.authority_boost - 1.0) * self.boost_config["authority_weight"] +
                (result.relevance_boost - 1.0) * self.boost_config["relevance_weight"] +
                (result.category_boost - 1.0) * self.boost_config["category_weight"]
            )
            
            # Apply quality and popularity boosts
            quality_factor = 1.0 + (result.metadata.quality_score - 1.0) * self.boost_config["quality_weight"]
            popularity_factor = 1.0 + min(result.metadata.access_count / 100.0, 0.5) * self.boost_config["popularity_weight"]
            
            result.final_score = boosted_score * quality_factor * popularity_factor
            
            # Ensure final score stays within reasonable bounds
            result.final_score = max(
                base_score * self.boost_config["min_boost_factor"],
                min(base_score * self.boost_config["max_boost_factor"], result.final_score)
            )
            
            # Generate explanation for transparency
            result.explanation = self._generate_boost_explanation(result, base_score)
        
        return results

    def _calculate_recency_boost(self, metadata: DocumentMetadata, current_time: datetime) -> float:
        """Calculate boost factor based on document recency."""
        
        if not metadata.updated_at and not metadata.created_at:
            return 1.0  # No timestamp info, no boost
        
        doc_time = metadata.updated_at or metadata.created_at
        days_old = (current_time - doc_time).days
        
        # Exponential decay: newer documents get higher boost
        if days_old <= 0:
            return 1.5  # Very recent
        elif days_old <= 30:
            return 1.3  # Recent (last month)
        elif days_old <= 90:
            return 1.1  # Moderately recent (last quarter)
        elif days_old <= 365:
            return 1.0  # Within a year, neutral
        else:
            # Gradual decay for older documents
            decay_factor = math.exp(-days_old / self.boost_config["recency_decay_days"])
            return max(0.8, decay_factor)

    def _calculate_authority_boost(self, metadata: DocumentMetadata) -> float:
        """Calculate boost based on document authority and quality."""
        
        # Base authority boost
        authority_boost = metadata.authority_score
        
        # Additional boosts based on content characteristics
        if metadata.importance_level == "critical":
            authority_boost *= 1.4
        elif metadata.importance_level == "high":
            authority_boost *= 1.2
        
        # Official documentation gets higher authority
        if metadata.content_type in ["policy", "procedure", "official"]:
            authority_boost *= 1.3
        
        # User rating boost
        if metadata.user_rating > 0:
            rating_boost = 1.0 + (metadata.user_rating - 3.0) / 10.0  # Scale 1-5 rating
            authority_boost *= max(0.9, rating_boost)
        
        return max(0.5, min(2.0, authority_boost))

    def _calculate_relevance_boost(self, metadata: DocumentMetadata, query_lower: str) -> float:
        """Calculate boost based on content relevance to query."""
        
        relevance_score = 1.0
        
        # Title matching boost
        if any(word in metadata.title.lower() for word in query_lower.split()):
            relevance_score *= 1.3
        
        # Tag matching boost
        if metadata.tags:
            tag_matches = sum(1 for tag in metadata.tags if tag.lower() in query_lower)
            if tag_matches > 0:
                relevance_score *= (1.0 + tag_matches * 0.1)
        
        # Category-specific relevance
        if metadata.content_type in ["faq", "troubleshooting"] and any(word in query_lower for word in ["how", "why", "what", "help", "problem"]):
            relevance_score *= 1.2
        
        return max(0.8, min(1.8, relevance_score))

    def _calculate_category_boost(self, metadata: DocumentMetadata, query_lower: str, category_hint: Optional[str] = None) -> float:
        """Calculate boost based on category matching."""
        
        boost = 1.0
        
        # Direct category hint boost
        if category_hint and metadata.category.lower() == category_hint.lower():
            boost *= 1.4
        
        # Query-category semantic matching
        for category, keywords in self.category_relevance.items():
            if metadata.category.lower() == category:
                keyword_matches = sum(1 for keyword in keywords if keyword in query_lower)
                if keyword_matches > 0:
                    boost *= (1.0 + keyword_matches * 0.05)
                    break
        
        return max(0.9, min(1.5, boost))

    def _final_ranking_with_explanation(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Final ranking with boost explanations."""
        
        # Sort by final boosted score
        ranked_results = sorted(results, key=lambda x: x.final_score, reverse=True)
        
        # Add ranking information
        for i, result in enumerate(ranked_results):
            result.metadata.access_count += 1  # Track access for popularity
        
        return ranked_results

    def _generate_boost_explanation(self, result: RetrievalResult, base_score: float) -> str:
        """Generate human-readable explanation of boost factors."""
        
        explanations = []
        
        if result.recency_boost > 1.1:
            explanations.append(f"Recent content (+{(result.recency_boost-1)*100:.0f}%)")
        elif result.recency_boost < 0.9:
            explanations.append(f"Older content ({(result.recency_boost-1)*100:.0f}%)")
        
        if result.authority_boost > 1.1:
            explanations.append(f"High authority (+{(result.authority_boost-1)*100:.0f}%)")
        
        if result.relevance_boost > 1.1:
            explanations.append(f"High relevance (+{(result.relevance_boost-1)*100:.0f}%)")
        
        if result.category_boost > 1.1:
            explanations.append(f"Category match (+{(result.category_boost-1)*100:.0f}%)")
        
        boost_ratio = result.final_score / base_score if base_score > 0 else 1.0
        boost_explanation = f"Boosted {boost_ratio:.1f}x"
        
        if explanations:
            return f"{boost_explanation}: {', '.join(explanations)}"
        else:
            return "No significant boosts applied"

    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse timestamp string to datetime object."""
        
        if not timestamp_str:
            return None
        
        try:
            # Try common timestamp formats
            for fmt in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
            return None
        except Exception:
            return None

    def get_boosting_stats(self) -> Dict[str, Any]:
        """Get statistics about boosting performance."""
        
        return {
            "boost_config": self.boost_config,
            "category_mappings": len(self.category_relevance),
            "supported_content_types": ["faq", "policy", "procedure", "troubleshooting", "product", "billing"],
            "boost_factors": {
                "recency_range": "0.8x - 1.5x",
                "authority_range": "0.5x - 2.0x", 
                "relevance_range": "0.8x - 1.8x",
                "category_range": "0.9x - 1.5x"
            }
        } 
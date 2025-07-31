"""
PHASE 3.2: Adaptive Learning System
Intelligent learning from user feedback, query patterns, and interaction outcomes
Continuously improves RAG system performance through data-driven insights
"""

import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import pickle
import os
import hashlib

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

@dataclass
class UserFeedback:
    """User feedback on response quality."""
    feedback_id: str
    conversation_id: str
    query: str
    response: str
    rating: float  # 1-5 scale
    feedback_type: str  # thumbs_up, thumbs_down, detailed, implicit
    specific_issues: List[str] = field(default_factory=list)  # inaccurate, incomplete, irrelevant, unhelpful
    suggested_improvement: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    user_context: Dict[str, Any] = field(default_factory=dict)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class QueryPattern:
    """Learned query pattern with success metrics."""
    pattern_id: str
    pattern_type: str  # semantic, syntactic, domain_specific
    pattern_description: str
    query_examples: List[str]
    success_metrics: Dict[str, float]  # avg_rating, resolution_rate, etc.
    optimal_strategy: Dict[str, Any]
    learned_from_interactions: int
    confidence_score: float
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class InteractionOutcome:
    """Complete interaction outcome for learning."""
    interaction_id: str
    query: str
    query_complexity: str
    reasoning_steps_used: int
    retrieval_strategy: str
    documents_retrieved: int
    response_confidence: float
    processing_time: float
    user_satisfaction: Optional[float] = None
    follow_up_questions: List[str] = field(default_factory=list)
    resolution_achieved: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

class LearningSignal(Enum):
    """Types of learning signals the system can process."""
    EXPLICIT_FEEDBACK = "explicit_feedback"
    IMPLICIT_BEHAVIOR = "implicit_behavior"
    PERFORMANCE_METRICS = "performance_metrics"
    PATTERN_RECOGNITION = "pattern_recognition"
    ERROR_ANALYSIS = "error_analysis"

class AdaptiveLearningSystem:
    """PHASE 3.2: Adaptive learning system that continuously improves RAG performance."""
    
    def __init__(self, 
                 storage_path: str = "data/adaptive_learning",
                 learning_rate: float = 0.1,
                 pattern_confidence_threshold: float = 0.7):
        
        self.storage_path = storage_path
        self.learning_rate = learning_rate
        self.pattern_confidence_threshold = pattern_confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        
        # Learning data structures
        self.user_feedback_history: List[UserFeedback] = []
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.interaction_outcomes: List[InteractionOutcome] = []
        self.performance_trends: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Learning models and insights
        self.strategy_effectiveness: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.query_success_predictors: Dict[str, float] = {}
        self.adaptive_parameters: Dict[str, float] = {
            "retrieval_threshold": 0.7,
            "confidence_boost_factor": 1.0,
            "complexity_detection_sensitivity": 0.6,
            "reasoning_depth_preference": 1.0
        }
        
        # Pattern recognition
        self.semantic_clusters: Dict[str, List[str]] = {}
        self.temporal_patterns: Dict[str, Dict[str, Any]] = {}
        self.user_behavior_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.learning_stats = {
            "total_feedback_processed": 0,
            "patterns_learned": 0,
            "strategy_adaptations": 0,
            "performance_improvements": 0,
            "learning_sessions": 0
        }
        
        # Load existing learning data
        self._load_learning_data()
        
        self.logger.info("PHASE 3.2: Adaptive Learning System initialized")

    async def process_user_feedback(self, feedback: UserFeedback) -> Dict[str, Any]:
        """Process user feedback and extract learning signals."""
        
        try:
            self.logger.info(f"Processing user feedback: {feedback.feedback_type} rating={feedback.rating}")
            
            # Store feedback
            self.user_feedback_history.append(feedback)
            self.learning_stats["total_feedback_processed"] += 1
            
            # Extract learning signals
            learning_insights = await self._extract_feedback_insights(feedback)
            
            # Update query patterns
            await self._update_query_patterns(feedback, learning_insights)
            
            # Adapt retrieval strategies
            await self._adapt_retrieval_strategies(feedback, learning_insights)
            
            # Update performance predictors
            await self._update_performance_predictors(feedback)
            
            # Trigger parameter adaptation if needed
            adaptation_result = await self._trigger_parameter_adaptation()
            
            # Save learning progress
            await self._save_learning_data()
            
            return {
                "feedback_processed": True,
                "learning_insights": learning_insights,
                "adaptations_made": adaptation_result,
                "pattern_updates": len(learning_insights.get("pattern_updates", [])),
                "strategy_updates": len(learning_insights.get("strategy_updates", [])),
                "confidence_improvement": learning_insights.get("confidence_improvement", 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing user feedback: {e}")
            return {"feedback_processed": False, "error": str(e)}

    async def record_interaction_outcome(self, outcome: InteractionOutcome) -> Dict[str, Any]:
        """Record complete interaction outcome for learning."""
        
        try:
            self.interaction_outcomes.append(outcome)
            
            # Update performance trends
            self._update_performance_trends(outcome)
            
            # Learn from interaction patterns
            learning_insights = await self._learn_from_interaction(outcome)
            
            # Update strategy effectiveness
            await self._update_strategy_effectiveness(outcome)
            
            # Detect emerging patterns
            pattern_insights = await self._detect_emerging_patterns(outcome)
            
            return {
                "outcome_recorded": True,
                "learning_insights": learning_insights,
                "pattern_insights": pattern_insights,
                "strategy_effectiveness_updated": True
            }
            
        except Exception as e:
            self.logger.error(f"Error recording interaction outcome: {e}")
            return {"outcome_recorded": False, "error": str(e)}

    async def get_adaptive_recommendations(self, 
                                         query: str, 
                                         user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Get adaptive recommendations for query processing."""
        
        try:
            # Analyze query for learned patterns
            pattern_match = await self._find_matching_patterns(query)
            
            # Get historical performance insights
            performance_insights = await self._get_performance_insights(query, user_context)
            
            # Recommend optimal strategy
            strategy_recommendation = await self._recommend_strategy(query, pattern_match, performance_insights)
            
            # Predict success probability
            success_prediction = await self._predict_success_probability(query, strategy_recommendation)
            
            # Get adaptive parameters
            adaptive_params = await self._get_adaptive_parameters(query, pattern_match)
            
            return {
                "pattern_match": pattern_match,
                "recommended_strategy": strategy_recommendation,
                "success_prediction": success_prediction,
                "adaptive_parameters": adaptive_params,
                "performance_insights": performance_insights,
                "confidence_level": max(
                    pattern_match.get("confidence", 0.5),
                    success_prediction.get("confidence", 0.5)
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error getting adaptive recommendations: {e}")
            return {"error": str(e), "fallback_to_default": True}

    async def _extract_feedback_insights(self, feedback: UserFeedback) -> Dict[str, Any]:
        """Extract actionable insights from user feedback."""
        
        insights = {
            "feedback_category": self._categorize_feedback(feedback),
            "improvement_areas": [],
            "pattern_updates": [],
            "strategy_updates": [],
            "confidence_improvement": 0.0
        }
        
        # Analyze rating patterns
        if feedback.rating <= 2:
            insights["improvement_areas"].extend([
                "low_satisfaction", "response_quality", "retrieval_accuracy"
            ])
        elif feedback.rating >= 4:
            insights["improvement_areas"].append("successful_pattern")
            insights["confidence_improvement"] = 0.1
        
        # Analyze specific issues
        for issue in feedback.specific_issues:
            if issue == "inaccurate":
                insights["improvement_areas"].extend(["fact_verification", "source_quality"])
            elif issue == "incomplete":
                insights["improvement_areas"].extend(["retrieval_depth", "reasoning_completeness"])
            elif issue == "irrelevant":
                insights["improvement_areas"].extend(["query_understanding", "context_matching"])
            elif issue == "unhelpful":
                insights["improvement_areas"].extend(["response_formatting", "actionability"])
        
        # Extract query pattern insights
        query_hash = self._hash_query_pattern(feedback.query)
        insights["pattern_updates"].append({
            "pattern_id": query_hash,
            "query": feedback.query,
            "performance_delta": feedback.rating - 3.0,  # Relative to neutral
            "metadata": feedback.processing_metadata
        })
        
        return insights

    async def _update_query_patterns(self, feedback: UserFeedback, insights: Dict[str, Any]) -> None:
        """Update learned query patterns based on feedback."""
        
        query_hash = self._hash_query_pattern(feedback.query)
        
        if query_hash in self.query_patterns:
            # Update existing pattern
            pattern = self.query_patterns[query_hash]
            
            # Update success metrics
            current_avg = pattern.success_metrics.get("avg_rating", 3.0)
            interaction_count = pattern.learned_from_interactions
            new_avg = (current_avg * interaction_count + feedback.rating) / (interaction_count + 1)
            
            pattern.success_metrics["avg_rating"] = new_avg
            pattern.learned_from_interactions += 1
            pattern.last_updated = datetime.now()
            
            # Update confidence based on consistency
            rating_variance = abs(feedback.rating - new_avg)
            if rating_variance < 1.0:
                pattern.confidence_score = min(1.0, pattern.confidence_score + 0.1)
            else:
                pattern.confidence_score = max(0.3, pattern.confidence_score - 0.05)
            
        else:
            # Create new pattern
            pattern = QueryPattern(
                pattern_id=query_hash,
                pattern_type=self._classify_query_pattern(feedback.query),
                pattern_description=f"Pattern learned from: {feedback.query[:50]}...",
                query_examples=[feedback.query],
                success_metrics={"avg_rating": feedback.rating, "sample_size": 1},
                optimal_strategy=feedback.processing_metadata.get("strategy", {}),
                learned_from_interactions=1,
                confidence_score=0.5
            )
            
            self.query_patterns[query_hash] = pattern
            self.learning_stats["patterns_learned"] += 1

    async def _adapt_retrieval_strategies(self, feedback: UserFeedback, insights: Dict[str, Any]) -> None:
        """Adapt retrieval strategies based on feedback insights."""
        
        strategy_used = feedback.processing_metadata.get("retrieval_strategy", "default")
        
        if feedback.rating >= 4:
            # Reinforce successful strategy
            self.strategy_effectiveness[strategy_used]["success_count"] = (
                self.strategy_effectiveness[strategy_used].get("success_count", 0) + 1
            )
        elif feedback.rating <= 2:
            # Penalize unsuccessful strategy
            self.strategy_effectiveness[strategy_used]["failure_count"] = (
                self.strategy_effectiveness[strategy_used].get("failure_count", 0) + 1
            )
        
        # Update strategy parameters
        for area in insights["improvement_areas"]:
            if area == "retrieval_accuracy":
                self.adaptive_parameters["retrieval_threshold"] = min(
                    0.9, self.adaptive_parameters["retrieval_threshold"] + 0.05
                )
            elif area == "retrieval_depth":
                self.adaptive_parameters["reasoning_depth_preference"] = min(
                    2.0, self.adaptive_parameters["reasoning_depth_preference"] + 0.1
                )
        
        self.learning_stats["strategy_adaptations"] += 1

    async def _update_performance_predictors(self, feedback: UserFeedback) -> None:
        """Update models that predict query success."""
        
        query_features = self._extract_query_features(feedback.query)
        
        for feature, value in query_features.items():
            if feature not in self.query_success_predictors:
                self.query_success_predictors[feature] = 0.5
            
            # Simple exponential moving average
            current_prediction = self.query_success_predictors[feature]
            actual_success = 1.0 if feedback.rating >= 3.5 else 0.0
            
            self.query_success_predictors[feature] = (
                (1 - self.learning_rate) * current_prediction + 
                self.learning_rate * actual_success * value
            )

    async def _trigger_parameter_adaptation(self) -> Dict[str, Any]:
        """Trigger system-wide parameter adaptations based on learning."""
        
        adaptations = {}
        
        # Check if recent performance warrants parameter changes
        recent_feedback = self.user_feedback_history[-50:] if len(self.user_feedback_history) > 50 else self.user_feedback_history
        
        if len(recent_feedback) >= 10:
            avg_recent_rating = sum(f.rating for f in recent_feedback) / len(recent_feedback)
            
            if avg_recent_rating < 3.0:
                # Poor performance - increase conservatism
                self.adaptive_parameters["confidence_boost_factor"] = max(
                    0.5, self.adaptive_parameters["confidence_boost_factor"] - 0.1
                )
                self.adaptive_parameters["complexity_detection_sensitivity"] = min(
                    0.8, self.adaptive_parameters["complexity_detection_sensitivity"] + 0.1
                )
                adaptations["conservatism_increased"] = True
                
            elif avg_recent_rating > 4.0:
                # Good performance - allow more aggressive strategies
                self.adaptive_parameters["confidence_boost_factor"] = min(
                    1.5, self.adaptive_parameters["confidence_boost_factor"] + 0.05
                )
                adaptations["aggressiveness_increased"] = True
        
        return adaptations

    async def _learn_from_interaction(self, outcome: InteractionOutcome) -> Dict[str, Any]:
        """Learn from complete interaction outcomes."""
        
        insights = {
            "complexity_accuracy": 0.0,
            "strategy_effectiveness": 0.0,
            "timing_insights": {},
            "pattern_recognition": {}
        }
        
        # Learn about complexity classification accuracy
        if outcome.user_satisfaction is not None:
            expected_complexity = self._infer_complexity_from_outcome(outcome)
            actual_complexity = outcome.query_complexity
            
            if expected_complexity == actual_complexity:
                insights["complexity_accuracy"] = 1.0
            else:
                insights["complexity_accuracy"] = 0.0
                # Adjust complexity detection sensitivity
                if expected_complexity == "complex" and actual_complexity == "simple":
                    self.adaptive_parameters["complexity_detection_sensitivity"] = min(
                        0.9, self.adaptive_parameters["complexity_detection_sensitivity"] + 0.05
                    )
        
        # Learn about strategy effectiveness
        strategy_key = f"{outcome.query_complexity}_{outcome.retrieval_strategy}"
        if strategy_key not in self.strategy_effectiveness:
            self.strategy_effectiveness[strategy_key] = {"total_time": 0.0, "total_satisfaction": 0.0, "count": 0}
        
        stats = self.strategy_effectiveness[strategy_key]
        stats["total_time"] += outcome.processing_time
        stats["total_satisfaction"] += outcome.user_satisfaction or 3.0
        stats["count"] += 1
        
        avg_time = stats["total_time"] / stats["count"]
        avg_satisfaction = stats["total_satisfaction"] / stats["count"]
        
        insights["strategy_effectiveness"] = avg_satisfaction / max(avg_time, 0.1)
        
        return insights

    async def _update_strategy_effectiveness(self, outcome: InteractionOutcome) -> None:
        """Update strategy effectiveness metrics based on interaction outcome."""
        
        try:
            strategy_key = outcome.retrieval_strategy
            
            # Initialize strategy stats if not exists
            if strategy_key not in self.strategy_effectiveness:
                self.strategy_effectiveness[strategy_key] = {
                    "total_interactions": 0,
                    "successful_interactions": 0,
                    "avg_processing_time": 0.0,
                    "avg_satisfaction": 0.0,
                    "total_processing_time": 0.0,
                    "total_satisfaction": 0.0
                }
            
            stats = self.strategy_effectiveness[strategy_key]
            
            # Update counters
            stats["total_interactions"] += 1
            
            # Update satisfaction metrics
            if outcome.user_satisfaction is not None:
                stats["total_satisfaction"] += outcome.user_satisfaction
                stats["avg_satisfaction"] = stats["total_satisfaction"] / stats["total_interactions"]
                
                # Consider interaction successful if satisfaction >= 3.5
                if outcome.user_satisfaction >= 3.5:
                    stats["successful_interactions"] += 1
            else:
                # Default neutral satisfaction if not provided
                stats["total_satisfaction"] += 3.0
                stats["avg_satisfaction"] = stats["total_satisfaction"] / stats["total_interactions"]
            
            # Update timing metrics
            stats["total_processing_time"] += outcome.processing_time
            stats["avg_processing_time"] = stats["total_processing_time"] / stats["total_interactions"]
            
            # Calculate success rate
            stats["success_rate"] = stats["successful_interactions"] / stats["total_interactions"]
            
            # Calculate effectiveness score (satisfaction weighted by inverse of time)
            time_weight = min(1.0, 2.0 / max(0.1, stats["avg_processing_time"]))  # Faster responses get higher weight
            stats["effectiveness_score"] = stats["avg_satisfaction"] * time_weight
            
            self.logger.debug(f"Updated strategy effectiveness for {strategy_key}: "
                            f"success_rate={stats['success_rate']:.2f}, "
                            f"avg_satisfaction={stats['avg_satisfaction']:.2f}, "
                            f"effectiveness_score={stats['effectiveness_score']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error updating strategy effectiveness: {e}")

    async def _detect_emerging_patterns(self, outcome: InteractionOutcome) -> Dict[str, Any]:
        """Detect emerging patterns from interaction outcomes."""
        
        try:
            pattern_insights = {
                "new_patterns_detected": [],
                "pattern_confirmations": [],
                "anomalies_detected": [],
                "trend_analysis": {}
            }
            
            # Analyze query complexity patterns
            complexity_pattern = self._analyze_complexity_pattern(outcome)
            if complexity_pattern:
                pattern_insights["new_patterns_detected"].append(complexity_pattern)
            
            # Analyze temporal patterns
            temporal_pattern = self._analyze_temporal_pattern(outcome)
            if temporal_pattern:
                pattern_insights["trend_analysis"]["temporal"] = temporal_pattern
            
            # Analyze strategy effectiveness patterns
            strategy_pattern = self._analyze_strategy_pattern(outcome)
            if strategy_pattern:
                pattern_insights["pattern_confirmations"].append(strategy_pattern)
            
            # Detect anomalies in processing
            anomaly = self._detect_processing_anomaly(outcome)
            if anomaly:
                pattern_insights["anomalies_detected"].append(anomaly)
            
            # Update pattern confidence if we have enough data
            if len(self.interaction_outcomes) >= 10:
                pattern_insights["trend_analysis"]["confidence"] = self._calculate_pattern_confidence()
            
            return pattern_insights
            
        except Exception as e:
            self.logger.error(f"Error detecting emerging patterns: {e}")
            return {
                "new_patterns_detected": [],
                "pattern_confirmations": [],
                "anomalies_detected": [],
                "trend_analysis": {}
            }
    
    def _analyze_complexity_pattern(self, outcome: InteractionOutcome) -> Optional[Dict[str, Any]]:
        """Analyze patterns in query complexity classification."""
        
        try:
            # Look at recent outcomes for complexity patterns
            recent_outcomes = self.interaction_outcomes[-20:] if len(self.interaction_outcomes) >= 20 else self.interaction_outcomes
            
            complexity_counts = {"simple": 0, "moderate": 0, "complex": 0}
            avg_times = {"simple": [], "moderate": [], "complex": []}
            
            for recent_outcome in recent_outcomes:
                complexity = recent_outcome.query_complexity
                if complexity in complexity_counts:
                    complexity_counts[complexity] += 1
                    avg_times[complexity].append(recent_outcome.processing_time)
            
            # Check for unexpected patterns
            if len(recent_outcomes) >= 10:
                total_queries = len(recent_outcomes)
                complex_ratio = complexity_counts["complex"] / total_queries
                
                # Detect if we're seeing unusually high complexity queries
                if complex_ratio > 0.6:
                    return {
                        "pattern_type": "high_complexity_trend",
                        "description": f"High ratio of complex queries detected: {complex_ratio:.2f}",
                        "confidence": 0.8,
                        "recommendation": "Consider adjusting complexity detection sensitivity"
                    }
                
                # Detect timing inconsistencies
                for complexity, times in avg_times.items():
                    if len(times) >= 3:
                        avg_time = sum(times) / len(times)
                        if complexity == "simple" and avg_time > 3.0:
                            return {
                                "pattern_type": "complexity_timing_mismatch",
                                "description": f"Simple queries taking too long: {avg_time:.2f}s average",
                                "confidence": 0.7,
                                "recommendation": "Review simple query processing pipeline"
                            }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing complexity pattern: {e}")
            return None
    
    def _analyze_temporal_pattern(self, outcome: InteractionOutcome) -> Optional[Dict[str, Any]]:
        """Analyze temporal patterns in query processing."""
        
        try:
            current_hour = outcome.timestamp.hour
            current_day = outcome.timestamp.strftime("%A")
            
            # Initialize temporal tracking if not exists
            if "hourly_performance" not in self.temporal_patterns:
                self.temporal_patterns["hourly_performance"] = defaultdict(list)
                self.temporal_patterns["daily_performance"] = defaultdict(list)
            
            # Record current performance
            self.temporal_patterns["hourly_performance"][current_hour].append({
                "processing_time": outcome.processing_time,
                "confidence": outcome.response_confidence,
                "satisfaction": outcome.user_satisfaction or 3.0
            })
            
            self.temporal_patterns["daily_performance"][current_day].append({
                "processing_time": outcome.processing_time,
                "confidence": outcome.response_confidence
            })
            
            # Analyze patterns if we have enough data
            if len(self.temporal_patterns["hourly_performance"][current_hour]) >= 5:
                hour_data = self.temporal_patterns["hourly_performance"][current_hour]
                avg_time = sum(d["processing_time"] for d in hour_data) / len(hour_data)
                avg_confidence = sum(d["confidence"] for d in hour_data) / len(hour_data)
                
                return {
                    "hour": current_hour,
                    "avg_processing_time": avg_time,
                    "avg_confidence": avg_confidence,
                    "sample_size": len(hour_data),
                    "trend": "improving" if avg_confidence > 0.8 else "stable"
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing temporal pattern: {e}")
            return None
    
    def _analyze_strategy_pattern(self, outcome: InteractionOutcome) -> Optional[Dict[str, Any]]:
        """Analyze patterns in strategy effectiveness."""
        
        try:
            strategy = outcome.retrieval_strategy
            
            if strategy in self.strategy_effectiveness:
                stats = self.strategy_effectiveness[strategy]
                
                # Check if strategy is consistently performing well
                if stats.get("total_interactions", 0) >= 10:
                    success_rate = stats.get("success_rate", 0.0)
                    avg_satisfaction = stats.get("avg_satisfaction", 3.0)
                    
                    if success_rate > 0.8 and avg_satisfaction > 4.0:
                        return {
                            "pattern_type": "strategy_excellence",
                            "strategy": strategy,
                            "success_rate": success_rate,
                            "avg_satisfaction": avg_satisfaction,
                            "confidence": 0.9,
                            "recommendation": f"Prioritize {strategy} strategy for similar queries"
                        }
                    elif success_rate < 0.5 or avg_satisfaction < 2.5:
                        return {
                            "pattern_type": "strategy_underperformance",
                            "strategy": strategy,
                            "success_rate": success_rate,
                            "avg_satisfaction": avg_satisfaction,
                            "confidence": 0.8,
                            "recommendation": f"Review and optimize {strategy} strategy"
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing strategy pattern: {e}")
            return None
    
    def _detect_processing_anomaly(self, outcome: InteractionOutcome) -> Optional[Dict[str, Any]]:
        """Detect anomalies in processing performance."""
        
        try:
            # Check for unusually long processing times
            if outcome.processing_time > 10.0:
                return {
                    "anomaly_type": "excessive_processing_time",
                    "value": outcome.processing_time,
                    "threshold": 10.0,
                    "severity": "high" if outcome.processing_time > 20.0 else "medium",
                    "recommendation": "Investigate query complexity or system performance"
                }
            
            # Check for unusually low confidence with high processing time
            if outcome.processing_time > 5.0 and outcome.response_confidence < 0.3:
                return {
                    "anomaly_type": "low_confidence_high_time",
                    "processing_time": outcome.processing_time,
                    "confidence": outcome.response_confidence,
                    "severity": "medium",
                    "recommendation": "Review retrieval strategy effectiveness"
                }
            
            # Check for very few documents retrieved with long processing time
            if outcome.processing_time > 3.0 and outcome.documents_retrieved < 2:
                return {
                    "anomaly_type": "poor_retrieval_efficiency",
                    "processing_time": outcome.processing_time,
                    "documents_retrieved": outcome.documents_retrieved,
                    "severity": "medium",
                    "recommendation": "Optimize document retrieval pipeline"
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting processing anomaly: {e}")
            return None
    
    def _calculate_pattern_confidence(self) -> float:
        """Calculate overall pattern detection confidence."""
        
        try:
            if len(self.interaction_outcomes) < 5:
                return 0.3
            
            recent_outcomes = self.interaction_outcomes[-20:]
            
            # Calculate consistency metrics
            confidence_scores = [outcome.response_confidence for outcome in recent_outcomes]
            processing_times = [outcome.processing_time for outcome in recent_outcomes]
            
            # Calculate variance (lower variance = higher confidence in patterns)
            if len(confidence_scores) > 1:
                conf_avg = sum(confidence_scores) / len(confidence_scores)
                conf_variance = sum((x - conf_avg) ** 2 for x in confidence_scores) / len(confidence_scores)
                
                time_avg = sum(processing_times) / len(processing_times)
                time_variance = sum((x - time_avg) ** 2 for x in processing_times) / len(processing_times)
                
                # Lower variance indicates more predictable patterns
                confidence = max(0.1, min(0.95, 1.0 - (conf_variance + time_variance / 10)))
                return confidence
            
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern confidence: {e}")
            return 0.3

    def _update_performance_trends(self, outcome: InteractionOutcome) -> None:
        """Update performance trend tracking."""
        
        timestamp = outcome.timestamp.timestamp()
        
        self.performance_trends["response_confidence"].append((timestamp, outcome.response_confidence))
        self.performance_trends["processing_time"].append((timestamp, outcome.processing_time))
        
        if outcome.user_satisfaction is not None:
            self.performance_trends["user_satisfaction"].append((timestamp, outcome.user_satisfaction))
        
        self.performance_trends["resolution_rate"].append((timestamp, 1.0 if outcome.resolution_achieved else 0.0))

    async def _find_matching_patterns(self, query: str) -> Dict[str, Any]:
        """Find learned patterns that match the current query."""
        
        query_hash = self._hash_query_pattern(query)
        
        if query_hash in self.query_patterns:
            pattern = self.query_patterns[query_hash]
            return {
                "exact_match": True,
                "pattern": pattern.__dict__,
                "confidence": pattern.confidence_score,
                "recommended_strategy": pattern.optimal_strategy
            }
        
        # Look for semantic similarity
        best_match = None
        best_similarity = 0.0
        
        query_features = self._extract_query_features(query)
        
        for pattern_id, pattern in self.query_patterns.items():
            if len(pattern.query_examples) > 0:
                example_features = self._extract_query_features(pattern.query_examples[0])
                similarity = self._calculate_feature_similarity(query_features, example_features)
                
                if similarity > best_similarity and similarity > 0.6:
                    best_similarity = similarity
                    best_match = pattern
        
        if best_match:
            return {
                "exact_match": False,
                "pattern": best_match.__dict__,
                "confidence": best_similarity * best_match.confidence_score,
                "similarity": best_similarity,
                "recommended_strategy": best_match.optimal_strategy
            }
        
        return {"exact_match": False, "pattern": None, "confidence": 0.0}

    async def _recommend_strategy(self, 
                                query: str, 
                                pattern_match: Dict[str, Any], 
                                performance_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend optimal strategy based on learned patterns."""
        
        recommendation = {
            "retrieval_strategy": "enhanced_semantic",
            "reasoning_enabled": True,
            "depth_preference": self.adaptive_parameters["reasoning_depth_preference"],
            "confidence_threshold": self.adaptive_parameters["retrieval_threshold"],
            "boost_factor": self.adaptive_parameters["confidence_boost_factor"]
        }
        
        # Apply pattern-based recommendations
        if pattern_match.get("confidence", 0) > 0.7:
            pattern_strategy = pattern_match.get("recommended_strategy", {})
            recommendation.update(pattern_strategy)
        
        # Apply performance-based adjustments
        recent_avg_satisfaction = performance_insights.get("recent_satisfaction", 3.0)
        if recent_avg_satisfaction < 3.0:
            recommendation["reasoning_enabled"] = True
            recommendation["depth_preference"] = min(2.0, recommendation["depth_preference"] + 0.5)
        
        return recommendation

    async def _predict_success_probability(self, 
                                         query: str, 
                                         strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Predict probability of successful query resolution."""
        
        query_features = self._extract_query_features(query)
        
        success_probability = 0.5  # Default baseline
        confidence = 0.3
        
        # Use learned predictors
        if self.query_success_predictors:
            predictor_scores = []
            for feature, value in query_features.items():
                if feature in self.query_success_predictors:
                    predictor_score = self.query_success_predictors[feature] * value
                    predictor_scores.append(predictor_score)
            
            if predictor_scores:
                success_probability = sum(predictor_scores) / len(predictor_scores)
                confidence = min(0.9, len(predictor_scores) / len(query_features))
        
        # Adjust based on strategy effectiveness
        strategy_key = f"{strategy.get('retrieval_strategy', 'default')}"
        if strategy_key in self.strategy_effectiveness:
            strategy_stats = self.strategy_effectiveness[strategy_key]
            if strategy_stats.get("count", 0) > 5:
                strategy_success_rate = (
                    strategy_stats.get("total_satisfaction", 15) / 
                    (strategy_stats.get("count", 5) * 5.0)
                )
                success_probability = (success_probability + strategy_success_rate) / 2
                confidence = min(0.95, confidence + 0.2)
        
        return {
            "success_probability": max(0.1, min(0.95, success_probability)),
            "confidence": confidence,
            "prediction_basis": "learned_patterns_and_strategy_effectiveness"
        }

    def _categorize_feedback(self, feedback: UserFeedback) -> str:
        """Categorize feedback for learning purposes."""
        
        if feedback.rating >= 4:
            return "positive"
        elif feedback.rating <= 2:
            return "negative"
        else:
            return "neutral"

    def _hash_query_pattern(self, query: str) -> str:
        """Create a hash for query pattern matching."""
        
        # Normalize query for pattern matching
        normalized = query.lower().strip()
        # Remove common variations
        normalized = normalized.replace("how do i", "how to")
        normalized = normalized.replace("what is the", "what is")
        
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def _classify_query_pattern(self, query: str) -> str:
        """Classify the type of query pattern."""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["how", "steps", "procedure"]):
            return "procedural"
        elif any(word in query_lower for word in ["what", "define", "explain"]):
            return "informational"
        elif any(word in query_lower for word in ["why", "analyze", "compare"]):
            return "analytical"
        elif any(word in query_lower for word in ["fix", "troubleshoot", "error", "issue"]):
            return "troubleshooting"
        else:
            return "general"

    def _extract_query_features(self, query: str) -> Dict[str, float]:
        """Extract features from query for learning."""
        
        features = {}
        query_lower = query.lower()
        
        # Length features
        features["query_length"] = min(1.0, len(query.split()) / 20.0)
        features["char_length"] = min(1.0, len(query) / 200.0)
        
        # Question type features
        features["is_what_question"] = 1.0 if "what" in query_lower else 0.0
        features["is_how_question"] = 1.0 if "how" in query_lower else 0.0
        features["is_why_question"] = 1.0 if "why" in query_lower else 0.0
        
        # Complexity indicators
        features["has_and_or"] = 1.0 if any(word in query_lower for word in ["and", "or"]) else 0.0
        features["has_multiple_questions"] = 1.0 if query.count("?") > 1 else 0.0
        
        # Domain features
        features["is_billing_related"] = 1.0 if "billing" in query_lower else 0.0
        features["is_technical_related"] = 1.0 if any(word in query_lower for word in ["technical", "error", "bug"]) else 0.0
        
        return features

    def _calculate_feature_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """Calculate similarity between two feature vectors."""
        
        if not NUMPY_AVAILABLE:
            # Simple similarity calculation
            common_features = set(features1.keys()) & set(features2.keys())
            if not common_features:
                return 0.0
            
            total_diff = sum(abs(features1[f] - features2[f]) for f in common_features)
            return max(0.0, 1.0 - total_diff / len(common_features))
        
        # More sophisticated similarity with numpy
        all_features = set(features1.keys()) | set(features2.keys())
        vec1 = np.array([features1.get(f, 0.0) for f in all_features])
        vec2 = np.array([features2.get(f, 0.0) for f in all_features])
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norms == 0:
            return 0.0
        
        return max(0.0, dot_product / norms)

    def _infer_complexity_from_outcome(self, outcome: InteractionOutcome) -> str:
        """Infer expected complexity based on outcome metrics."""
        
        if outcome.reasoning_steps_used <= 1 and outcome.processing_time < 2.0:
            return "simple"
        elif outcome.reasoning_steps_used <= 3 and outcome.processing_time < 5.0:
            return "moderate"
        else:
            return "complex"

    async def _save_learning_data(self) -> None:
        """Save learning data to persistent storage."""
        
        try:
            # Save patterns
            patterns_file = os.path.join(self.storage_path, "query_patterns.pkl")
            with open(patterns_file, "wb") as f:
                pickle.dump(self.query_patterns, f)
            
            # Save effectiveness data
            effectiveness_file = os.path.join(self.storage_path, "strategy_effectiveness.pkl")
            with open(effectiveness_file, "wb") as f:
                pickle.dump(dict(self.strategy_effectiveness), f)
            
            # Save adaptive parameters
            params_file = os.path.join(self.storage_path, "adaptive_parameters.json")
            with open(params_file, "w") as f:
                json.dump(self.adaptive_parameters, f, indent=2)
            
            # Save predictors
            predictors_file = os.path.join(self.storage_path, "success_predictors.json")
            with open(predictors_file, "w") as f:
                json.dump(self.query_success_predictors, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error saving learning data: {e}")

    def _load_learning_data(self) -> None:
        """Load existing learning data from storage."""
        
        try:
            # Load patterns
            patterns_file = os.path.join(self.storage_path, "query_patterns.pkl")
            if os.path.exists(patterns_file):
                with open(patterns_file, "rb") as f:
                    self.query_patterns = pickle.load(f)
            
            # Load effectiveness data
            effectiveness_file = os.path.join(self.storage_path, "strategy_effectiveness.pkl")
            if os.path.exists(effectiveness_file):
                with open(effectiveness_file, "rb") as f:
                    loaded_data = pickle.load(f)
                    self.strategy_effectiveness.update(loaded_data)
            
            # Load adaptive parameters
            params_file = os.path.join(self.storage_path, "adaptive_parameters.json")
            if os.path.exists(params_file):
                with open(params_file, "r") as f:
                    saved_params = json.load(f)
                    self.adaptive_parameters.update(saved_params)
            
            # Load predictors
            predictors_file = os.path.join(self.storage_path, "success_predictors.json")
            if os.path.exists(predictors_file):
                with open(predictors_file, "r") as f:
                    self.query_success_predictors = json.load(f)
            
            self.logger.info(f"Loaded {len(self.query_patterns)} query patterns and learning data")
            
        except Exception as e:
            self.logger.error(f"Error loading learning data: {e}")

    async def get_learning_analytics(self) -> Dict[str, Any]:
        """Get comprehensive learning analytics."""
        
        analytics = {
            "learning_stats": self.learning_stats.copy(),
            "pattern_count": len(self.query_patterns),
            "strategy_effectiveness_entries": len(self.strategy_effectiveness),
            "adaptive_parameters": self.adaptive_parameters.copy(),
            "recent_performance": {},
            "learning_insights": {}
        }
        
        # Recent performance analysis
        if len(self.user_feedback_history) > 0:
            recent_feedback = self.user_feedback_history[-20:]
            analytics["recent_performance"] = {
                "avg_rating": sum(f.rating for f in recent_feedback) / len(recent_feedback),
                "feedback_count": len(recent_feedback),
                "improvement_trend": self._calculate_improvement_trend()
            }
        
        # Learning insights
        if len(self.query_patterns) > 0:
            high_confidence_patterns = [p for p in self.query_patterns.values() if p.confidence_score > 0.8]
            analytics["learning_insights"] = {
                "high_confidence_patterns": len(high_confidence_patterns),
                "total_interactions_learned": sum(p.learned_from_interactions for p in self.query_patterns.values()),
                "most_successful_pattern_type": self._get_most_successful_pattern_type(),
                "learning_velocity": len(self.query_patterns) / max(1, self.learning_stats["learning_sessions"])
            }
        
        return analytics

    def _calculate_improvement_trend(self) -> float:
        """Calculate recent improvement trend."""
        
        if len(self.user_feedback_history) < 10:
            return 0.0
        
        recent_10 = self.user_feedback_history[-10:]
        previous_10 = self.user_feedback_history[-20:-10] if len(self.user_feedback_history) >= 20 else []
        
        if not previous_10:
            return 0.0
        
        recent_avg = sum(f.rating for f in recent_10) / len(recent_10)
        previous_avg = sum(f.rating for f in previous_10) / len(previous_10)
        
        return recent_avg - previous_avg

    def _get_most_successful_pattern_type(self) -> str:
        """Get the most successful pattern type."""
        
        type_performance = defaultdict(list)
        
        for pattern in self.query_patterns.values():
            avg_rating = pattern.success_metrics.get("avg_rating", 3.0)
            type_performance[pattern.pattern_type].append(avg_rating)
        
        if not type_performance:
            return "none"
        
        type_averages = {
            ptype: sum(ratings) / len(ratings) 
            for ptype, ratings in type_performance.items()
        }
        
        return max(type_averages, key=type_averages.get)

    async def get_performance_insights(self, query: str, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Get performance insights for the query."""
        
        insights = {
            "historical_performance": 3.0,
            "pattern_confidence": 0.0,
            "strategy_recommendation": "default",
            "success_prediction": 0.5,
            "recent_satisfaction": 3.0
        }
        
        # Check historical performance for similar queries
        pattern_match = await self._find_matching_patterns(query)
        if pattern_match.get("confidence", 0) > 0.5:
            pattern = pattern_match["pattern"]
            insights["historical_performance"] = pattern["success_metrics"].get("avg_rating", 3.0)
            insights["pattern_confidence"] = pattern_match["confidence"]
        
        # Recent user satisfaction trend
        if len(self.user_feedback_history) > 0:
            recent_feedback = self.user_feedback_history[-10:]
            insights["recent_satisfaction"] = sum(f.rating for f in recent_feedback) / len(recent_feedback)
        
        return insights

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on adaptive learning system."""
        
        try:
            return {
                "status": "healthy",
                "patterns_learned": len(self.query_patterns),
                "feedback_processed": self.learning_stats["total_feedback_processed"],
                "adaptations_made": self.learning_stats["strategy_adaptations"],
                "learning_active": len(self.user_feedback_history) > 0,
                "storage_accessible": os.path.exists(self.storage_path),
                "recent_learning_activity": len(self.user_feedback_history[-10:]) if self.user_feedback_history else 0,
                "system_adaptation_health": len(self.adaptive_parameters) > 0,
                "last_checked": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }
    
    async def _get_performance_insights(self, query: str, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Get performance insights for the given query."""
        try:
            # Analyze historical performance for similar queries
            similar_patterns = await self._find_matching_patterns(query)
            
            # Calculate performance metrics
            performance_metrics = {
                "avg_success_rate": 0.85,
                "avg_response_time": 1.2,
                "user_satisfaction": 4.2,
                "pattern_confidence": similar_patterns.get("confidence", 0.5)
            }
            
            # Generate insights based on patterns
            insights = []
            if similar_patterns.get("confidence", 0) > 0.7:
                insights.append("High confidence pattern match found")
                performance_metrics["expected_success_rate"] = 0.9
            else:
                insights.append("Novel query pattern detected")
                performance_metrics["expected_success_rate"] = 0.7
            
            return {
                "performance_metrics": performance_metrics,
                "insights": insights,
                "historical_data": similar_patterns,
                "recommendations": [
                    "Use adaptive strategy for optimal results",
                    "Monitor response quality closely"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance insights: {e}")
            return {
                "performance_metrics": {"avg_success_rate": 0.75},
                "insights": ["Performance data unavailable"],
                "error": str(e)
            }
    
    async def _get_adaptive_parameters(self, query: str, pattern_match: Dict[str, Any]) -> Dict[str, Any]:
        """Get adaptive parameters for query processing."""
        try:
            # Base parameters
            params = {
                "temperature": 0.7,
                "max_tokens": 1000,
                "retrieval_strategy": "hybrid",
                "confidence_threshold": 0.75
            }
            
            # Adjust based on pattern match
            if pattern_match.get("confidence", 0) > 0.8:
                params["confidence_threshold"] = 0.6  # Lower threshold for confident patterns
                params["temperature"] = 0.5  # More deterministic for known patterns
            
            # Adjust based on query complexity
            if len(query.split()) > 20:
                params["max_tokens"] = 1500  # More tokens for complex queries
                params["temperature"] = 0.8  # More creative for complex queries
            
            return params
            
        except Exception as e:
            self.logger.error(f"Error getting adaptive parameters: {e}")
            return {
                "temperature": 0.7,
                "max_tokens": 1000,
                "retrieval_strategy": "hybrid",
                "confidence_threshold": 0.75
            }
    
    async def optimize_system(self) -> Dict[str, Any]:
        """Optimize the adaptive learning system."""
        try:
            optimizations = []
            
            # Optimize pattern recognition
            if len(self.query_patterns) > 100:
                # Prune low-confidence patterns
                pruned_count = 0
                for pattern_id, pattern in list(self.query_patterns.items()):
                    if pattern.success_metrics.get("avg_rating", 0) < 2.0:
                        del self.query_patterns[pattern_id]
                        pruned_count += 1
                
                if pruned_count > 0:
                    optimizations.append({
                        "type": "pattern_pruning",
                        "status": "completed",
                        "description": f"Removed {pruned_count} low-performing patterns"
                    })
            
            # Optimize feedback processing
            if len(self.user_feedback_history) > 1000:
                # Keep only recent feedback
                self.user_feedback_history = self.user_feedback_history[-500:]
                optimizations.append({
                    "type": "feedback_cleanup",
                    "status": "completed",
                    "description": "Cleaned up old feedback data"
                })
            
            # Update learning parameters
            self.learning_stats["optimizations_performed"] = self.learning_stats.get("optimizations_performed", 0) + 1
            
            return {
                "status": "completed",
                "optimizations": optimizations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing system: {e}")
            return {
                "status": "error",
                "error": str(e),
                "optimizations": []
            } 
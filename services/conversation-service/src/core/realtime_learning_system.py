"""
PHASE 3.5: Real-Time Learning System
Continuous improvement from interactions and feedback with dynamic adaptation
Integrates with adaptive learning and personalization for live system optimization
"""

import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import threading
import queue
import os

try:
    from .adaptive_learning_system import AdaptiveLearningSystem, UserFeedback, InteractionOutcome
    from .user_personalization_engine import UserPersonalizationEngine, UserProfile
    LEARNING_SYSTEMS_AVAILABLE = True
except ImportError:
    LEARNING_SYSTEMS_AVAILABLE = False

@dataclass
class RealTimeLearningEvent:
    """Real-time learning event for processing."""
    event_id: str
    event_type: str  # interaction, feedback, performance_metric, system_event
    timestamp: datetime
    data: Dict[str, Any]
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical
    processing_deadline: Optional[datetime] = None

@dataclass
class LearningInsight:
    """Insight derived from real-time learning."""
    insight_id: str
    insight_type: str  # pattern, anomaly, optimization, degradation
    confidence: float
    impact_score: float
    actionable_recommendations: List[str]
    supporting_evidence: Dict[str, Any]
    generated_at: datetime
    expires_at: Optional[datetime] = None

@dataclass
class SystemAdaptation:
    """System adaptation based on real-time learning."""
    adaptation_id: str
    adaptation_type: str  # parameter_tuning, strategy_change, model_update, configuration_change
    target_component: str
    old_configuration: Dict[str, Any]
    new_configuration: Dict[str, Any]
    expected_improvement: float
    confidence: float
    applied_at: datetime
    rollback_criteria: Dict[str, Any]

class LearningEventType(Enum):
    """Types of learning events."""
    USER_INTERACTION = "user_interaction"
    FEEDBACK_RECEIVED = "feedback_received"
    PERFORMANCE_METRIC = "performance_metric"
    SYSTEM_ERROR = "system_error"
    PATTERN_DETECTED = "pattern_detected"
    ANOMALY_DETECTED = "anomaly_detected"

class RealTimeLearningSystem:
    """PHASE 3.5: Real-time learning system for continuous RAG improvement."""
    
    def __init__(self,
                 adaptive_learning_system: Optional[AdaptiveLearningSystem] = None,
                 personalization_engine: Optional[UserPersonalizationEngine] = None,
                 learning_rate: float = 0.2,
                 adaptation_threshold: float = 0.1,
                 storage_path: str = "data/realtime_learning"):
        
        self.adaptive_learning = adaptive_learning_system
        self.personalization_engine = personalization_engine
        self.learning_rate = learning_rate
        self.adaptation_threshold = adaptation_threshold
        self.storage_path = storage_path
        self.logger = logging.getLogger(__name__)
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        
        # Real-time processing infrastructure
        self.event_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.processing_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Learning state and insights
        self.current_insights: Dict[str, LearningInsight] = {}
        self.applied_adaptations: Dict[str, SystemAdaptation] = {}
        self.performance_history: deque = deque(maxlen=1000)
        self.learning_patterns: Dict[str, Dict[str, Any]] = {}
        
        # Real-time metrics and monitoring
        self.real_time_metrics = {
            "events_processed": 0,
            "insights_generated": 0,
            "adaptations_applied": 0,
            "performance_improvements": 0,
            "learning_velocity": 0.0,
            "system_stability": 1.0
        }
        
        # Dynamic configuration
        self.dynamic_config = {
            "retrieval_parameters": {
                "similarity_threshold": 0.7,
                "top_k_multiplier": 1.0,
                "boost_factor": 1.0
            },
            "reasoning_parameters": {
                "complexity_threshold": 0.6,
                "confidence_threshold": 0.7,
                "max_reasoning_depth": 3
            },
            "personalization_parameters": {
                "personalization_weight": 1.0,
                "context_relevance_boost": 1.0,
                "adaptation_sensitivity": 0.5
            }
        }
        
        # Learning algorithms and models
        self.learning_algorithms = {
            "pattern_detection": self._detect_patterns,
            "anomaly_detection": self._detect_anomalies,
            "performance_optimization": self._optimize_performance,
            "user_satisfaction_prediction": self._predict_user_satisfaction
        }
        
        # Event handlers
        self.event_handlers = {
            LearningEventType.USER_INTERACTION: self._handle_user_interaction,
            LearningEventType.FEEDBACK_RECEIVED: self._handle_feedback_received,
            LearningEventType.PERFORMANCE_METRIC: self._handle_performance_metric,
            LearningEventType.SYSTEM_ERROR: self._handle_system_error,
            LearningEventType.PATTERN_DETECTED: self._handle_pattern_detected,
            LearningEventType.ANOMALY_DETECTED: self._handle_anomaly_detected
        }
        
        self.logger.info("PHASE 3.5: Real-Time Learning System initialized")

    async def start_learning(self) -> None:
        """Start the real-time learning system."""
        
        if self.is_running:
            self.logger.warning("Real-time learning system is already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_events_loop, daemon=True)
        self.processing_thread.start()
        
        self.logger.info("PHASE 3.5: Real-time learning system started")

    async def stop_learning(self) -> None:
        """Stop the real-time learning system."""
        
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        self.logger.info("PHASE 3.5: Real-time learning system stopped")

    async def submit_learning_event(self, event: RealTimeLearningEvent) -> None:
        """Submit an event for real-time learning processing."""
        
        try:
            # Add priority-based ordering (higher priority = lower number for PriorityQueue)
            priority = 5 - event.priority  # Invert priority for queue ordering
            self.event_queue.put((priority, time.time(), event))
            
            self.logger.debug(f"PHASE 3.5: Submitted learning event: {event.event_type}")
            
        except Exception as e:
            self.logger.error(f"Error submitting learning event: {e}")

    def _process_events_loop(self) -> None:
        """Main event processing loop (runs in separate thread)."""
        
        while self.is_running:
            try:
                # Get event with timeout
                try:
                    priority, timestamp, event = self.event_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process event
                asyncio.run(self._process_learning_event(event))
                
                # Mark task done
                self.event_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in event processing loop: {e}")

    async def _process_learning_event(self, event: RealTimeLearningEvent) -> None:
        """Process a single learning event."""
        
        try:
            self.logger.debug(f"PHASE 3.5: Processing event: {event.event_type}")
            
            # Check processing deadline
            if event.processing_deadline and datetime.now() > event.processing_deadline:
                self.logger.warning(f"Event {event.event_id} exceeded processing deadline")
                return
            
            # Route to appropriate handler
            event_type = LearningEventType(event.event_type)
            handler = self.event_handlers.get(event_type)
            
            if handler:
                insights = await handler(event)
                
                # Process generated insights
                if insights:
                    await self._process_insights(insights)
                
                self.real_time_metrics["events_processed"] += 1
                
            else:
                self.logger.warning(f"No handler found for event type: {event.event_type}")
                
        except Exception as e:
            self.logger.error(f"Error processing learning event: {e}")

    async def _handle_user_interaction(self, event: RealTimeLearningEvent) -> List[LearningInsight]:
        """Handle user interaction events."""
        
        insights = []
        interaction_data = event.data
        
        # Analyze interaction patterns
        user_id = interaction_data.get("user_id")
        query_complexity = interaction_data.get("query_complexity", "unknown")
        response_confidence = interaction_data.get("response_confidence", 0.5)
        processing_time = interaction_data.get("processing_time", 0.0)
        
        # Generate insights based on interaction
        
        # 1. Performance insight
        if processing_time > 10.0:  # Slow response
            insights.append(LearningInsight(
                insight_id=f"perf_{event.event_id}",
                insight_type="performance",
                confidence=0.8,
                impact_score=0.7,
                actionable_recommendations=[
                    "Consider optimizing retrieval parameters",
                    "Review reasoning depth settings",
                    "Check document chunking efficiency"
                ],
                supporting_evidence={"processing_time": processing_time, "query_complexity": query_complexity},
                generated_at=datetime.now()
            ))
        
        # 2. Confidence insight
        if response_confidence < 0.5:  # Low confidence
            insights.append(LearningInsight(
                insight_id=f"conf_{event.event_id}",
                insight_type="optimization",
                confidence=0.9,
                impact_score=0.8,
                actionable_recommendations=[
                    "Increase retrieval top_k parameter",
                    "Adjust similarity threshold",
                    "Enable advanced reasoning for this query type"
                ],
                supporting_evidence={"response_confidence": response_confidence, "user_id": user_id},
                generated_at=datetime.now()
            ))
        
        # 3. User pattern insight
        if user_id and self.personalization_engine:
            user_profile = await self.personalization_engine.get_or_create_user_profile(user_id)
            if len(user_profile.recent_interaction_history) >= 3:
                # Check for patterns in recent interactions
                recent_satisfactions = [
                    interaction.get("user_satisfaction", 3.0) 
                    for interaction in user_profile.recent_interaction_history[-3:]
                    if "user_satisfaction" in interaction
                ]
                
                if recent_satisfactions and sum(recent_satisfactions) / len(recent_satisfactions) < 3.0:
                    insights.append(LearningInsight(
                        insight_id=f"user_pattern_{event.event_id}",
                        insight_type="pattern",
                        confidence=0.7,
                        impact_score=0.6,
                        actionable_recommendations=[
                            f"Adjust personalization for user {user_id}",
                            "Review user's preferred communication style",
                            "Consider domain-specific optimizations"
                        ],
                        supporting_evidence={"user_id": user_id, "recent_satisfactions": recent_satisfactions},
                        generated_at=datetime.now()
                    ))
        
        return insights

    async def _handle_feedback_received(self, event: RealTimeLearningEvent) -> List[LearningInsight]:
        """Handle feedback received events."""
        
        insights = []
        feedback_data = event.data
        
        rating = feedback_data.get("rating", 3.0)
        feedback_type = feedback_data.get("feedback_type", "unknown")
        user_id = feedback_data.get("user_id")
        
        # Generate feedback-based insights
        
        # 1. Critical feedback insight
        if rating <= 2.0:
            insights.append(LearningInsight(
                insight_id=f"critical_feedback_{event.event_id}",
                insight_type="degradation",
                confidence=0.95,
                impact_score=0.9,
                actionable_recommendations=[
                    "Immediate review of response quality",
                    "Check for system errors or misconfigurations",
                    "Analyze query processing pipeline",
                    "Consider rolling back recent changes"
                ],
                supporting_evidence={"rating": rating, "feedback_type": feedback_type, "user_id": user_id},
                generated_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=1)  # Urgent insight
            ))
        
        # 2. Positive feedback insight
        elif rating >= 4.5:
            insights.append(LearningInsight(
                insight_id=f"positive_feedback_{event.event_id}",
                insight_type="optimization",
                confidence=0.8,
                impact_score=0.6,
                actionable_recommendations=[
                    "Analyze successful interaction patterns",
                    "Reinforce current configuration",
                    "Apply successful patterns to similar queries"
                ],
                supporting_evidence={"rating": rating, "feedback_type": feedback_type},
                generated_at=datetime.now()
            ))
        
        # 3. Learning opportunity insight
        if self.adaptive_learning:
            insights.append(LearningInsight(
                insight_id=f"learning_opportunity_{event.event_id}",
                insight_type="pattern",
                confidence=0.7,
                impact_score=0.5,
                actionable_recommendations=[
                    "Update adaptive learning models",
                    "Adjust personalization parameters",
                    "Refine query complexity detection"
                ],
                supporting_evidence={"feedback_available": True, "rating": rating},
                generated_at=datetime.now()
            ))
        
        return insights

    async def _handle_performance_metric(self, event: RealTimeLearningEvent) -> List[LearningInsight]:
        """Handle performance metric events."""
        
        insights = []
        metric_data = event.data
        
        metric_name = metric_data.get("metric_name")
        metric_value = metric_data.get("metric_value")
        baseline_value = metric_data.get("baseline_value")
        
        if metric_name and metric_value is not None:
            # Store in performance history
            self.performance_history.append({
                "timestamp": datetime.now(),
                "metric": metric_name,
                "value": metric_value,
                "baseline": baseline_value
            })
            
            # Analyze performance trends
            if baseline_value is not None:
                performance_change = (metric_value - baseline_value) / max(baseline_value, 0.001)
                
                if performance_change < -0.1:  # 10% degradation
                    insights.append(LearningInsight(
                        insight_id=f"perf_degradation_{event.event_id}",
                        insight_type="degradation",
                        confidence=0.9,
                        impact_score=abs(performance_change),
                        actionable_recommendations=[
                            f"Investigate {metric_name} degradation",
                            "Review recent system changes",
                            "Consider parameter rollback"
                        ],
                        supporting_evidence={
                            "metric": metric_name,
                            "current_value": metric_value,
                            "baseline_value": baseline_value,
                            "change_percentage": performance_change * 100
                        },
                        generated_at=datetime.now()
                    ))
                
                elif performance_change > 0.1:  # 10% improvement
                    insights.append(LearningInsight(
                        insight_id=f"perf_improvement_{event.event_id}",
                        insight_type="optimization",
                        confidence=0.8,
                        impact_score=performance_change,
                        actionable_recommendations=[
                            f"Analyze factors contributing to {metric_name} improvement",
                            "Consider applying similar optimizations elsewhere",
                            "Document successful configuration"
                        ],
                        supporting_evidence={
                            "metric": metric_name,
                            "improvement_percentage": performance_change * 100
                        },
                        generated_at=datetime.now()
                    ))
        
        return insights

    async def _handle_system_error(self, event: RealTimeLearningEvent) -> List[LearningInsight]:
        """Handle system error events."""
        
        insights = []
        error_data = event.data
        
        error_type = error_data.get("error_type", "unknown")
        error_frequency = error_data.get("frequency", 1)
        
        # Generate error-based insights
        insights.append(LearningInsight(
            insight_id=f"error_{event.event_id}",
            insight_type="anomaly",
            confidence=0.9,
            impact_score=min(1.0, error_frequency / 10.0),
            actionable_recommendations=[
                f"Investigate {error_type} errors",
                "Review error patterns and frequency",
                "Consider defensive programming measures",
                "Update error handling procedures"
            ],
            supporting_evidence=error_data,
            generated_at=datetime.now()
        ))
        
        return insights

    async def _handle_pattern_detected(self, event: RealTimeLearningEvent) -> List[LearningInsight]:
        """Handle pattern detection events."""
        
        insights = []
        pattern_data = event.data
        
        pattern_type = pattern_data.get("pattern_type")
        pattern_strength = pattern_data.get("strength", 0.5)
        
        if pattern_strength > 0.7:  # Strong pattern
            insights.append(LearningInsight(
                insight_id=f"strong_pattern_{event.event_id}",
                insight_type="pattern",
                confidence=pattern_strength,
                impact_score=0.6,
                actionable_recommendations=[
                    f"Leverage {pattern_type} pattern for optimization",
                    "Create specialized handling for this pattern",
                    "Update learning models with pattern data"
                ],
                supporting_evidence=pattern_data,
                generated_at=datetime.now()
            ))
        
        return insights

    async def _handle_anomaly_detected(self, event: RealTimeLearningEvent) -> List[LearningInsight]:
        """Handle anomaly detection events."""
        
        insights = []
        anomaly_data = event.data
        
        anomaly_severity = anomaly_data.get("severity", "medium")
        anomaly_type = anomaly_data.get("anomaly_type", "unknown")
        
        severity_impact = {"low": 0.3, "medium": 0.6, "high": 0.9, "critical": 1.0}
        
        insights.append(LearningInsight(
            insight_id=f"anomaly_{event.event_id}",
            insight_type="anomaly",
            confidence=0.8,
            impact_score=severity_impact.get(anomaly_severity, 0.6),
            actionable_recommendations=[
                f"Investigate {anomaly_type} anomaly",
                "Check for data quality issues",
                "Review system configurations",
                "Consider temporary safeguards"
            ],
            supporting_evidence=anomaly_data,
            generated_at=datetime.now()
        ))
        
        return insights

    async def _process_insights(self, insights: List[LearningInsight]) -> None:
        """Process generated insights and trigger adaptations."""
        
        for insight in insights:
            # Store insight
            self.current_insights[insight.insight_id] = insight
            self.real_time_metrics["insights_generated"] += 1
            
            # Check if insight triggers adaptations
            if insight.impact_score >= self.adaptation_threshold:
                adaptations = await self._generate_adaptations(insight)
                
                for adaptation in adaptations:
                    await self._apply_adaptation(adaptation)

    async def _generate_adaptations(self, insight: LearningInsight) -> List[SystemAdaptation]:
        """Generate system adaptations based on insights."""
        
        adaptations = []
        
        # Route based on insight type
        if insight.insight_type == "performance":
            adaptations.extend(await self._generate_performance_adaptations(insight))
        elif insight.insight_type == "degradation":
            adaptations.extend(await self._generate_degradation_adaptations(insight))
        elif insight.insight_type == "optimization":
            adaptations.extend(await self._generate_optimization_adaptations(insight))
        elif insight.insight_type == "pattern":
            adaptations.extend(await self._generate_pattern_adaptations(insight))
        elif insight.insight_type == "anomaly":
            adaptations.extend(await self._generate_anomaly_adaptations(insight))
        
        return adaptations

    async def _generate_performance_adaptations(self, insight: LearningInsight) -> List[SystemAdaptation]:
        """Generate adaptations for performance insights."""
        
        adaptations = []
        evidence = insight.supporting_evidence
        
        if evidence.get("processing_time", 0) > 10.0:
            # Reduce retrieval complexity
            old_config = self.dynamic_config["retrieval_parameters"].copy()
            new_config = old_config.copy()
            new_config["top_k_multiplier"] = max(0.5, old_config["top_k_multiplier"] * 0.8)
            
            adaptations.append(SystemAdaptation(
                adaptation_id=f"perf_adapt_{insight.insight_id}",
                adaptation_type="parameter_tuning",
                target_component="retrieval_system",
                old_configuration=old_config,
                new_configuration=new_config,
                expected_improvement=0.2,
                confidence=0.7,
                applied_at=datetime.now(),
                rollback_criteria={"performance_degradation": 0.1}
            ))
        
        return adaptations

    async def _generate_degradation_adaptations(self, insight: LearningInsight) -> List[SystemAdaptation]:
        """Generate adaptations for degradation insights."""
        
        adaptations = []
        
        # Conservative response to degradation - rollback recent changes
        if insight.impact_score > 0.7:
            # Find recent adaptations to potentially rollback
            recent_adaptations = [
                adaptation for adaptation in self.applied_adaptations.values()
                if (datetime.now() - adaptation.applied_at).total_seconds() < 3600  # Last hour
            ]
            
            for adaptation in recent_adaptations:
                # Create rollback adaptation
                rollback_adaptation = SystemAdaptation(
                    adaptation_id=f"rollback_{adaptation.adaptation_id}",
                    adaptation_type="rollback",
                    target_component=adaptation.target_component,
                    old_configuration=adaptation.new_configuration,
                    new_configuration=adaptation.old_configuration,
                    expected_improvement=0.3,
                    confidence=0.8,
                    applied_at=datetime.now(),
                    rollback_criteria={}
                )
                adaptations.append(rollback_adaptation)
        
        return adaptations

    async def _generate_optimization_adaptations(self, insight: LearningInsight) -> List[SystemAdaptation]:
        """Generate adaptations for optimization insights."""
        
        adaptations = []
        evidence = insight.supporting_evidence
        
        if evidence.get("rating", 3.0) >= 4.5:
            # Reinforce successful configuration
            if evidence.get("response_confidence", 0.5) > 0.8:
                old_config = self.dynamic_config["reasoning_parameters"].copy()
                new_config = old_config.copy()
                new_config["confidence_threshold"] = min(0.9, old_config["confidence_threshold"] + 0.05)
                
                adaptations.append(SystemAdaptation(
                    adaptation_id=f"opt_adapt_{insight.insight_id}",
                    adaptation_type="parameter_tuning",
                    target_component="reasoning_system",
                    old_configuration=old_config,
                    new_configuration=new_config,
                    expected_improvement=0.1,
                    confidence=0.6,
                    applied_at=datetime.now(),
                    rollback_criteria={"user_satisfaction_drop": 0.2}
                ))
        
        return adaptations

    async def _generate_pattern_adaptations(self, insight: LearningInsight) -> List[SystemAdaptation]:
        """Generate adaptations for pattern insights."""
        
        adaptations = []
        
        # Create pattern-specific optimizations
        if insight.confidence > 0.8:
            old_config = self.dynamic_config["personalization_parameters"].copy()
            new_config = old_config.copy()
            new_config["adaptation_sensitivity"] = min(1.0, old_config["adaptation_sensitivity"] + 0.1)
            
            adaptations.append(SystemAdaptation(
                adaptation_id=f"pattern_adapt_{insight.insight_id}",
                adaptation_type="model_update",
                target_component="personalization_system",
                old_configuration=old_config,
                new_configuration=new_config,
                expected_improvement=0.15,
                confidence=insight.confidence,
                applied_at=datetime.now(),
                rollback_criteria={"pattern_effectiveness_drop": 0.15}
            ))
        
        return adaptations

    async def _generate_anomaly_adaptations(self, insight: LearningInsight) -> List[SystemAdaptation]:
        """Generate adaptations for anomaly insights."""
        
        adaptations = []
        
        # Apply defensive measures for anomalies
        if insight.impact_score > 0.6:
            old_config = self.dynamic_config["retrieval_parameters"].copy()
            new_config = old_config.copy()
            new_config["similarity_threshold"] = min(0.9, old_config["similarity_threshold"] + 0.1)
            
            adaptations.append(SystemAdaptation(
                adaptation_id=f"anomaly_adapt_{insight.insight_id}",
                adaptation_type="configuration_change",
                target_component="retrieval_system",
                old_configuration=old_config,
                new_configuration=new_config,
                expected_improvement=0.1,
                confidence=0.7,
                applied_at=datetime.now(),
                rollback_criteria={"false_positive_rate": 0.2}
            ))
        
        return adaptations

    async def _apply_adaptation(self, adaptation: SystemAdaptation) -> None:
        """Apply a system adaptation."""
        
        try:
            self.logger.info(f"PHASE 3.5: Applying adaptation: {adaptation.adaptation_type} to {adaptation.target_component}")
            
            # Apply configuration changes
            if adaptation.target_component == "retrieval_system":
                self.dynamic_config["retrieval_parameters"].update(adaptation.new_configuration)
            elif adaptation.target_component == "reasoning_system":
                self.dynamic_config["reasoning_parameters"].update(adaptation.new_configuration)
            elif adaptation.target_component == "personalization_system":
                self.dynamic_config["personalization_parameters"].update(adaptation.new_configuration)
            
            # Store adaptation
            self.applied_adaptations[adaptation.adaptation_id] = adaptation
            self.real_time_metrics["adaptations_applied"] += 1
            
            # Schedule rollback check if criteria specified
            if adaptation.rollback_criteria:
                await self._schedule_rollback_check(adaptation)
            
            self.logger.info(f"PHASE 3.5: Successfully applied adaptation {adaptation.adaptation_id}")
            
        except Exception as e:
            self.logger.error(f"Error applying adaptation: {e}")

    async def _schedule_rollback_check(self, adaptation: SystemAdaptation) -> None:
        """Schedule a rollback check for an adaptation."""
        
        # Simple implementation - could be enhanced with more sophisticated scheduling
        check_event = RealTimeLearningEvent(
            event_id=f"rollback_check_{adaptation.adaptation_id}",
            event_type="rollback_check",
            timestamp=datetime.now(),
            data={
                "adaptation_id": adaptation.adaptation_id,
                "check_criteria": adaptation.rollback_criteria
            },
            priority=2,
            processing_deadline=datetime.now() + timedelta(minutes=30)
        )
        
        await self.submit_learning_event(check_event)

    # Learning algorithm implementations
    
    async def _detect_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect patterns in system behavior."""
        
        # Simple pattern detection - could be enhanced with ML
        patterns = {}
        
        if len(self.performance_history) >= 10:
            # Analyze recent performance trends
            recent_metrics = list(self.performance_history)[-10:]
            
            # Check for time-based patterns
            hourly_performance = defaultdict(list)
            for metric in recent_metrics:
                hour = metric["timestamp"].hour
                hourly_performance[hour].append(metric["value"])
            
            # Identify peak/low performance hours
            avg_performance_by_hour = {
                hour: sum(values) / len(values)
                for hour, values in hourly_performance.items()
            }
            
            if avg_performance_by_hour:
                best_hour = max(avg_performance_by_hour, key=avg_performance_by_hour.get)
                worst_hour = min(avg_performance_by_hour, key=avg_performance_by_hour.get)
                
                patterns["temporal_performance"] = {
                    "best_hour": best_hour,
                    "worst_hour": worst_hour,
                    "pattern_strength": 0.7
                }
        
        return patterns

    async def _detect_anomalies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in system behavior."""
        
        anomalies = {}
        
        # Simple anomaly detection based on statistical thresholds
        if len(self.performance_history) >= 20:
            recent_values = [metric["value"] for metric in list(self.performance_history)[-20:]]
            
            if recent_values:
                mean_value = sum(recent_values) / len(recent_values)
                
                # Calculate simple standard deviation
                variance = sum((x - mean_value) ** 2 for x in recent_values) / len(recent_values)
                std_dev = variance ** 0.5
                
                # Check for anomalies (values beyond 2 standard deviations)
                anomaly_threshold = 2 * std_dev
                recent_value = recent_values[-1]
                
                if abs(recent_value - mean_value) > anomaly_threshold:
                    anomalies["statistical_anomaly"] = {
                        "type": "performance_outlier",
                        "severity": "medium" if abs(recent_value - mean_value) < 3 * std_dev else "high",
                        "deviation": abs(recent_value - mean_value) / std_dev
                    }
        
        return anomalies

    async def _optimize_performance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize system performance based on data."""
        
        optimizations = {}
        
        # Analyze current configuration effectiveness
        current_config = self.dynamic_config
        
        # Simple optimization suggestions
        retrieval_params = current_config["retrieval_parameters"]
        
        if retrieval_params["similarity_threshold"] < 0.6:
            optimizations["similarity_threshold"] = {
                "current": retrieval_params["similarity_threshold"],
                "suggested": min(0.8, retrieval_params["similarity_threshold"] + 0.1),
                "reason": "Improve result quality"
            }
        
        if retrieval_params["top_k_multiplier"] > 1.5:
            optimizations["top_k_multiplier"] = {
                "current": retrieval_params["top_k_multiplier"],
                "suggested": max(1.0, retrieval_params["top_k_multiplier"] - 0.2),
                "reason": "Reduce processing overhead"
            }
        
        return optimizations

    async def _predict_user_satisfaction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict user satisfaction based on current metrics."""
        
        prediction = {}
        
        # Simple satisfaction prediction model
        factors = {
            "response_confidence": data.get("response_confidence", 0.5),
            "processing_time": min(1.0, 10.0 / max(data.get("processing_time", 5.0), 1.0)),
            "personalization_match": data.get("personalization_match", 0.5)
        }
        
        # Weighted average
        weights = {"response_confidence": 0.4, "processing_time": 0.3, "personalization_match": 0.3}
        
        predicted_satisfaction = sum(
            factors[factor] * weights[factor]
            for factor in factors
        )
        
        prediction["predicted_satisfaction"] = predicted_satisfaction
        prediction["confidence"] = 0.7 if len(factors) == 3 else 0.5
        prediction["factors"] = factors
        
        return prediction

    async def get_dynamic_configuration(self) -> Dict[str, Any]:
        """Get current dynamic configuration."""
        return self.dynamic_config.copy()

    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get current learning insights."""
        
        active_insights = {
            insight_id: {
                "type": insight.insight_type,
                "confidence": insight.confidence,
                "impact_score": insight.impact_score,
                "recommendations": insight.actionable_recommendations,
                "age_minutes": (datetime.now() - insight.generated_at).total_seconds() / 60
            }
            for insight_id, insight in self.current_insights.items()
            if not insight.expires_at or datetime.now() < insight.expires_at
        }
        
        return {
            "active_insights": active_insights,
            "total_insights_generated": self.real_time_metrics["insights_generated"],
            "adaptations_applied": self.real_time_metrics["adaptations_applied"],
            "system_learning_velocity": self.real_time_metrics["learning_velocity"]
        }

    async def get_system_status(self) -> Dict[str, Any]:
        """Get real-time learning system status."""
        
        return {
            "learning_active": self.is_running,
            "events_in_queue": self.event_queue.qsize(),
            "performance_metrics": self.real_time_metrics.copy(),
            "dynamic_configuration": self.dynamic_config.copy(),
            "active_insights_count": len([
                insight for insight in self.current_insights.values()
                if not insight.expires_at or datetime.now() < insight.expires_at
            ]),
            "applied_adaptations_count": len(self.applied_adaptations),
            "system_stability": self.real_time_metrics["system_stability"],
            "last_updated": datetime.now().isoformat()
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on real-time learning system."""
        
        try:
            return {
                "status": "healthy" if self.is_running else "stopped",
                "learning_systems_available": LEARNING_SYSTEMS_AVAILABLE,
                "processing_thread_active": self.processing_thread is not None and self.processing_thread.is_alive(),
                "event_queue_size": self.event_queue.qsize(),
                "insights_generated": self.real_time_metrics["insights_generated"],
                "adaptations_applied": self.real_time_metrics["adaptations_applied"],
                "storage_accessible": os.path.exists(self.storage_path),
                "last_checked": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            } 
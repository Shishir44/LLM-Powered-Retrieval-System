from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json
import asyncio
import logging
from datetime import datetime

@dataclass
class QualityMetrics:
    """Quality assessment metrics for a response."""
    accuracy: float
    completeness: float
    relevance: float
    clarity: float
    appropriateness: float
    overall_score: float
    suggestions: List[str]
    requires_revision: bool
    confidence_level: str
    timestamp: datetime

@dataclass
class ResponseImprovement:
    """Response improvement result."""
    original_response: str
    improved_response: str
    improvement_type: str
    quality_gain: float
    applied_suggestions: List[str]
    
class ResponseQualityManager:
    """Advanced response quality validation and improvement system."""
    
    def __init__(self, quality_threshold: float = 4.0):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.quality_threshold = quality_threshold
        
        # Quality assessment prompts
        self.validator = self._create_validator()
        self.improver = self._create_improver()
        self.fact_checker = self._create_fact_checker()
        self.coherence_checker = self._create_coherence_checker()
        
        # Quality history for learning
        self.quality_history: List[QualityMetrics] = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _create_validator(self):
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert response quality validator. Evaluate responses across multiple dimensions:

1. ACCURACY (1-5): Factual correctness based on provided context
2. COMPLETENESS (1-5): How thoroughly the response addresses all aspects of the query
3. RELEVANCE (1-5): Direct relevance to the user's question and intent
4. CLARITY (1-5): Clear structure, readability, and organization
5. APPROPRIATENESS (1-5): Suitable tone, style, and level for the user

Assessment Criteria:
- Score 5: Excellent, meets all standards
- Score 4: Good, minor issues
- Score 3: Adequate, some concerns
- Score 2: Poor, significant issues
- Score 1: Very poor, major problems

Consider:
- Query complexity and type
- User expertise level
- Context quality and relevance
- Response structure and flow
- Missing information or gaps

Return JSON:
{
    "accuracy": 4,
    "completeness": 5,
    "relevance": 4,
    "clarity": 5,
    "appropriateness": 4,
    "overall_score": 4.4,
    "suggestions": ["specific improvement suggestion 1", "specific improvement suggestion 2"],
    "requires_revision": false,
    "confidence_level": "high|medium|low",
    "missing_elements": ["element1", "element2"],
    "strengths": ["strength1", "strength2"]
}"""),
            ("human", """QUERY: {query}
QUERY_TYPE: {query_type}
USER_EXPERTISE: {user_expertise}

RESPONSE TO EVALUATE:
{response}

CONTEXT PROVIDED:
{context}

Evaluate this response quality:""")
        ])
    
    def _create_improver(self):
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert response improvement specialist. Given a response and quality feedback, create a significantly improved version.

IMPROVEMENT STRATEGIES:
1. Address specific quality issues mentioned in feedback
2. Enhance clarity and structure
3. Add missing information from context
4. Improve tone and appropriateness
5. Strengthen logical flow and coherence
6. Ensure completeness while maintaining conciseness

QUALITY STANDARDS:
- Use clear, engaging language
- Structure information logically
- Include specific examples when helpful
- Maintain appropriate technical level
- Address all aspects of the original query
- Provide actionable insights when applicable

Generate an improved response that addresses the identified issues while maintaining accuracy."""),
            ("human", """ORIGINAL QUERY: {query}
ORIGINAL RESPONSE: {response}

QUALITY ASSESSMENT:
{quality_feedback}

ADDITIONAL CONTEXT:
{context}

USER PROFILE: {user_profile}

Create an improved response:""")
        ])
    
    def _create_fact_checker(self):
        return ChatPromptTemplate.from_messages([
            ("system", """You are a fact-checking expert. Verify the accuracy of claims in the response against the provided context.

Check for:
1. Factual accuracy - Are claims supported by context?
2. Consistency - Are there contradictions?
3. Completeness - Are important facts omitted?
4. Currency - Is information up-to-date based on context?
5. Attribution - Are claims properly sourced?

Return JSON:
{
    "accuracy_score": 4.5,
    "verified_facts": ["fact1", "fact2"],
    "questionable_claims": ["claim1", "claim2"],
    "missing_important_facts": ["fact1", "fact2"],
    "recommendations": ["recommendation1", "recommendation2"]
}"""),
            ("human", """RESPONSE TO CHECK:
{response}

SOURCE CONTEXT:
{context}

QUERY: {query}

Verify factual accuracy:""")
        ])
    
    def _create_coherence_checker(self):
        return ChatPromptTemplate.from_messages([
            ("system", """You are a coherence and clarity expert. Evaluate the logical flow and readability of the response.

Assess:
1. Logical structure and organization
2. Smooth transitions between ideas
3. Clear and engaging language
4. Appropriate level of detail
5. Consistent tone and style

Return JSON:
{
    "coherence_score": 4.2,
    "structure_quality": "excellent|good|fair|poor",
    "readability_level": "appropriate|too_simple|too_complex",
    "flow_issues": ["issue1", "issue2"],
    "clarity_improvements": ["improvement1", "improvement2"]
}"""),
            ("human", """RESPONSE TO ANALYZE:
{response}

QUERY TYPE: {query_type}
USER LEVEL: {user_level}

Evaluate coherence and clarity:""")
        ])
    
    async def validate_response_quality(self, 
                                      response: str,
                                      query: str,
                                      query_type: str,
                                      context: str,
                                      user_expertise: str = "intermediate") -> QualityMetrics:
        """Comprehensive response quality validation."""
        try:
            # Run quality assessments in parallel
            validation_task = self._run_validation(response, query, query_type, context, user_expertise)
            fact_check_task = self._run_fact_check(response, query, context)
            coherence_task = self._run_coherence_check(response, query_type, user_expertise)
            
            validation_result, fact_result, coherence_result = await asyncio.gather(
                validation_task, fact_check_task, coherence_task
            )
            
            # Combine results
            combined_metrics = self._combine_quality_metrics(
                validation_result, fact_result, coherence_result
            )
            
            # Store for learning
            self.quality_history.append(combined_metrics)
            
            return combined_metrics
            
        except Exception as e:
            self.logger.error(f"Error in quality validation: {e}")
            return self._fallback_quality_assessment(response, query)
    
    async def _run_validation(self, response: str, query: str, query_type: str, context: str, user_expertise: str) -> Dict[str, Any]:
        """Run primary quality validation."""
        try:
            result = await self.validator.ainvoke({
                "response": response,
                "query": query,
                "query_type": query_type,
                "context": context[:2000],  # Limit context length
                "user_expertise": user_expertise
            })
            return json.loads(result.content)
        except Exception:
            return self._basic_validation(response, query)
    
    async def _run_fact_check(self, response: str, query: str, context: str) -> Dict[str, Any]:
        """Run fact-checking validation."""
        try:
            result = await self.fact_checker.ainvoke({
                "response": response,
                "query": query,
                "context": context[:2000]
            })
            return json.loads(result.content)
        except Exception:
            return {"accuracy_score": 3.5, "verified_facts": [], "questionable_claims": []}
    
    async def _run_coherence_check(self, response: str, query_type: str, user_level: str) -> Dict[str, Any]:
        """Run coherence and clarity check."""
        try:
            result = await self.coherence_checker.ainvoke({
                "response": response,
                "query_type": query_type,
                "user_level": user_level
            })
            return json.loads(result.content)
        except Exception:
            return {"coherence_score": 3.5, "structure_quality": "fair", "readability_level": "appropriate"}
    
    def _combine_quality_metrics(self, 
                               validation_result: Dict[str, Any],
                               fact_result: Dict[str, Any],
                               coherence_result: Dict[str, Any]) -> QualityMetrics:
        """Combine multiple quality assessments into final metrics."""
        
        # Adjust accuracy based on fact-checking
        accuracy = validation_result.get("accuracy", 3.0)
        fact_score = fact_result.get("accuracy_score", 3.0)
        combined_accuracy = (accuracy + fact_score) / 2
        
        # Adjust clarity based on coherence check
        clarity = validation_result.get("clarity", 3.0)
        coherence_score = coherence_result.get("coherence_score", 3.0)
        combined_clarity = (clarity + coherence_score) / 2
        
        # Calculate overall score
        completeness = validation_result.get("completeness", 3.0)
        relevance = validation_result.get("relevance", 3.0)
        appropriateness = validation_result.get("appropriateness", 3.0)
        
        overall_score = (
            combined_accuracy * 0.25 +
            completeness * 0.25 +
            relevance * 0.25 +
            combined_clarity * 0.15 +
            appropriateness * 0.10
        )
        
        # Combine suggestions
        all_suggestions = (
            validation_result.get("suggestions", []) +
            fact_result.get("recommendations", []) +
            coherence_result.get("clarity_improvements", [])
        )
        
        # Determine confidence level
        confidence_level = validation_result.get("confidence_level", "medium")
        if overall_score >= 4.5:
            confidence_level = "high"
        elif overall_score <= 3.0:
            confidence_level = "low"
        
        return QualityMetrics(
            accuracy=combined_accuracy,
            completeness=completeness,
            relevance=relevance,
            clarity=combined_clarity,
            appropriateness=appropriateness,
            overall_score=overall_score,
            suggestions=all_suggestions[:5],  # Limit to top 5 suggestions
            requires_revision=overall_score < self.quality_threshold,
            confidence_level=confidence_level,
            timestamp=datetime.now()
        )
    
    async def improve_response(self, 
                             response: str,
                             quality_metrics: QualityMetrics,
                             query: str,
                             context: str,
                             user_profile: Dict[str, Any] = None) -> ResponseImprovement:
        """Improve response based on quality assessment."""
        
        if not quality_metrics.requires_revision:
            return ResponseImprovement(
                original_response=response,
                improved_response=response,
                improvement_type="no_improvement_needed",
                quality_gain=0.0,
                applied_suggestions=[]
            )
        
        try:
            # Prepare feedback for improvement
            quality_feedback = {
                "overall_score": quality_metrics.overall_score,
                "main_issues": quality_metrics.suggestions[:3],
                "accuracy": quality_metrics.accuracy,
                "completeness": quality_metrics.completeness,
                "clarity": quality_metrics.clarity,
                "requires_revision": quality_metrics.requires_revision
            }
            
            # Generate improved response
            result = await self.improver.ainvoke({
                "query": query,
                "response": response,
                "quality_feedback": json.dumps(quality_feedback),
                "context": context[:2000],
                "user_profile": json.dumps(user_profile or {})
            })
            
            improved_response = result.content.strip()
            
            # Determine improvement type
            improvement_type = self._classify_improvement_type(quality_metrics.suggestions)
            
            # Estimate quality gain (simplified)
            quality_gain = min(5.0 - quality_metrics.overall_score, 1.5)
            
            return ResponseImprovement(
                original_response=response,
                improved_response=improved_response,
                improvement_type=improvement_type,
                quality_gain=quality_gain,
                applied_suggestions=quality_metrics.suggestions[:3]
            )
            
        except Exception as e:
            self.logger.error(f"Error in response improvement: {e}")
            return ResponseImprovement(
                original_response=response,
                improved_response=response,
                improvement_type="error",
                quality_gain=0.0,
                applied_suggestions=[]
            )
    
    def _classify_improvement_type(self, suggestions: List[str]) -> str:
        """Classify the type of improvement needed."""
        suggestion_text = " ".join(suggestions).lower()
        
        if "accuracy" in suggestion_text or "fact" in suggestion_text:
            return "accuracy_improvement"
        elif "complete" in suggestion_text or "missing" in suggestion_text:
            return "completeness_improvement"
        elif "clarity" in suggestion_text or "structure" in suggestion_text:
            return "clarity_improvement"
        elif "tone" in suggestion_text or "appropriate" in suggestion_text:
            return "tone_improvement"
        else:
            return "general_improvement"
    
    def _basic_validation(self, response: str, query: str) -> Dict[str, Any]:
        """Basic rule-based validation fallback."""
        # Simple heuristics for basic quality assessment
        response_length = len(response)
        query_length = len(query)
        
        # Basic completeness check
        completeness = min(5.0, (response_length / max(query_length * 3, 100)) * 4)
        
        # Basic relevance check (keyword overlap)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words & response_words)
        relevance = min(5.0, (overlap / max(len(query_words), 1)) * 5)
        
        # Basic structure check
        has_structure = any(marker in response for marker in [".", ":", "-", "\n"])
        clarity = 4.0 if has_structure else 3.0
        
        overall = (completeness + relevance + clarity) / 3
        
        return {
            "accuracy": 3.5,
            "completeness": completeness,
            "relevance": relevance,
            "clarity": clarity,
            "appropriateness": 3.5,
            "overall_score": overall,
            "suggestions": ["Consider adding more specific details", "Improve response structure"],
            "requires_revision": overall < self.quality_threshold,
            "confidence_level": "medium"
        }
    
    def _fallback_quality_assessment(self, response: str, query: str) -> QualityMetrics:
        """Fallback quality assessment when LLM validation fails."""
        basic_result = self._basic_validation(response, query)
        
        return QualityMetrics(
            accuracy=basic_result["accuracy"],
            completeness=basic_result["completeness"],
            relevance=basic_result["relevance"],
            clarity=basic_result["clarity"],
            appropriateness=basic_result["appropriateness"],
            overall_score=basic_result["overall_score"],
            suggestions=basic_result["suggestions"],
            requires_revision=basic_result["requires_revision"],
            confidence_level=basic_result["confidence_level"],
            timestamp=datetime.now()
        )
    
    async def auto_improve_if_needed(self, 
                                   response: str,
                                   query: str,
                                   query_type: str,
                                   context: str,
                                   user_profile: Dict[str, Any] = None,
                                   max_iterations: int = 2) -> Tuple[str, QualityMetrics]:
        """Automatically improve response if quality is below threshold."""
        
        current_response = response
        iterations = 0
        
        while iterations < max_iterations:
            # Validate current response
            quality_metrics = await self.validate_response_quality(
                current_response, query, query_type, context, 
                user_profile.get("expertise_level", "intermediate") if user_profile else "intermediate"
            )
            
            # If quality is sufficient, return
            if not quality_metrics.requires_revision:
                return current_response, quality_metrics
            
            # Attempt improvement
            improvement = await self.improve_response(
                current_response, quality_metrics, query, context, user_profile
            )
            
            current_response = improvement.improved_response
            iterations += 1
            
            self.logger.info(f"Response improved (iteration {iterations}): {improvement.improvement_type}")
        
        # Final validation
        final_quality = await self.validate_response_quality(
            current_response, query, query_type, context,
            user_profile.get("expertise_level", "intermediate") if user_profile else "intermediate"
        )
        
        return current_response, final_quality
    
    def get_quality_statistics(self) -> Dict[str, Any]:
        """Get quality statistics from history."""
        if not self.quality_history:
            return {}
        
        recent_history = self.quality_history[-100:]  # Last 100 assessments
        
        return {
            "total_assessments": len(self.quality_history),
            "average_quality": sum(m.overall_score for m in recent_history) / len(recent_history),
            "revision_rate": sum(1 for m in recent_history if m.requires_revision) / len(recent_history),
            "quality_distribution": {
                "excellent": sum(1 for m in recent_history if m.overall_score >= 4.5),
                "good": sum(1 for m in recent_history if 3.5 <= m.overall_score < 4.5),
                "fair": sum(1 for m in recent_history if 2.5 <= m.overall_score < 3.5),
                "poor": sum(1 for m in recent_history if m.overall_score < 2.5)
            },
            "common_suggestions": self._get_common_suggestions(recent_history)
        }
    
    def _get_common_suggestions(self, history: List[QualityMetrics]) -> List[str]:
        """Get most common improvement suggestions."""
        all_suggestions = []
        for metrics in history:
            all_suggestions.extend(metrics.suggestions)
        
        # Count suggestion frequencies
        suggestion_counts = {}
        for suggestion in all_suggestions:
            suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1
        
        # Return top 5 most common suggestions
        return sorted(suggestion_counts.keys(), key=suggestion_counts.get, reverse=True)[:5]
    
    def update_quality_threshold(self, new_threshold: float) -> None:
        """Update quality threshold based on system performance."""
        if 1.0 <= new_threshold <= 5.0:
            self.quality_threshold = new_threshold
            self.logger.info(f"Quality threshold updated to {new_threshold}")
    
    def clear_history(self) -> None:
        """Clear quality assessment history."""
        self.quality_history = []
        self.logger.info("Quality assessment history cleared")
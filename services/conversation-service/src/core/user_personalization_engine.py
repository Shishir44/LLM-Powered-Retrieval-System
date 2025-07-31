"""
PHASE 3.4: User Personalization Engine
Context-aware responses based on user history, preferences, and behavioral patterns
Provides personalized RAG experiences while maintaining privacy and accuracy
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
import hashlib
import os

@dataclass
class UserProfile:
    """Comprehensive user profile for personalization."""
    user_id: str
    created_at: datetime
    
    # Preferences
    communication_style: str = "balanced"  # concise, detailed, balanced, technical
    expertise_level: str = "intermediate"  # beginner, intermediate, advanced, expert
    preferred_response_format: str = "structured"  # bullet_points, structured, narrative, step_by_step
    language_preference: str = "en"
    
    # Behavioral patterns
    typical_query_domains: List[str] = field(default_factory=list)  # billing, technical, product, etc.
    query_complexity_preference: str = "moderate"  # simple, moderate, complex
    interaction_frequency: str = "regular"  # occasional, regular, frequent, power_user
    
    # Historical insights
    successful_interaction_patterns: Dict[str, float] = field(default_factory=dict)
    preferred_information_sources: List[str] = field(default_factory=list)
    common_follow_up_patterns: List[str] = field(default_factory=list)
    
    # Context and session data
    current_session_context: Dict[str, Any] = field(default_factory=dict)
    recent_interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Privacy settings
    personalization_enabled: bool = True
    data_retention_days: int = 90
    anonymize_sensitive_data: bool = True
    
    # Derived insights
    confidence_score: float = 0.5  # Confidence in personalization accuracy
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class PersonalizationContext:
    """Context for personalizing a specific interaction."""
    user_profile: UserProfile
    current_query: str
    session_history: List[Dict[str, Any]]
    
    # Personalization recommendations
    recommended_communication_style: str = "balanced"
    recommended_detail_level: str = "moderate"
    recommended_sources: List[str] = field(default_factory=list)
    context_relevance_boost: float = 1.0
    
    # Predicted user needs
    likely_follow_up_questions: List[str] = field(default_factory=list)
    anticipated_information_gaps: List[str] = field(default_factory=list)
    suggested_proactive_information: List[str] = field(default_factory=list)

class CommunicationStyle(Enum):
    """Communication style preferences."""
    CONCISE = "concise"
    DETAILED = "detailed"
    BALANCED = "balanced"
    TECHNICAL = "technical"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"

class ExpertiseLevel(Enum):
    """User expertise level classification."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class UserPersonalizationEngine:
    """PHASE 3.4: User personalization engine for context-aware RAG responses."""
    
    def __init__(self, 
                 storage_path: str = "data/user_personalization",
                 learning_rate: float = 0.15,
                 privacy_mode: bool = True):
        
        self.storage_path = storage_path
        self.learning_rate = learning_rate
        self.privacy_mode = privacy_mode
        self.logger = logging.getLogger(__name__)
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        
        # User profiles and personalization data
        self.user_profiles: Dict[str, UserProfile] = {}
        self.interaction_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.personalization_patterns: Dict[str, Dict[str, Any]] = {}
        
        # Personalization models and insights
        self.communication_style_models: Dict[str, Dict[str, float]] = {}
        self.domain_expertise_models: Dict[str, Dict[str, float]] = {}
        self.context_relevance_models: Dict[str, Dict[str, float]] = {}
        
        # Privacy and security
        self.anonymization_salt = self._generate_salt()
        self.sensitive_data_patterns = [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'  # Phone
        ]
        
        # Performance tracking
        self.personalization_stats = {
            "total_users": 0,
            "active_profiles": 0,
            "personalization_improvements": 0,
            "privacy_anonymizations": 0,
            "context_enhancements": 0
        }
        
        # Load existing data
        self._load_personalization_data()
        
        self.logger.info("PHASE 3.4: User Personalization Engine initialized")

    async def get_or_create_user_profile(self, user_id: str, initial_context: Optional[Dict[str, Any]] = None) -> UserProfile:
        """Get existing user profile or create new one."""
        
        try:
            # Check if profile exists
            if user_id in self.user_profiles:
                profile = self.user_profiles[user_id]
                profile.last_updated = datetime.now()
                return profile
            
            # Create new profile
            profile = UserProfile(
                user_id=user_id,
                created_at=datetime.now()
            )
            
            # Initialize with context if provided
            if initial_context:
                await self._initialize_profile_from_context(profile, initial_context)
            
            self.user_profiles[user_id] = profile
            self.personalization_stats["total_users"] += 1
            
            self.logger.info(f"PHASE 3.4: Created new user profile for {user_id}")
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Error creating user profile: {e}")
            # Return default profile on error
            return UserProfile(user_id=user_id, created_at=datetime.now())

    async def personalize_query_processing(self, 
                                         user_id: str,
                                         query: str,
                                         session_context: Optional[Dict[str, Any]] = None) -> PersonalizationContext:
        """Generate personalization context for query processing."""
        
        try:
            self.logger.info(f"PHASE 3.4: Personalizing query processing for user {user_id}")
            
            # Get user profile
            user_profile = await self.get_or_create_user_profile(user_id, session_context)
            
            # Analyze current query in user context
            query_analysis = await self._analyze_query_for_personalization(query, user_profile)
            
            # Generate personalization recommendations
            personalization_context = await self._generate_personalization_context(
                user_profile, query, query_analysis, session_context
            )
            
            # Update user profile with current interaction
            await self._update_profile_with_interaction(user_profile, query, personalization_context)
            
            return personalization_context
            
        except Exception as e:
            self.logger.error(f"Error in personalized query processing: {e}")
            # Return basic context on error
            return PersonalizationContext(
                user_profile=UserProfile(user_id=user_id, created_at=datetime.now()),
                current_query=query,
                session_history=[]
            )

    async def personalize_response(self, 
                                 user_id: str,
                                 base_response: str,
                                 personalization_context: PersonalizationContext,
                                 sources: List[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Personalize response based on user preferences and context."""
        
        try:
            self.logger.info(f"PHASE 3.4: Personalizing response for user {user_id}")
            
            user_profile = personalization_context.user_profile
            
            # Apply communication style personalization
            styled_response = await self._apply_communication_style(
                base_response, user_profile.communication_style, user_profile.expertise_level
            )
            
            # Add personalized context and insights
            enhanced_response = await self._add_personalized_context(
                styled_response, personalization_context, sources
            )
            
            # Generate personalized metadata
            personalization_metadata = {
                "communication_style_applied": user_profile.communication_style,
                "expertise_level": user_profile.expertise_level,
                "personalization_confidence": user_profile.confidence_score,
                "context_enhancements": len(personalization_context.suggested_proactive_information),
                "anticipated_follow_ups": personalization_context.likely_follow_up_questions,
                "user_domain_match": self._calculate_domain_match(personalization_context)
            }
            
            self.personalization_stats["context_enhancements"] += 1
            
            return enhanced_response, personalization_metadata
            
        except Exception as e:
            self.logger.error(f"Error personalizing response: {e}")
            return base_response, {"personalization_error": str(e)}

    async def record_interaction_feedback(self, 
                                        user_id: str,
                                        query: str,
                                        response: str,
                                        user_satisfaction: float,
                                        follow_up_occurred: bool = False,
                                        resolution_achieved: bool = False) -> Dict[str, Any]:
        """Record interaction feedback for personalization learning."""
        
        try:
            # Get user profile
            user_profile = self.user_profiles.get(user_id)
            if not user_profile:
                return {"feedback_recorded": False, "reason": "user_profile_not_found"}
            
            # Create interaction record
            interaction_record = {
                "timestamp": datetime.now().isoformat(),
                "query": self._anonymize_if_needed(query),
                "response_length": len(response),
                "user_satisfaction": user_satisfaction,
                "follow_up_occurred": follow_up_occurred,
                "resolution_achieved": resolution_achieved,
                "query_domain": self._classify_query_domain(query),
                "communication_style_used": user_profile.communication_style
            }
            
            # Store interaction
            self.interaction_history[user_id].append(interaction_record)
            
            # Learn from feedback
            learning_insights = await self._learn_from_interaction_feedback(
                user_profile, interaction_record
            )
            
            # Update personalization models
            await self._update_personalization_models(user_id, interaction_record, learning_insights)
            
            # Save updated data
            await self._save_personalization_data()
            
            return {
                "feedback_recorded": True,
                "learning_insights": learning_insights,
                "profile_updates": len(learning_insights.get("profile_updates", [])),
                "model_improvements": learning_insights.get("model_improvements", 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error recording interaction feedback: {e}")
            return {"feedback_recorded": False, "error": str(e)}

    async def _analyze_query_for_personalization(self, 
                                               query: str, 
                                               user_profile: UserProfile) -> Dict[str, Any]:
        """Analyze query in context of user personalization."""
        
        analysis = {
            "query_domain": self._classify_query_domain(query),
            "complexity_level": self._assess_query_complexity(query),
            "user_domain_familiarity": 0.5,
            "communication_style_match": 0.5,
            "historical_pattern_match": 0.0
        }
        
        # Check domain familiarity
        query_domain = analysis["query_domain"]
        if query_domain in user_profile.typical_query_domains:
            domain_frequency = user_profile.typical_query_domains.count(query_domain)
            analysis["user_domain_familiarity"] = min(1.0, domain_frequency / 10.0)
        
        # Check historical patterns
        if user_profile.recent_interaction_history:
            recent_domains = [
                interaction.get("query_domain", "general") 
                for interaction in user_profile.recent_interaction_history[-5:]
            ]
            if query_domain in recent_domains:
                analysis["historical_pattern_match"] = 0.8
        
        # Assess communication style alignment
        if query.count('?') > 1 or len(query.split()) > 20:
            if user_profile.communication_style in ["detailed", "technical"]:
                analysis["communication_style_match"] = 0.9
        elif len(query.split()) < 10:
            if user_profile.communication_style in ["concise", "balanced"]:
                analysis["communication_style_match"] = 0.9
        
        return analysis

    async def _generate_personalization_context(self, 
                                               user_profile: UserProfile,
                                               query: str,
                                               query_analysis: Dict[str, Any],
                                               session_context: Optional[Dict[str, Any]]) -> PersonalizationContext:
        """Generate comprehensive personalization context."""
        
        context = PersonalizationContext(
            user_profile=user_profile,
            current_query=query,
            session_history=user_profile.recent_interaction_history[-10:]  # Last 10 interactions
        )
        
        # Determine optimal communication style
        context.recommended_communication_style = await self._recommend_communication_style(
            user_profile, query_analysis
        )
        
        # Determine detail level
        context.recommended_detail_level = await self._recommend_detail_level(
            user_profile, query_analysis
        )
        
        # Recommend sources based on user preferences
        context.recommended_sources = user_profile.preferred_information_sources[:5]
        
        # Calculate context relevance boost
        domain_familiarity = query_analysis.get("user_domain_familiarity", 0.5)
        pattern_match = query_analysis.get("historical_pattern_match", 0.0)
        context.context_relevance_boost = 1.0 + (domain_familiarity * 0.3) + (pattern_match * 0.2)
        
        # Predict likely follow-up questions
        context.likely_follow_up_questions = await self._predict_follow_up_questions(
            user_profile, query, query_analysis
        )
        
        # Identify potential information gaps
        context.anticipated_information_gaps = await self._identify_information_gaps(
            user_profile, query, query_analysis
        )
        
        # Suggest proactive information
        context.suggested_proactive_information = await self._suggest_proactive_information(
            user_profile, query, query_analysis
        )
        
        return context

    async def _apply_communication_style(self, 
                                       response: str, 
                                       style: str, 
                                       expertise_level: str) -> str:
        """Apply communication style personalization to response."""
        
        # Communication style transformations
        if style == "concise":
            # Make response more concise
            sentences = response.split('.')
            key_sentences = [s for s in sentences if len(s.strip()) > 20][:3]
            return '. '.join(key_sentences) + '.'
            
        elif style == "detailed":
            # Add more detailed explanations
            if len(response) < 300:
                return response + "\n\nFor additional context: This information is based on comprehensive analysis of relevant documentation and policies."
            return response
            
        elif style == "technical":
            # Add technical precision
            technical_prefix = "Technical Analysis: "
            return technical_prefix + response
            
        elif style == "friendly":
            # Make more conversational
            friendly_prefix = "I'd be happy to help! "
            return friendly_prefix + response
            
        elif style == "professional":
            # Maintain professional tone
            if not response.startswith("Based on"):
                return "Based on available information: " + response
            return response
        
        # Default: balanced style
        return response

    async def _add_personalized_context(self, 
                                       response: str, 
                                       personalization_context: PersonalizationContext,
                                       sources: List[str] = None) -> str:
        """Add personalized context and insights to response."""
        
        enhanced_response = response
        user_profile = personalization_context.user_profile
        
        # Add proactive information if relevant
        if personalization_context.suggested_proactive_information:
            proactive_info = personalization_context.suggested_proactive_information[0]
            enhanced_response += f"\n\n**Additional Context**: {proactive_info}"
        
        # Add anticipated follow-up guidance
        if personalization_context.likely_follow_up_questions:
            follow_up = personalization_context.likely_follow_up_questions[0]
            enhanced_response += f"\n\n*You might also want to know*: {follow_up}"
        
        # Add personalized source recommendations
        if user_profile.preferred_information_sources and sources:
            preferred_sources = [s for s in sources if s in user_profile.preferred_information_sources]
            if preferred_sources:
                enhanced_response += f"\n\n*Recommended sources for you*: {', '.join(preferred_sources[:3])}"
        
        # Add expertise-level appropriate guidance
        if user_profile.expertise_level == "beginner":
            enhanced_response += "\n\n*Need more help?* Feel free to ask for clarification on any part of this response."
        elif user_profile.expertise_level == "expert":
            enhanced_response += "\n\n*For advanced details*, let me know if you need deeper technical insights."
        
        return enhanced_response

    async def _recommend_communication_style(self, 
                                           user_profile: UserProfile, 
                                           query_analysis: Dict[str, Any]) -> str:
        """Recommend optimal communication style for user."""
        
        # Use historical success patterns
        if user_profile.successful_interaction_patterns:
            best_style = max(
                user_profile.successful_interaction_patterns.items(),
                key=lambda x: x[1]
            )[0]
            if best_style in ["concise", "detailed", "balanced", "technical"]:
                return best_style
        
        # Fallback to user preference
        return user_profile.communication_style

    async def _recommend_detail_level(self, 
                                     user_profile: UserProfile, 
                                     query_analysis: Dict[str, Any]) -> str:
        """Recommend optimal detail level for response."""
        
        # Consider expertise level
        if user_profile.expertise_level == "beginner":
            return "detailed"
        elif user_profile.expertise_level == "expert":
            return "concise"
        
        # Consider query complexity
        query_complexity = query_analysis.get("complexity_level", "moderate")
        if query_complexity == "complex":
            return "detailed"
        elif query_complexity == "simple":
            return "concise"
        
        return "moderate"

    async def _predict_follow_up_questions(self, 
                                         user_profile: UserProfile,
                                         query: str,
                                         query_analysis: Dict[str, Any]) -> List[str]:
        """Predict likely follow-up questions based on user patterns."""
        
        follow_ups = []
        query_domain = query_analysis.get("query_domain", "general")
        
        # Domain-specific follow-ups
        if query_domain == "billing":
            follow_ups.extend([
                "How can I update my payment method?",
                "When is my next billing cycle?",
                "How can I view my billing history?"
            ])
        elif query_domain == "technical":
            follow_ups.extend([
                "What are the troubleshooting steps?",
                "How can I prevent this issue in the future?",
                "Are there any known limitations?"
            ])
        elif query_domain == "product":
            follow_ups.extend([
                "What are the available alternatives?",
                "How does this compare to other options?",
                "What are the system requirements?"
            ])
        
        # User-specific patterns
        if user_profile.common_follow_up_patterns:
            follow_ups.extend(user_profile.common_follow_up_patterns[:2])
        
        return follow_ups[:3]

    async def _identify_information_gaps(self, 
                                       user_profile: UserProfile,
                                       query: str,
                                       query_analysis: Dict[str, Any]) -> List[str]:
        """Identify potential information gaps based on user context."""
        
        gaps = []
        
        # Expertise-based gaps
        if user_profile.expertise_level == "beginner":
            gaps.extend([
                "Basic terminology and concepts",
                "Step-by-step guidance",
                "Common pitfalls to avoid"
            ])
        elif user_profile.expertise_level == "expert":
            gaps.extend([
                "Advanced configuration options",
                "Performance implications",
                "Integration considerations"
            ])
        
        # Domain-specific gaps
        query_domain = query_analysis.get("query_domain", "general")
        if query_domain not in user_profile.typical_query_domains:
            gaps.append("Domain-specific context and background")
        
        return gaps[:3]

    async def _suggest_proactive_information(self, 
                                           user_profile: UserProfile,
                                           query: str,
                                           query_analysis: Dict[str, Any]) -> List[str]:
        """Suggest proactive information based on user patterns."""
        
        suggestions = []
        
        # Based on recent interaction patterns
        if user_profile.recent_interaction_history:
            recent_domains = [
                interaction.get("query_domain") 
                for interaction in user_profile.recent_interaction_history[-3:]
            ]
            
            current_domain = query_analysis.get("query_domain")
            if recent_domains.count(current_domain) >= 2:
                suggestions.append(f"Since you're working on {current_domain} topics, you might find our comprehensive {current_domain} guide helpful.")
        
        # Based on expertise level
        if user_profile.expertise_level == "beginner":
            suggestions.append("Consider reviewing our getting started guide for additional context.")
        
        # Based on communication style
        if user_profile.communication_style == "technical":
            suggestions.append("Technical documentation and API references are available for deeper implementation details.")
        
        return suggestions[:2]

    def _classify_query_domain(self, query: str) -> str:
        """Classify query into domain categories."""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["bill", "payment", "invoice", "charge", "cost", "price"]):
            return "billing"
        elif any(word in query_lower for word in ["error", "bug", "issue", "problem", "troubleshoot", "fix"]):
            return "technical"
        elif any(word in query_lower for word in ["product", "feature", "specification", "catalog", "model"]):
            return "product"
        elif any(word in query_lower for word in ["account", "login", "password", "profile", "settings"]):
            return "account"
        elif any(word in query_lower for word in ["policy", "terms", "condition", "privacy", "agreement"]):
            return "policy"
        else:
            return "general"

    def _assess_query_complexity(self, query: str) -> str:
        """Assess query complexity level."""
        
        word_count = len(query.split())
        question_count = query.count('?')
        
        if word_count > 25 or question_count > 2:
            return "complex"
        elif word_count > 10 or question_count > 1:
            return "moderate"
        else:
            return "simple"

    def _calculate_domain_match(self, personalization_context: PersonalizationContext) -> float:
        """Calculate how well the query matches user's typical domains."""
        
        user_domains = personalization_context.user_profile.typical_query_domains
        current_domain = self._classify_query_domain(personalization_context.current_query)
        
        if not user_domains:
            return 0.5
        
        domain_frequency = user_domains.count(current_domain)
        return min(1.0, domain_frequency / len(user_domains))

    async def _learn_from_interaction_feedback(self, 
                                             user_profile: UserProfile,
                                             interaction_record: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from interaction feedback to improve personalization."""
        
        insights = {
            "profile_updates": [],
            "model_improvements": 0,
            "pattern_discoveries": []
        }
        
        satisfaction = interaction_record["user_satisfaction"]
        communication_style = interaction_record["communication_style_used"]
        query_domain = interaction_record["query_domain"]
        
        # Update successful interaction patterns
        if satisfaction >= 4.0:
            current_score = user_profile.successful_interaction_patterns.get(communication_style, 0.0)
            new_score = (current_score * 0.8) + (satisfaction / 5.0 * 0.2)
            user_profile.successful_interaction_patterns[communication_style] = new_score
            insights["profile_updates"].append(f"Improved {communication_style} style score")
        
        # Update domain preferences
        if satisfaction >= 3.5:
            if query_domain not in user_profile.typical_query_domains:
                user_profile.typical_query_domains.append(query_domain)
            insights["profile_updates"].append(f"Added {query_domain} to preferred domains")
        
        # Update follow-up patterns
        if interaction_record.get("follow_up_occurred"):
            pattern = f"follow_up_after_{query_domain}"
            if pattern not in user_profile.common_follow_up_patterns:
                user_profile.common_follow_up_patterns.append(pattern)
            insights["pattern_discoveries"].append(pattern)
        
        # Update confidence score
        interaction_count = len(self.interaction_history.get(user_profile.user_id, []))
        if interaction_count > 0:
            avg_satisfaction = sum(
                interaction.get("user_satisfaction", 3.0) 
                for interaction in self.interaction_history[user_profile.user_id]
            ) / interaction_count
            
            user_profile.confidence_score = min(0.95, avg_satisfaction / 5.0)
        
        insights["model_improvements"] = len(insights["profile_updates"])
        
        return insights

    async def _update_personalization_models(self, 
                                           user_id: str,
                                           interaction_record: Dict[str, Any],
                                           learning_insights: Dict[str, Any]) -> None:
        """Update personalization models based on learning insights."""
        
        # Update communication style models
        communication_style = interaction_record["communication_style_used"]
        satisfaction = interaction_record["user_satisfaction"]
        
        if user_id not in self.communication_style_models:
            self.communication_style_models[user_id] = {}
        
        current_score = self.communication_style_models[user_id].get(communication_style, 0.5)
        new_score = (current_score * (1 - self.learning_rate)) + (satisfaction / 5.0 * self.learning_rate)
        self.communication_style_models[user_id][communication_style] = new_score
        
        # Update domain expertise models
        query_domain = interaction_record["query_domain"]
        resolution_achieved = interaction_record.get("resolution_achieved", False)
        
        if user_id not in self.domain_expertise_models:
            self.domain_expertise_models[user_id] = {}
        
        if resolution_achieved:
            current_expertise = self.domain_expertise_models[user_id].get(query_domain, 0.5)
            self.domain_expertise_models[user_id][query_domain] = min(1.0, current_expertise + 0.1)

    async def _initialize_profile_from_context(self, 
                                             profile: UserProfile, 
                                             context: Dict[str, Any]) -> None:
        """Initialize user profile from initial context."""
        
        # Infer communication style from context
        if context.get("preferred_style"):
            profile.communication_style = context["preferred_style"]
        
        # Infer expertise level
        if context.get("technical_level"):
            profile.expertise_level = context["technical_level"]
        
        # Set privacy preferences
        if context.get("privacy_settings"):
            privacy_settings = context["privacy_settings"]
            profile.personalization_enabled = privacy_settings.get("enable_personalization", True)
            profile.anonymize_sensitive_data = privacy_settings.get("anonymize_data", True)

    def _anonymize_if_needed(self, text: str) -> str:
        """Anonymize sensitive data if privacy mode is enabled."""
        
        if not self.privacy_mode:
            return text
        
        anonymized_text = text
        
        # Apply anonymization patterns
        import re
        for pattern in self.sensitive_data_patterns:
            anonymized_text = re.sub(pattern, "[REDACTED]", anonymized_text)
        
        if anonymized_text != text:
            self.personalization_stats["privacy_anonymizations"] += 1
        
        return anonymized_text

    def _generate_salt(self) -> str:
        """Generate salt for anonymization."""
        import secrets
        return secrets.token_hex(16)

    async def _save_personalization_data(self) -> None:
        """Save personalization data to persistent storage."""
        
        try:
            # Save user profiles
            profiles_file = os.path.join(self.storage_path, "user_profiles.json")
            profiles_data = {}
            for user_id, profile in self.user_profiles.items():
                profiles_data[user_id] = {
                    "user_id": profile.user_id,
                    "created_at": profile.created_at.isoformat(),
                    "communication_style": profile.communication_style,
                    "expertise_level": profile.expertise_level,
                    "preferred_response_format": profile.preferred_response_format,
                    "typical_query_domains": profile.typical_query_domains,
                    "successful_interaction_patterns": profile.successful_interaction_patterns,
                    "common_follow_up_patterns": profile.common_follow_up_patterns,
                    "confidence_score": profile.confidence_score,
                    "last_updated": profile.last_updated.isoformat()
                }
            
            with open(profiles_file, "w") as f:
                json.dump(profiles_data, f, indent=2)
            
            # Save models
            models_file = os.path.join(self.storage_path, "personalization_models.json")
            models_data = {
                "communication_style_models": self.communication_style_models,
                "domain_expertise_models": self.domain_expertise_models,
                "personalization_stats": self.personalization_stats
            }
            
            with open(models_file, "w") as f:
                json.dump(models_data, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error saving personalization data: {e}")

    def _load_personalization_data(self) -> None:
        """Load existing personalization data from storage."""
        
        try:
            # Load user profiles
            profiles_file = os.path.join(self.storage_path, "user_profiles.json")
            if os.path.exists(profiles_file):
                with open(profiles_file, "r") as f:
                    profiles_data = json.load(f)
                
                for user_id, profile_data in profiles_data.items():
                    profile = UserProfile(
                        user_id=profile_data["user_id"],
                        created_at=datetime.fromisoformat(profile_data["created_at"]),
                        communication_style=profile_data.get("communication_style", "balanced"),
                        expertise_level=profile_data.get("expertise_level", "intermediate"),
                        preferred_response_format=profile_data.get("preferred_response_format", "structured"),
                        typical_query_domains=profile_data.get("typical_query_domains", []),
                        successful_interaction_patterns=profile_data.get("successful_interaction_patterns", {}),
                        common_follow_up_patterns=profile_data.get("common_follow_up_patterns", []),
                        confidence_score=profile_data.get("confidence_score", 0.5),
                        last_updated=datetime.fromisoformat(profile_data["last_updated"])
                    )
                    self.user_profiles[user_id] = profile
            
            # Load models
            models_file = os.path.join(self.storage_path, "personalization_models.json")
            if os.path.exists(models_file):
                with open(models_file, "r") as f:
                    models_data = json.load(f)
                
                self.communication_style_models = models_data.get("communication_style_models", {})
                self.domain_expertise_models = models_data.get("domain_expertise_models", {})
                self.personalization_stats.update(models_data.get("personalization_stats", {}))
            
            self.logger.info(f"Loaded {len(self.user_profiles)} user profiles and personalization models")
            
        except Exception as e:
            self.logger.error(f"Error loading personalization data: {e}")

    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get analytics for a specific user."""
        
        if user_id not in self.user_profiles:
            return {"error": "User profile not found"}
        
        profile = self.user_profiles[user_id]
        interactions = self.interaction_history.get(user_id, [])
        
        analytics = {
            "profile_summary": {
                "user_id": profile.user_id,
                "created_at": profile.created_at.isoformat(),
                "expertise_level": profile.expertise_level,
                "communication_style": profile.communication_style,
                "confidence_score": profile.confidence_score,
                "total_interactions": len(interactions)
            },
            "interaction_patterns": {
                "typical_domains": profile.typical_query_domains,
                "successful_patterns": profile.successful_interaction_patterns,
                "common_follow_ups": profile.common_follow_up_patterns
            },
            "performance_metrics": {},
            "personalization_effectiveness": {}
        }
        
        if interactions:
            # Calculate performance metrics
            avg_satisfaction = sum(i.get("user_satisfaction", 3.0) for i in interactions) / len(interactions)
            resolution_rate = sum(1 for i in interactions if i.get("resolution_achieved", False)) / len(interactions)
            
            analytics["performance_metrics"] = {
                "average_satisfaction": avg_satisfaction,
                "resolution_rate": resolution_rate,
                "total_interactions": len(interactions),
                "recent_activity": len([i for i in interactions if (datetime.now() - datetime.fromisoformat(i["timestamp"])).days <= 7])
            }
        
        return analytics

    async def get_system_analytics(self) -> Dict[str, Any]:
        """Get system-wide personalization analytics."""
        
        analytics = {
            "system_stats": self.personalization_stats.copy(),
            "user_distribution": {},
            "effectiveness_metrics": {},
            "privacy_compliance": {}
        }
        
        # User distribution
        total_users = len(self.user_profiles)
        if total_users > 0:
            expertise_distribution = {}
            style_distribution = {}
            
            for profile in self.user_profiles.values():
                expertise_distribution[profile.expertise_level] = expertise_distribution.get(profile.expertise_level, 0) + 1
                style_distribution[profile.communication_style] = style_distribution.get(profile.communication_style, 0) + 1
            
            analytics["user_distribution"] = {
                "total_users": total_users,
                "expertise_levels": expertise_distribution,
                "communication_styles": style_distribution
            }
        
        # Privacy compliance
        analytics["privacy_compliance"] = {
            "anonymizations_performed": self.personalization_stats.get("privacy_anonymizations", 0),
            "privacy_mode_enabled": self.privacy_mode,
            "data_retention_compliance": True  # Simplified for demo
        }
        
        return analytics

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on personalization engine."""
        
        try:
            return {
                "status": "healthy",
                "active_user_profiles": len(self.user_profiles),
                "personalization_models_loaded": len(self.communication_style_models) > 0,
                "storage_accessible": os.path.exists(self.storage_path),
                "privacy_mode_enabled": self.privacy_mode,
                "total_personalizations": self.personalization_stats.get("context_enhancements", 0),
                "system_learning_active": len(self.user_profiles) > 0,
                "last_checked": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            } 
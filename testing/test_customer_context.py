"""Test customer context integration functionality."""

import asyncio
import pytest
from services.conversation_service.src.core.customer_profile import CustomerProfileManager, CustomerTier, PreferredTone, ExpertiseLevel
from services.conversation_service.src.core.context_aware_generator import ContextAwareResponseGenerator
from services.shared.config import EnhancedRAGConfig

class TestCustomerContextIntegration:
    
    @pytest.fixture
    async def profile_manager(self):
        config = EnhancedRAGConfig()
        return CustomerProfileManager(config)
    
    @pytest.fixture
    async def context_generator(self, profile_manager):
        config = EnhancedRAGConfig()
        return ContextAwareResponseGenerator(config, profile_manager)
    
    async def test_customer_profile_creation(self, profile_manager):
        """Test customer profile creation and management."""
        
        customer_id = "test_customer_123"
        
        # Get initial profile (should create default)
        profile = await profile_manager.get_customer_profile(customer_id)
        
        assert profile.customer_id == customer_id
        assert profile.tier == CustomerTier.BASIC
        assert profile.interaction_history.total_conversations == 0
        
        # Update profile
        updates = {
            "tier": CustomerTier.PREMIUM,
            "preferred_tone": PreferredTone.TECHNICAL,
            "expertise_level": ExpertiseLevel.ADVANCED
        }
        
        updated_profile = await profile_manager.update_customer_profile(customer_id, updates)
        
        assert updated_profile.tier == CustomerTier.PREMIUM
        assert updated_profile.preferences.preferred_tone == PreferredTone.TECHNICAL
        assert updated_profile.preferences.expertise_level == ExpertiseLevel.ADVANCED
        
        print(f"✅ Customer profile created and updated successfully")
        print(f"📊 Customer tier: {updated_profile.tier.value}")
        print(f"🎯 Preferred tone: {updated_profile.preferences.preferred_tone.value}")

    async def test_interaction_recording(self, profile_manager):
        """Test interaction recording and history tracking."""
        
        customer_id = "test_customer_456"
        
        # Record multiple interactions
        interactions = [
            ("How do I deploy Docker containers?", "Here's how to deploy...", 4.5, "resolved", ["docker", "deployment"]),
            ("What's the difference between Docker and Kubernetes?", "Docker is...", 4.0, "resolved", ["docker", "kubernetes"]),
            ("I'm having trouble with container networking", "Let me help...", 3.5, "partially_resolved", ["docker", "networking"])
        ]
        
        for query, response, score, status, topics in interactions:
            await profile_manager.record_interaction(
                customer_id, query, response, score, status, topics
            )
        
        # Check updated profile
        profile = await profile_manager.get_customer_profile(customer_id)
        
        assert profile.interaction_history.total_conversations == 3
        assert profile.interaction_history.successful_resolutions == 2
        assert profile.interaction_history.average_satisfaction_score > 3.5
        assert "docker" in profile.interaction_history.common_topics
        
        print(f"✅ Interaction history recorded successfully")
        print(f"📈 Total conversations: {profile.interaction_history.total_conversations}")
        print(f"⭐ Average satisfaction: {profile.interaction_history.average_satisfaction_score:.2f}")
        print(f"🏷️ Common topics: {profile.interaction_history.common_topics}")

    async def test_contextual_response_generation(self, context_generator, profile_manager):
        """Test context-aware response generation."""
        
        customer_id = "test_customer_789"
        
        # Set up customer profile
        await profile_manager.update_customer_profile(customer_id, {
            "tier": CustomerTier.ENTERPRISE,
            "preferred_tone": PreferredTone.TECHNICAL,
            "expertise_level": ExpertiseLevel.ADVANCED,
            "include_code_examples": True
        })
        
        # Record some interaction history
        await profile_manager.record_interaction(
            customer_id, 
            "Previous Docker question", 
            "Previous answer", 
            4.5, 
            "resolved", 
            ["docker", "containers"]
        )
        
        # Generate contextual response
        query = "How do I optimize Docker container performance?"
        context = "Docker containers can be optimized through various techniques including multi-stage builds, layer caching, and resource limits."
        
        # Mock query analysis
        from services.conversation_service.src.models.query_analysis import QueryAnalysis, QueryType
        query_analysis = QueryAnalysis(
            query_type=QueryType.PROCEDURAL,
            complexity_score=0.7,
            intent="optimization_help"
        )
        
        response = await context_generator.generate_contextual_response(
            query, context, query_analysis, customer_id
        )
        
        assert response.content
        assert response.tone_used == "technical"
        assert len(response.personalization_applied) > 0
        assert response.customer_context["customer_tier"] == "enterprise"
        assert response.customer_context["expertise_level"] == "advanced"
        
        print(f"✅ Contextual response generated successfully")
        print(f"🎭 Tone used: {response.tone_used}")
        print(f"🎯 Personalizations: {response.personalization_applied}")
        print(f"⏱️ Generation time: {response.generation_time:.3f}s")
        print(f"📝 Response preview: {response.content[:100]}...")

if __name__ == "__main__":
    asyncio.run(TestCustomerContextIntegration().test_customer_profile_creation(None))
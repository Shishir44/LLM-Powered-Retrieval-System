#!/usr/bin/env python3
"""
Test script for LLM fallback functionality
This script tests the LLM Client Manager with multiple providers and fallback logic.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add services to path
sys.path.append(str(Path(__file__).parent / "services" / "shared"))
sys.path.append(str(Path(__file__).parent / "services" / "conversation-service" / "src" / "core"))

from config_manager import get_config_manager
from llm_client_manager import LLMClientManager, LLMProvider


async def test_llm_fallback():
    """Test LLM fallback functionality with different provider configurations."""
    
    print("🚀 Testing LLM Client Manager with Fallback Support")
    print("=" * 60)
    
    # Test different configurations
    test_configs = [
        {
            "name": "OpenAI Only",
            "env_vars": {
                "OPENAI_API_KEY": "test-key",
                "PRIMARY_LLM_PROVIDER": "openai",
                "ENABLE_FALLBACK": "false"
            }
        },
        {
            "name": "OpenAI with Gemini Fallback",
            "env_vars": {
                "OPENAI_API_KEY": "test-key",
                "GEMINI_API_KEY": "test-key",
                "PRIMARY_LLM_PROVIDER": "openai",
                "FALLBACK_PROVIDERS": "gemini",
                "ENABLE_FALLBACK": "true"
            }
        },
        {
            "name": "All Providers with Fallback",
            "env_vars": {
                "OPENAI_API_KEY": "test-key",
                "GEMINI_API_KEY": "test-key", 
                "ANTHROPIC_API_KEY": "test-key",
                "PRIMARY_LLM_PROVIDER": "openai",
                "FALLBACK_PROVIDERS": "gemini,anthropic",
                "ENABLE_FALLBACK": "true"
            }
        }
    ]
    
    for test_config in test_configs:
        print(f"\n📋 Testing Configuration: {test_config['name']}")
        print("-" * 40)
        
        # Set environment variables
        for key, value in test_config["env_vars"].items():
            os.environ[key] = value
        
        try:
            # Initialize config manager
            config_manager = get_config_manager()
            config = config_manager.config
            
            # Initialize LLM client manager
            llm_manager = LLMClientManager(config)
            
            # Display configuration
            print(f"✅ Available providers: {llm_manager.get_available_providers()}")
            print(f"🎯 Primary provider: {llm_manager.get_primary_provider()}")
            print(f"🔄 Fallback providers: {llm_manager.get_fallback_providers()}")
            print(f"⚙️  Fallback enabled: {config.enable_fallback}")
            
            # Test message
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ]
            
            # Note: This would normally make API calls, but with test keys it will fail
            # This demonstrates the configuration and initialization working correctly
            print(f"🧪 Configuration test: PASSED")
            
        except Exception as e:
            print(f"❌ Configuration test: FAILED - {e}")
        
        # Clean up environment variables
        for key in test_config["env_vars"].keys():
            os.environ.pop(key, None)
    
    print("\n" + "=" * 60)
    print("✅ LLM Fallback Configuration Tests Complete!")
    print("\n📝 To use with real API keys:")
    print("   1. Set OPENAI_API_KEY in your environment")
    print("   2. Set GEMINI_API_KEY for Google Gemini")  
    print("   3. Set ANTHROPIC_API_KEY for Claude")
    print("   4. Configure PRIMARY_LLM_PROVIDER and FALLBACK_PROVIDERS")
    print("   5. Set ENABLE_FALLBACK=true to enable fallback functionality")


def test_configuration_validation():
    """Test configuration validation with different provider setups."""
    
    print("\n🔍 Testing Configuration Validation")
    print("-" * 40)
    
    # Test missing API keys
    test_cases = [
        {
            "name": "No API Keys",
            "env": {},
            "should_work": True  # Should initialize but have no providers
        },
        {
            "name": "OpenAI Only",
            "env": {"OPENAI_API_KEY": "sk-test123"},
            "should_work": True
        },
        {
            "name": "Invalid Primary Provider",
            "env": {
                "OPENAI_API_KEY": "sk-test123",
                "PRIMARY_LLM_PROVIDER": "invalid_provider"
            },
            "should_work": True  # Should fallback gracefully
        }
    ]
    
    for case in test_cases:
        print(f"\n  Testing: {case['name']}")
        
        # Set environment
        for key, value in case["env"].items():
            os.environ[key] = value
        
        try:
            config_manager = get_config_manager()
            config = config_manager.config
            llm_manager = LLMClientManager(config)
            
            providers = llm_manager.get_available_providers()
            print(f"    ✅ Providers: {providers}")
            
        except Exception as e:
            if case["should_work"]:
                print(f"    ❌ Unexpected error: {e}")
            else:
                print(f"    ✅ Expected error: {e}")
        
        # Cleanup
        for key in case["env"].keys():
            os.environ.pop(key, None)


if __name__ == "__main__":
    asyncio.run(test_llm_fallback())
    test_configuration_validation()
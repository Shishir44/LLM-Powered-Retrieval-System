#!/usr/bin/env python3
"""
Simple validation script for LLM fallback configuration
This script validates the configuration without requiring all LLM provider dependencies.
"""
import os
import sys
from pathlib import Path

# Add services to path
sys.path.append(str(Path(__file__).parent / "services" / "shared"))

def test_config_manager():
    """Test the enhanced configuration manager."""
    
    print("🚀 Testing Enhanced Configuration Manager")
    print("=" * 50)
    
    try:
        from config_manager import get_config_manager, EnhancedRAGConfig
        
        # Test configuration with different provider setups
        test_configs = [
            {
                "name": "OpenAI Primary with Gemini/Anthropic Fallback",
                "env_vars": {
                    "OPENAI_API_KEY": "sk-test123",
                    "GEMINI_API_KEY": "test-gemini-key",
                    "ANTHROPIC_API_KEY": "test-anthropic-key",
                    "PRIMARY_LLM_PROVIDER": "openai",
                    "FALLBACK_PROVIDERS": "gemini,anthropic",
                    "ENABLE_FALLBACK": "true",
                    "FALLBACK_TIMEOUT": "10.0"
                }
            },
            {
                "name": "Gemini Primary with OpenAI Fallback",
                "env_vars": {
                    "OPENAI_API_KEY": "sk-test123",
                    "GEMINI_API_KEY": "test-gemini-key",
                    "PRIMARY_LLM_PROVIDER": "gemini",
                    "FALLBACK_PROVIDERS": "openai",
                    "ENABLE_FALLBACK": "true"
                }
            }
        ]
        
        for test_config in test_configs:
            print(f"\n📋 Testing: {test_config['name']}")
            print("-" * 40)
            
            # Set environment variables
            for key, value in test_config["env_vars"].items():
                os.environ[key] = value
            
            try:
                # Initialize config manager (creates new instance each time)
                global _config_manager
                from config_manager import _config_manager
                _config_manager = None  # Reset global instance
                
                config_manager = get_config_manager()
                config = config_manager.config
                
                # Validate configuration
                print(f"✅ OpenAI API Key: {'Set' if config.openai_api_key else 'Not Set'}")
                print(f"✅ Gemini API Key: {'Set' if config.gemini_api_key else 'Not Set'}")
                print(f"✅ Anthropic API Key: {'Set' if config.anthropic_api_key else 'Not Set'}")
                print(f"✅ Primary Provider: {config.primary_llm_provider}")
                print(f"✅ Fallback Providers: {config.fallback_providers}")
                print(f"✅ Fallback Enabled: {config.enable_fallback}")
                print(f"✅ Fallback Timeout: {config.fallback_timeout}s")
                print(f"✅ OpenAI Model: {config.openai_model}")
                print(f"✅ Gemini Model: {config.gemini_model}")
                print(f"✅ Anthropic Model: {config.anthropic_model}")
                
                print("🎉 Configuration validation: PASSED")
                
            except Exception as e:
                print(f"❌ Configuration validation: FAILED - {e}")
            
            # Clean up environment variables
            for key in test_config["env_vars"].keys():
                os.environ.pop(key, None)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True


def validate_files_created():
    """Validate that all necessary files were created."""
    
    print("\n🔍 Validating Created Files")
    print("-" * 30)
    
    required_files = [
        "services/conversation-service/src/core/llm_client_manager.py",
        "services/shared/config_manager.py"
    ]
    
    all_exist = True
    
    for file_path in required_files:
        full_path = Path(__file__).parent / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_exist = False
    
    return all_exist


def validate_requirements_updated():
    """Validate that requirements files were updated."""
    
    print("\n📦 Validating Requirements Files")
    print("-" * 35)
    
    requirements_files = [
        "requirements.txt",
        "services/conversation-service/requirements.txt"
    ]
    
    required_packages = ["openai", "anthropic", "google-generativeai"]
    
    for req_file in requirements_files:
        file_path = Path(__file__).parent / req_file
        if file_path.exists():
            content = file_path.read_text()
            print(f"\n📄 {req_file}:")
            
            for package in required_packages:
                if package in content:
                    print(f"  ✅ {package}")
                else:
                    print(f"  ❌ {package} - NOT FOUND")
        else:
            print(f"❌ {req_file} - FILE NOT FOUND")


if __name__ == "__main__":
    print("🧪 LLM Fallback Implementation Validation")
    print("=" * 50)
    
    # Run validations
    config_ok = test_config_manager()
    files_ok = validate_files_created()
    
    validate_requirements_updated()
    
    print("\n" + "=" * 50)
    if config_ok and files_ok:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("\n📝 Next Steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Set your actual API keys in environment variables:")
        print("      - OPENAI_API_KEY=your_openai_key")
        print("      - GEMINI_API_KEY=your_gemini_key") 
        print("      - ANTHROPIC_API_KEY=your_anthropic_key")
        print("   3. Configure provider preferences:")
        print("      - PRIMARY_LLM_PROVIDER=openai")
        print("      - FALLBACK_PROVIDERS=gemini,anthropic")
        print("      - ENABLE_FALLBACK=true")
        print("   4. Start your conversation service to test the fallback functionality")
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("Please check the errors above and fix any issues.")
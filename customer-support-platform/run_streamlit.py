#!/usr/bin/env python3
"""
Launch script for the Customer Support Knowledge Base Streamlit UI

This script starts the Streamlit interface for the LangChain-powered
customer support knowledge base system.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_services():
    """Check if the required services are running"""
    import requests
    
    services = {
        "Knowledge Base Service": "http://127.0.0.1:8001/health",
        "Conversation Service": "http://127.0.0.1:8002/health", 
        "API Gateway": "http://127.0.0.1:8000/health"
    }
    
    print("🔍 Checking service availability...")
    print("-" * 50)
    
    for service_name, health_url in services.items():
        try:
            response = requests.get(health_url, timeout=2)
            if response.status_code == 200:
                print(f"✅ {service_name}: Running")
            else:
                print(f"⚠️  {service_name}: Unhealthy (HTTP {response.status_code})")
        except requests.exceptions.RequestException:
            print(f"❌ {service_name}: Not accessible")
    
    print("-" * 50)

def main():
    """Launch the Streamlit application"""
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("🚀 Starting Customer Support Knowledge Base UI...")
    print("📚 LangChain-powered Document Management System")
    print("=" * 60)
    
    # Check if services are running
    check_services()
    
    print("\n📋 Available Services:")
    print("  • Knowledge Base Service (Port 8001)")
    print("  • Conversation Service (Port 8002)")
    print("  • API Gateway (Port 8000)")
    
    print(f"\n🌐 Streamlit UI will be available at: http://localhost:8501")
    print("💡 Use demo mode if authentication services are not configured")
    print("-" * 60)
    
    try:
        # Run streamlit with optimized settings
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_ui.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down Knowledge Base UI...")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running Streamlit: {e}")
        print("\n💡 Troubleshooting tips:")
        print("  • Make sure Streamlit is installed: pip install streamlit")
        print("  • Check if port 8501 is available")
        print("  • Verify Python environment is activated")
        return 1
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
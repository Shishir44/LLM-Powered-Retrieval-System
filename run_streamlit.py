#!/usr/bin/env python3
"""
Simple script to run the Streamlit UI for the LLM-Powered Retrieval System.
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed."""
    try:
        import streamlit
        import requests
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("\n📦 Install requirements:")
        print("pip install -r streamlit_requirements.txt")
        return False

def run_streamlit():
    """Run the Streamlit application."""
    if not check_requirements():
        return False
    
    print("🚀 Starting Streamlit UI...")
    print("📱 The UI will be available at: http://localhost:8501")
    print("\n📋 Make sure the services are running:")
    print("  - API Gateway: http://localhost:8000")
    print("  - Knowledge Base Service: http://localhost:8002")
    print("  - Conversation Service: http://localhost:8001")
    print("\n" + "="*50)
    
    try:
        subprocess.run([
            "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Streamlit UI stopped by user")
        return True

if __name__ == "__main__":
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("🤖 LLM-Powered Retrieval System - Streamlit UI")
    print("="*50)
    
    success = run_streamlit()
    sys.exit(0 if success else 1)
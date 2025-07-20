#!/usr/bin/env python3
"""
Simple script to run the Streamlit UI for the Vector Database Testing Interface
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application"""
    
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("🚀 Starting Vector Database Testing Interface...")
    print("📍 Make sure your FastAPI server is running on http://127.0.0.1:5601")
    print("🌐 Streamlit will open in your browser at http://localhost:8501")
    print("-" * 60)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_ui.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Streamlit application...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
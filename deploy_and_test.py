#!/usr/bin/env python3
"""
RAG System Deployment and Testing Script
This script deploys all services and runs comprehensive tests
"""

import os
import sys
import subprocess
import time
import json
import requests
from pathlib import Path

def run_command(command: str, cwd: str = None, shell: bool = True) -> bool:
    """Run a shell command and return success status."""
    try:
        print(f"🔧 Running: {command}")
        result = subprocess.run(
            command, 
            shell=shell, 
            cwd=cwd, 
            capture_output=True, 
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            print("✅ Success")
            return True
        else:
            print(f"❌ Failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_port_available(port: int) -> bool:
    """Check if a port is available."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def wait_for_service(url: str, timeout: int = 60) -> bool:
    """Wait for a service to become available."""
    print(f"⏳ Waiting for service at {url}")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("✅ Service is ready")
                return True
        except:
            pass
        time.sleep(2)
    
    print(f"❌ Service at {url} did not become available within {timeout} seconds")
    return False

def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    # Check Python packages
    required_packages = [
        "fastapi", "uvicorn", "streamlit", "requests", 
        "langchain", "sentence-transformers", "faiss-cpu",
        "numpy", "pydantic", "httpx"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies are installed")
    return True

def setup_environment() -> bool:
    """Set up the environment and configuration."""
    print("🛠️  Setting up environment...")
    
    # Create necessary directories
    directories = [
        "logs",
        "data",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    # Set environment variables
    env_vars = {
        "PYTHONPATH": os.getcwd(),
        "RAG_ENV": "development",
        "LOG_LEVEL": "INFO"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("✅ Environment setup complete")
    return True

def start_services() -> bool:
    """Start all microservices."""
    print("🚀 Starting microservices...")
    
    services = [
        {
            "name": "Knowledge Base Service",
            "port": 8002,
            "command": "python -m uvicorn services.knowledge-base-service.src.main:app --host 0.0.0.0 --port 8002",
            "health_url": "http://localhost:8002/health"
        },
        {
            "name": "Conversation Service", 
            "port": 8001,
            "command": "python -m uvicorn services.conversation-service.src.main:app --host 0.0.0.0 --port 8001",
            "health_url": "http://localhost:8001/health"
        },
        {
            "name": "API Gateway",
            "port": 8080,
            "command": "python -m uvicorn services.api-gateway.src.main:app --host 0.0.0.0 --port 8080",
            "health_url": "http://localhost:8080/health"
        }
    ]
    
    # Check if ports are available
    for service in services:
        if not check_port_available(service["port"]):
            print(f"❌ Port {service['port']} is already in use for {service['name']}")
            return False
    
    # Start services in background
    processes = []
    for service in services:
        print(f"🚀 Starting {service['name']} on port {service['port']}")
        
        # Start service in background
        process = subprocess.Popen(
            service["command"],
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd()
        )
        processes.append(process)
        time.sleep(3)  # Give service time to start
        
        # Check if service started successfully
        if not wait_for_service(service["health_url"], timeout=30):
            # Kill all started processes
            for p in processes:
                p.terminate()
            return False
    
    print("✅ All services started successfully")
    return True

def load_sample_data() -> bool:
    """Load sample documents into the knowledge base."""
    print("📖 Loading sample documents...")
    
    try:
        # Run the document loader
        result = subprocess.run([
            sys.executable, "load_sample_documents.py"
        ], capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0:
            print("✅ Sample documents loaded successfully")
            return True
        else:
            print(f"❌ Failed to load sample documents: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading sample documents: {e}")
        return False

def run_tests() -> bool:
    """Run comprehensive tests."""
    print("🧪 Running comprehensive tests...")
    
    try:
        # Run the evaluation framework
        result = subprocess.run([
            sys.executable, "evaluate_rag_system.py"
        ], capture_output=True, text=True, timeout=300)
        
        print("📊 Test Results:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  Warnings/Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ All tests completed successfully")
            return True
        else:
            print("❌ Some tests failed")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def test_streamlit_ui() -> bool:
    """Test the Streamlit UI."""
    print("🎨 Testing Streamlit UI...")
    
    # Check if streamlit app exists and can be started
    if not Path("streamlit_app.py").exists():
        print("❌ streamlit_app.py not found")
        return False
    
    # Test basic functionality
    try:
        # Simple test queries through the API
        test_queries = [
            "What is Docker?",
            "How does CI/CD work?",
            "Define zero trust security"
        ]
        
        for query in test_queries[:2]:  # Test first 2 queries
            print(f"🔍 Testing query: {query}")
            response = requests.post(
                "http://localhost:8080/conversation/api/v1/chat",
                json={
                    "message": query,
                    "conversation_id": None,
                    "context": {}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                print("✅ Query successful")
            else:
                print(f"❌ Query failed: {response.status_code}")
                return False
        
        print("✅ UI backend tests passed")
        print("💡 To test the full UI, run: streamlit run streamlit_app.py")
        return True
        
    except Exception as e:
        print(f"❌ UI test error: {e}")
        return False

def generate_deployment_report() -> Dict:
    """Generate a deployment report."""
    print("📋 Generating deployment report...")
    
    services_status = {}
    
    # Test each service
    services = [
        ("API Gateway", "http://localhost:8080/health"),
        ("Knowledge Base", "http://localhost:8002/health"),  
        ("Conversation", "http://localhost:8001/health")
    ]
    
    for service_name, health_url in services:
        try:
            response = requests.get(health_url, timeout=5)
            services_status[service_name] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time": response.elapsed.total_seconds(),
                "details": response.json() if response.status_code == 200 else None
            }
        except Exception as e:
            services_status[service_name] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    # Get knowledge base stats
    kb_stats = {}
    try:
        response = requests.get("http://localhost:8002/stats", timeout=10)
        if response.status_code == 200:
            kb_stats = response.json()
    except:
        kb_stats = {"error": "Could not retrieve stats"}
    
    report = {
        "deployment_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "services_status": services_status,
        "knowledge_base_stats": kb_stats,
        "endpoints": {
            "api_gateway": "http://localhost:8080",
            "knowledge_base": "http://localhost:8002", 
            "conversation": "http://localhost:8001",
            "streamlit_ui": "streamlit run streamlit_app.py"
        },
        "test_queries": [
            "What is Docker?",
            "How does CI/CD work?",
            "Define zero trust security",
            "Compare traditional APIs with Web3",
            "Design a system using serverless and AI"
        ]
    }
    
    # Save report
    with open("deployment_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("✅ Deployment report saved to deployment_report.json")
    return report

def main():
    """Main deployment and testing function."""
    print("🚀 RAG System Deployment and Testing")
    print("=" * 50)
    
    success = True
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        sys.exit(1)
    
    # Step 2: Setup environment
    if not setup_environment():
        print("❌ Environment setup failed")
        sys.exit(1)
    
    # Step 3: Start services
    print("\n" + "=" * 50)
    print("🚀 STARTING SERVICES")
    print("=" * 50)
    
    if not start_services():
        print("❌ Failed to start services")
        sys.exit(1)
    
    # Step 4: Load sample data
    print("\n" + "=" * 50)  
    print("📖 LOADING DATA")
    print("=" * 50)
    
    if not load_sample_data():
        print("❌ Failed to load sample data")
        success = False
    
    # Step 5: Run tests
    print("\n" + "=" * 50)
    print("🧪 RUNNING TESTS") 
    print("=" * 50)
    
    if not run_tests():
        print("❌ Tests failed")
        success = False
    
    # Step 6: Test UI
    print("\n" + "=" * 50)
    print("🎨 TESTING UI")
    print("=" * 50)
    
    if not test_streamlit_ui():
        print("❌ UI tests failed")
        success = False
    
    # Step 7: Generate report
    print("\n" + "=" * 50)
    print("📋 GENERATING REPORT")
    print("=" * 50)
    
    report = generate_deployment_report()
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎯 DEPLOYMENT SUMMARY") 
    print("=" * 50)
    
    if success:
        print("🎉 RAG system deployed and tested successfully!")
        print("\n🔗 Access points:")
        print("  • API Gateway: http://localhost:8080")
        print("  • Knowledge Base API: http://localhost:8002")  
        print("  • Conversation API: http://localhost:8001")
        print("  • Streamlit UI: streamlit run streamlit_app.py")
        
        print("\n💡 Next steps:")
        print("  1. Review evaluation report: rag_evaluation_report.json")
        print("  2. Start Streamlit UI: streamlit run streamlit_app.py")
        print("  3. Test with custom queries")
        print("  4. Monitor logs in the logs/ directory")
    else:
        print("❌ Deployment completed with errors")
        print("🔍 Check logs and reports for details")
        print("💡 Try running individual components separately")
    
    print(f"\n📊 Services Status:")
    for service, status in report["services_status"].items():
        emoji = "✅" if status["status"] == "healthy" else "❌"
        print(f"  {emoji} {service}: {status['status']}")
    
    print("\n📁 Generated files:")
    print("  • deployment_report.json - Deployment details")
    print("  • rag_evaluation_report.json - Test results")
    print("  • logs/ - Service logs")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Deployment interrupted by user")
        print("🧹 Cleaning up...")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
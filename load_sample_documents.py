#!/usr/bin/env python3
"""
Script to load sample documents into the knowledge base.
This script loads the technology documents from sample_documents.json into the RAG system.
"""

import json
import requests
import sys
import time
from typing import List, Dict, Any

def load_documents_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load documents from JSON file."""
    try:
        with open(file_path, 'r') as f:
            documents = json.load(f)
        print(f"✅ Loaded {len(documents)} documents from {file_path}")
        return documents
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in file: {e}")
        sys.exit(1)

def check_service_health(base_url: str) -> bool:
    """Check if the knowledge base service is running."""
    try:
        response = requests.get(f"{base_url.replace('/api/v1', '')}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def bulk_upload_documents(documents: List[Dict[str, Any]], base_url: str) -> Dict[str, Any]:
    """Upload documents in bulk to the knowledge base."""
    try:
        response = requests.post(
            f"{base_url}/documents/bulk",
            json=documents,
            timeout=120  # Longer timeout for bulk operations
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Bulk upload failed: {response.status_code}")
            print(f"Error: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out. The service might be overloaded.")
        return None
    except Exception as e:
        print(f"❌ Error during bulk upload: {e}")
        return None

def upload_documents_individually(documents: List[Dict[str, Any]], base_url: str) -> Dict[str, Any]:
    """Upload documents one by one if bulk upload fails."""
    print("📤 Uploading documents individually...")
    
    created_count = 0
    failed_count = 0
    failed_documents = []
    
    for i, doc in enumerate(documents):
        try:
            response = requests.post(
                f"{base_url}/documents",
                json=doc,
                timeout=30
            )
            
            if response.status_code == 200:
                created_count += 1
                print(f"✅ Created: {doc['title']} ({i+1}/{len(documents)})")
            else:
                failed_count += 1
                failed_documents.append({
                    "title": doc["title"],
                    "error": f"HTTP {response.status_code}: {response.text}"
                })
                print(f"❌ Failed: {doc['title']} - {response.status_code}")
                
        except Exception as e:
            failed_count += 1
            failed_documents.append({
                "title": doc["title"],
                "error": str(e)
            })
            print(f"❌ Error creating {doc['title']}: {e}")
        
        # Small delay to avoid overwhelming the service
        time.sleep(0.1)
    
    return {
        "created_count": created_count,
        "failed_count": failed_count,
        "failed_documents": failed_documents
    }

def get_knowledge_base_stats(base_url: str) -> Dict[str, Any]:
    """Get statistics from the knowledge base."""
    try:
        response = requests.get(f"{base_url}/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def main():
    # Configuration
    KNOWLEDGE_BASE_URL = "http://localhost:8002/api/v1"
    SAMPLE_DOCS_FILE = "sample_documents.json"
    
    print("🚀 RAG System Document Loader")
    print("=" * 50)
    
    # Check service health
    print("🔍 Checking service health...")
    if not check_service_health(KNOWLEDGE_BASE_URL):
        print(f"❌ Knowledge Base service is not running at {KNOWLEDGE_BASE_URL}")
        print("Please start the services using:")
        print("cd setup && docker-compose up -d")
        sys.exit(1)
    
    print("✅ Knowledge Base service is running")
    
    # Load documents
    print(f"📖 Loading documents from {SAMPLE_DOCS_FILE}...")
    documents = load_documents_from_file(SAMPLE_DOCS_FILE)
    
    # Get current stats
    print("📊 Getting current knowledge base stats...")
    initial_stats = get_knowledge_base_stats(KNOWLEDGE_BASE_URL)
    if initial_stats:
        print(f"Current documents: {initial_stats.get('total_documents', 0)}")
        print(f"Current chunks: {initial_stats.get('total_chunks', 0)}")
    
    # Try bulk upload first
    print(f"📤 Attempting bulk upload of {len(documents)} documents...")
    result = bulk_upload_documents(documents, KNOWLEDGE_BASE_URL)
    
    if result is None:
        # Fall back to individual uploads
        print("⚠️  Bulk upload failed, trying individual uploads...")
        result = upload_documents_individually(documents, KNOWLEDGE_BASE_URL)
    
    # Show results
    print("\n" + "=" * 50)
    print("📊 Upload Results")
    print("=" * 50)
    print(f"✅ Successfully created: {result.get('created_count', 0)} documents")
    print(f"❌ Failed: {result.get('failed_count', 0)} documents")
    
    if result.get('failed_documents'):
        print("\n❌ Failed documents:")
        for failed in result['failed_documents']:
            print(f"  - {failed.get('title', 'Unknown')}: {failed.get('error', 'Unknown error')}")
    
    # Get final stats
    print("\n📊 Final knowledge base stats...")
    final_stats = get_knowledge_base_stats(KNOWLEDGE_BASE_URL)
    if final_stats:
        print(f"Total documents: {final_stats.get('total_documents', 0)}")
        print(f"Total chunks: {final_stats.get('total_chunks', 0)}")
        print(f"Average chunks per document: {final_stats.get('average_chunks_per_document', 0)}")
        
        categories = final_stats.get('categories', {})
        if categories:
            print("\nCategories:")
            for category, count in categories.items():
                print(f"  - {category}: {count} documents")
    
    print("\n🎉 Document loading complete!")
    print("You can now test the RAG system with the loaded documents.")
    
    # Show some example queries
    print("\n💡 Example queries to test:")
    example_queries = [
        "What is Docker?",
        "How does CI/CD work?",
        "Define zero trust security",
        "What are neural networks used for?",
        "Compare traditional APIs with Web3",
        "How would Docker help in CI/CD pipelines?"
    ]
    
    for query in example_queries:
        print(f"  - {query}")

if __name__ == "__main__":
    main()
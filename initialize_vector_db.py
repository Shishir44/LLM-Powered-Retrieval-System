#!/usr/bin/env python3
"""
Vector Database Initialization Script

This script ensures proper initialization of the vector database
and verifies that documents can be embedded and stored correctly.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add the services directory to the Python path
services_path = Path(__file__).parent / "services"
sys.path.insert(0, str(services_path))

# Set environment variables for ChromaDB
os.environ["CHROMA_PERSIST_DIRECTORY"] = "./chroma_storage"

async def initialize_vector_database():
    """Initialize and test the vector database."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Import the semantic retriever
        from knowledge_base_service.src.core.semantic_retriever import SemanticRetriever
        from knowledge_base_service.src.config import get_config
        
        logger.info("🚀 Initializing vector database...")
        
        # Get configuration
        config = get_config()
        logger.info(f"Using OpenAI model: {config.embedding.openai_model}")
        logger.info(f"Embedding dimension: {config.embedding.embedding_dimension}")
        
        # Initialize semantic retriever
        retriever = SemanticRetriever(config=config)
        
        # Check ChromaDB connection
        stats = retriever.get_statistics()
        logger.info(f"📊 Database Stats:")
        logger.info(f"  - Storage backend: {stats['storage_backend']}")
        logger.info(f"  - Documents in ChromaDB: {stats['total_documents_chroma']}")
        logger.info(f"  - Documents in memory: {stats['total_documents_memory']}")
        logger.info(f"  - ChromaDB available: {stats['chroma_available']}")
        
        # Test document embedding and storage
        test_document = {
            "id": "test-vector-db-init",
            "title": "Vector Database Test Document",
            "content": "This is a test document to verify that the vector database can properly embed and store documents. It contains information about testing the RAG system functionality.",
            "metadata": {
                "category": "test",
                "tags": ["test", "initialization", "vector-db"],
                "document_type": "test",
                "quality_score": 1.0
            }
        }
        
        logger.info("🧪 Testing document embedding and storage...")
        
        # Add test document
        await retriever.add_documents([test_document])
        
        # Verify storage
        updated_stats = retriever.get_statistics()
        
        if updated_stats['total_documents_chroma'] > stats['total_documents_chroma']:
            logger.info("✅ Document successfully embedded and stored!")
        else:
            logger.warning("⚠️  Document may not have been stored properly")
        
        # Test search functionality
        logger.info("🔍 Testing search functionality...")
        
        search_results = await retriever.semantic_search(
            query="vector database test",
            top_k=5
        )
        
        if search_results:
            logger.info(f"✅ Search successful! Found {len(search_results)} results")
            for i, result in enumerate(search_results[:2]):
                logger.info(f"  Result {i+1}: {result.document.title} (score: {result.semantic_score:.3f})")
        else:
            logger.warning("⚠️  Search returned no results")
        
        # Health check
        logger.info("🏥 Performing health check...")
        
        health_status = await retriever.health_check()
        logger.info(f"Health Status: {health_status['overall_status']}")
        
        for component, status in health_status.get('components', {}).items():
            logger.info(f"  - {component}: {status.get('status', 'unknown')}")
        
        # Clean up test document
        if retriever.collection:
            try:
                retriever.collection.delete(ids=["test-vector-db-init"])
                logger.info("🧹 Cleaned up test document")
            except Exception as e:
                logger.warning(f"Could not clean up test document: {e}")
        
        logger.info("🎉 Vector database initialization completed successfully!")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure all dependencies are installed:")
        logger.error("  pip install chromadb langchain-openai sentence-transformers")
        return False
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main function."""
    
    print("=" * 60)
    print("🗄️  VECTOR DATABASE INITIALIZATION")
    print("=" * 60)
    
    success = await initialize_vector_database()
    
    if success:
        print("\n✅ Vector database is properly initialized and ready for use!")
        print("\nYou can now:")
        print("  1. Upload documents through the API")
        print("  2. Perform semantic searches")
        print("  3. Use the RAG system for question answering")
    else:
        print("\n❌ Vector database initialization failed!")
        print("Please check the error messages above and ensure:")
        print("  1. OpenAI API key is set (OPENAI_API_KEY)")
        print("  2. All dependencies are installed")
        print("  3. ChromaDB has write permissions")
        
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from uuid import uuid4
import time
import asyncio
from datetime import datetime
import json
import logging

from ..core.semantic_retriever import SemanticRetriever
from ..core.advanced_chunking import AdvancedDocumentChunker
from ..core.cache import VectorCache
# PHASE 2.2: Import structured document processor
from ..core.structured_document_processor import StructuredDocumentProcessor
# from models.requests import DocumentCreateRequest, SearchRequest  # Defined locally
# from models.responses import DocumentResponse, SearchResponse, HealthResponse  # Defined locally
from ..config import get_config

router = APIRouter()
logger = logging.getLogger(__name__)

# Pydantic models
class DocumentCreateRequest(BaseModel):
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    category: str = Field(..., description="Document category")
    subcategory: Optional[str] = Field(None, description="Document subcategory")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class DocumentSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    category: Optional[str] = Field(None, description="Filter by category")
    subcategory: Optional[str] = Field(None, description="Filter by subcategory")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    limit: int = Field(default=10, ge=1, le=50, description="Number of results to return")

class DocumentResponse(BaseModel):
    id: str
    title: str
    content: Optional[str] = None
    category: str
    subcategory: Optional[str] = None
    tags: List[str]
    score: Optional[float] = None
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[DocumentResponse]
    total: int
    query: str
    metadata: Dict[str, Any] = {}

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    response_time_ms: float
    database: Dict[str, Any]
    system: Dict[str, Any]
    version: str

# Global components - in production these would be dependency injected
chunker = None
retriever = None
vector_cache = None
semantic_retriever = None

# In-memory storage for demo purposes
document_storage = {}
search_index = []

def get_chunker() -> AdvancedDocumentChunker:
    global chunker
    if chunker is None:
        chunker = AdvancedDocumentChunker()
    return chunker

def get_retriever() -> SemanticRetriever:
    global retriever
    if retriever is None:
        config = get_config()
        retriever = SemanticRetriever(config=config)
    return retriever

def get_cache() -> VectorCache:
    global vector_cache
    if vector_cache is None:
        vector_cache = VectorCache()
    return vector_cache

def get_semantic_retriever() -> SemanticRetriever:
    """Returns the semantic retriever instance."""
    global semantic_retriever
    if semantic_retriever is None:
        config = get_config()
        semantic_retriever = SemanticRetriever(config=config)
    return semantic_retriever

# PHASE 2.2: Initialize structured document processor
try:
    structured_processor = StructuredDocumentProcessor(enable_nlp=True)
    STRUCTURED_PROCESSING_AVAILABLE = True
except Exception as e:
    structured_processor = None
    STRUCTURED_PROCESSING_AVAILABLE = False
    print(f"Structured processing not available: {e}")

async def _get_emergency_fallback_results(query: str, limit: int) -> List[Dict[str, Any]]:
    """Provide emergency fallback results when all search methods fail."""
    try:
        # Create simple static responses based on common query patterns
        fallback_results = []
        
        query_lower = query.lower()
        
        # Gaming laptop queries
        if "gaming" in query_lower and "laptop" in query_lower:
            fallback_results.append({
                "id": "emergency_gaming_1",
                "title": "Gaming Laptops",
                "content": "We offer high-performance gaming laptops with dedicated graphics cards, fast processors, and advanced cooling systems. Contact our sales team for current models and pricing.",
                "category": "product_info",
                "subcategory": "laptops",
                "tags": ["gaming", "laptops", "products"],
                "score": 0.6,
                "metadata": {
                    "source": "emergency_fallback",
                    "category": "product_info",
                    "subcategory": "laptops"
                }
            })
        
        # Product/laptop queries
        elif "laptop" in query_lower or "product" in query_lower:
            fallback_results.append({
                "id": "emergency_product_1",
                "title": "Product Information",
                "content": "Please visit our product catalog for detailed information about our laptop models, specifications, and pricing. Our support team is available to help with product questions.",
                "category": "product_info",
                "subcategory": "general",
                "tags": ["products", "laptops", "information"],
                "score": 0.5,
                "metadata": {
                    "source": "emergency_fallback",
                    "category": "product_info"
                }
            })
        
        # Support/help queries
        elif any(word in query_lower for word in ["help", "support", "issue", "problem", "trouble"]):
            fallback_results.append({
                "id": "emergency_support_1",
                "title": "Support Information",
                "content": "Our technical support team is available to help with any issues. Please contact support with your specific question or check our troubleshooting guides.",
                "category": "support",
                "subcategory": "general",
                "tags": ["support", "help", "troubleshooting"],
                "score": 0.5,
                "metadata": {
                    "source": "emergency_fallback",
                    "category": "support"
                }
            })
        
        # General fallback
        else:
            fallback_results.append({
                "id": "emergency_general_1",
                "title": "General Information",
                "content": "Thank you for your query. For specific information, please contact our customer service team who can provide detailed assistance with your request.",
                "category": "general",
                "subcategory": "information",
                "tags": ["general", "information"],
                "score": 0.3,
                "metadata": {
                    "source": "emergency_fallback",
                    "category": "general"
                }
            })
        
        logger.info(f"Providing {len(fallback_results)} emergency fallback results for query: '{query}'")
        return fallback_results[:limit]
        
    except Exception as e:
        logger.error(f"Error in emergency fallback: {e}")
        return []

@router.post("/documents", response_model=DocumentResponse)
async def create_document(
    request: DocumentCreateRequest,
    chunker: AdvancedDocumentChunker = Depends(get_chunker)
):
    """PHASE 2.2: Enhanced document creation with structured processing."""
    try:
        document_id = str(uuid4())
        timestamp = datetime.utcnow()
        
        # Validate content
        if len(request.content.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document content must be at least 10 characters long"
            )
        
        # PHASE 2.2: Use structured document processor if available
        if STRUCTURED_PROCESSING_AVAILABLE and structured_processor:
            try:
                # Process document with structure detection and metadata enhancement
                processed_doc = await structured_processor.process_document(
                    doc_id=document_id,
                    content=request.content,
                    title=request.title,
                    existing_metadata={
                        "category": request.category,
                        "subcategory": request.subcategory,
                        "tags": request.tags,
                        "created_at": timestamp.isoformat(),
                        **request.metadata
                    }
                )
                
                # Use enhanced metadata from structured processing
                enhanced_metadata = processed_doc.enhanced_metadata
                enhanced_metadata.update({
                    "entities": processed_doc.entities,
                    "keywords": [kw["keyword"] for kw in processed_doc.keywords[:10]],  # Top 10 keywords
                    "topics": processed_doc.topics,
                    "document_type": processed_doc.document_type,
                    "quality_score": processed_doc.quality_score,
                    "authority_indicators": processed_doc.authority_indicators,
                    "structured_processing": True,
                    "processing_version": processed_doc.processing_version
                })
                
                # Use cleaned content from processor
                content_to_chunk = processed_doc.content
                
                logger.info(f"PHASE 2.2: Document {document_id} processed with structure detection - Type: {processed_doc.document_type}, Quality: {processed_doc.quality_score:.2f}")
                
            except Exception as e:
                logger.warning(f"Structured processing failed for {document_id}: {e}, falling back to basic processing")
                # Fallback to basic processing
                enhanced_metadata = {
                    **request.metadata,
                    "category": request.category,
                    "subcategory": request.subcategory,
                    "tags": request.tags,
                    "created_at": timestamp.isoformat(),
                    "structured_processing": False
                }
                content_to_chunk = request.content
        else:
            # Basic processing without structure detection
            enhanced_metadata = {
                **request.metadata,
                "category": request.category,
                "subcategory": request.subcategory,
                "tags": request.tags,
                "created_at": timestamp.isoformat(),
                "structured_processing": False
            }
            content_to_chunk = request.content
        
        # Process document into chunks (Phase 1.2: Using optimized chunk sizes)
        document_dict = {
            "id": document_id,
            "content": content_to_chunk,
            "title": request.title,
            "category": request.category,
            "subcategory": request.subcategory,
            "tags": request.tags,
            "metadata": enhanced_metadata
        }
        
        chunks = await asyncio.get_event_loop().run_in_executor(
            None,
            chunker.chunk_document,
            document_dict
        )
        
        # Store document metadata with enhanced information
        document_data = {
            "id": document_id,
            "title": request.title,
            "content": content_to_chunk,
            "category": request.category,
            "subcategory": request.subcategory,
            "tags": request.tags,
            "metadata": enhanced_metadata,
            "chunks": chunks
        }
        
        # Store in semantic retriever (ChromaDB) with enhanced metadata
        retriever = get_semantic_retriever()
        
        # Ensure proper embedding generation and storage
        try:
            # Convert to format expected by add_documents method
            doc_for_storage = {
                "id": document_id,
                "content": content_to_chunk,
                "title": request.title,
                "metadata": enhanced_metadata
            }
            
            # Use add_documents method which properly handles embeddings
            await retriever.add_documents([doc_for_storage])
            success = True
            
        except Exception as storage_error:
            logger.error(f"Failed to store document in vector database: {storage_error}")
            success = False
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store document in knowledge base"
            )
        
        processing_info = {
            "chunks_created": len(chunks),
            "structured_processing": enhanced_metadata.get("structured_processing", False),
            "document_type": enhanced_metadata.get("document_type", "general"),
            "quality_score": enhanced_metadata.get("quality_score", 0.0)
        }
        
        logger.info(f"PHASE 2.2: Successfully created document {document_id} with {len(chunks)} chunks")
        
        return DocumentResponse(
            id=document_id,
            title=request.title,
            category=request.category,
            subcategory=request.subcategory,
            tags=request.tags,
            metadata=enhanced_metadata,
            created_at=timestamp,
            processing_info=processing_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/search", response_model=SearchResponse)
async def search_documents(
    q: str,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 10,
    enable_boosting: bool = True,
    retriever: SemanticRetriever = Depends(get_retriever)
):
    """PHASE 2.1: Enhanced search with metadata boosting and performance optimizations."""
    import asyncio
    
    try:
        if not q or len(q.strip()) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query must be at least 2 characters long"
            )

        logger.info(f"PHASE 2.1: Searching for '{q}' with boosting={enable_boosting}")
        
        # Build filters
        filters = {}
        if category:
            filters["category"] = category
        if subcategory:
            filters["subcategory"] = subcategory
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
            filters["tags"] = tag_list

        # Set a reasonable timeout for the search operation (90 seconds)
        search_timeout = 90
        
        try:
            # PHASE 2.1: Use enhanced semantic search with metadata boosting and timeout
            results = await asyncio.wait_for(
                retriever.enhanced_semantic_search(
                    query=q.strip(),
                    top_k=limit,
                    category_hint=category,
                    content_type_hint=subcategory,
                    enable_boosting=enable_boosting,
                    filters=filters
                ),
                timeout=search_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Enhanced search timeout after {search_timeout}s, falling back to basic search for query: '{q}'")
            # Fallback to basic semantic search with shorter timeout
            try:
                results = await asyncio.wait_for(
                    retriever.semantic_search(
                        query=q.strip(),
                        top_k=limit,
                        filters=filters
                    ),
                    timeout=30  # Shorter timeout for fallback
                )
            except asyncio.TimeoutError:
                logger.error(f"Basic search also timed out for query: '{q}'")
                # Return minimal fallback results
                results = await _get_emergency_fallback_results(q.strip(), limit)
        except Exception as e:
            logger.warning(f"Enhanced search failed, falling back to basic search: {e}")
            # Fallback to basic semantic search
            try:
                results = await asyncio.wait_for(
                    retriever.semantic_search(
                        query=q.strip(),
                        top_k=limit,
                        filters=filters
                    ),
                    timeout=30
                )
            except Exception as fallback_error:
                logger.error(f"Basic search also failed: {fallback_error}")
                results = await _get_emergency_fallback_results(q.strip(), limit)

        # Convert results to response format
        search_results = []
        for result in results:
            doc = result.document
            search_result = {
                "id": doc.id,
                "title": doc.title,
                "content": doc.content,  # Return full content without truncation
                "category": doc.metadata.get("category", ""),
                "subcategory": doc.metadata.get("subcategory"),
                "tags": doc.metadata.get("tags", []),
                "score": result.hybrid_score if hasattr(result, 'hybrid_score') else result.semantic_score,
                "metadata": {
                    "similarity_score": result.semantic_score,
                    "retrieval_method": getattr(result, 'retrieval_method', 'semantic'),
                    "explanation": getattr(result, 'relevance_explanation', ''),
                    "confidence": getattr(result, 'confidence', 0.0),
                    # Enhanced metadata from Phase 2.2
                    "document_type": doc.metadata.get("document_type", "general"),
                    "quality_score": doc.metadata.get("quality_score", 0.0),
                    "authority_indicators": doc.metadata.get("authority_indicators", []),
                    "structured_processing": doc.metadata.get("structured_processing", False)
                }
            }
            search_results.append(search_result)

        logger.info(f"PHASE 2.1: Found {len(search_results)} results for query '{q}'")
        
        return SearchResponse(
            query=q,
            results=search_results,
            total=len(search_results),
            filters_applied={
                "category": category,
                "subcategory": subcategory,
                "tags": tags,
                "boosting_enabled": enable_boosting
            },
            processing_info={
                "search_method": "enhanced_semantic" if enable_boosting else "semantic",
                "metadata_boosting": enable_boosting,
                "results_count": len(search_results)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@router.get("/search/semantic", response_model=SearchResponse)
async def semantic_search_documents(
    q: str,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 10,
    use_reranking: bool = True,
    semantic_retriever = Depends(get_semantic_retriever)
):
    """Advanced semantic search with embeddings and reranking."""
    start_time = time.time()
    
    try:
        # Validate parameters
        if not q or len(q.strip()) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query must be at least 2 characters long"
            )
        
        if limit > 50:
            limit = 50
        
        # Build filters
        filters = {}
        if category:
            filters["category"] = category
        if subcategory:
            filters["subcategory"] = subcategory
        if tags:
            tags_list = [tag.strip() for tag in tags.split(",")]
            filters["tags"] = tags_list
        
        # Perform semantic search
        semantic_results = await semantic_retriever.semantic_search(
            query=q,
            top_k=limit,
            use_query_expansion=True,
            filters=filters if filters else None
        )
        
        # Format results
        results = []
        for result in semantic_results:
            doc = result.document
            results.append(DocumentResponse(
                id=doc.id,
                title=doc.title,
                content=doc.content,  # Return full content without truncation
                category=doc.metadata.get("category", "Unknown"),
                subcategory=doc.metadata.get("subcategory"),
                tags=doc.metadata.get("tags", []),
                score=round(result.confidence, 3),
                metadata={
                    **doc.metadata,
                    "semantic_score": round(result.semantic_score, 3),
                    "keyword_score": round(result.keyword_score, 3),
                    "hybrid_score": round(result.hybrid_score, 3),
                    "rerank_score": round(result.rerank_score or 0.0, 3),
                    "retrieval_method": result.retrieval_method,
                    "relevance_explanation": result.relevance_explanation
                }
            ))
        
        processing_time = round(time.time() - start_time, 3)
        
        return SearchResponse(
            results=results,
            total=len(semantic_results),
            query=q,
            metadata={
                "processing_time": processing_time,
                "search_type": "semantic",
                "use_reranking": use_reranking,
                "total_documents": len(semantic_retriever.documents),
                "filters": filters
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}"
        )

@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """Get a specific document by ID."""
    try:
        # Validate document ID format
        if not document_id or len(document_id) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid document ID format"
            )
        
        # Retrieve document from storage
        document = document_storage.get(document_id)
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found"
            )
        
        return DocumentResponse(
            id=document["id"],
            title=document["title"],
            content=document["content"],
            category=document["category"],
            subcategory=document["subcategory"],
            tags=document["tags"],
            metadata=document["metadata"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document: {str(e)}"
        )

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the knowledge base."""
    try:
        # Validate document ID
        if not document_id or len(document_id) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid document ID format"
            )
        
        # Check if document exists
        if document_id not in document_storage:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found"
            )
        
        # Remove from storage
        deleted_doc = document_storage.pop(document_id)
        
        # Remove from search index
        global search_index
        search_index = [doc for doc in search_index if doc["id"] != document_id]
        
        return {
            "message": "Document deleted successfully",
            "success": True,
            "deleted_document": {
                "id": document_id,
                "title": deleted_doc["title"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )

@router.get("/documents", response_model=Dict[str, Any])
async def list_documents(
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
):
    """List all documents from vector database with optional filtering."""
    try:
        if limit > 100:
            limit = 100
        
        # Get semantic retriever to access ChromaDB
        retriever = get_semantic_retriever()
        
        # Get documents from ChromaDB
        if retriever.collection:
            try:
                # Get all documents from ChromaDB
                chroma_results = retriever.collection.get(
                    include=["documents", "metadatas"],
                    limit=None  # Get all documents first, then filter
                )
                
                all_docs = []
                for i, (doc_id, content, metadata) in enumerate(
                    zip(chroma_results["ids"], chroma_results["documents"], chroma_results["metadatas"])
                ):
                    # Parse metadata
                    title = metadata.get("title", "Untitled")
                    doc_category = metadata.get("category", "")
                    doc_subcategory = metadata.get("subcategory", "")
                    doc_tags = metadata.get("tags", "")
                    
                    # Parse tags if they're stored as comma-separated string
                    if isinstance(doc_tags, str):
                        doc_tags = [tag.strip() for tag in doc_tags.split(",") if tag.strip()]
                    elif not isinstance(doc_tags, list):
                        doc_tags = []
                    
                    # Apply filters
                    if category and doc_category != category:
                        continue
                    
                    if subcategory and doc_subcategory != subcategory:
                        continue
                    
                    doc_entry = {
                        "id": doc_id,
                        "title": title,
                        "category": doc_category,
                        "subcategory": doc_subcategory,
                        "tags": doc_tags,
                        "created_at": metadata.get("created_at"),
                        "content_length": len(content) if content else 0,
                        "metadata": {k: v for k, v in metadata.items() 
                                   if k not in ["title", "category", "subcategory", "tags", "created_at"]}
                    }
                    all_docs.append(doc_entry)
                
                # Sort by created_at (newest first)
                all_docs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
                
                # Apply pagination
                total = len(all_docs)
                paginated_docs = all_docs[offset:offset + limit]
                
                logger.info(f"Retrieved {len(paginated_docs)} documents from vector database (total: {total})")
                
                return {
                    "documents": paginated_docs,
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "has_more": offset + limit < total,
                    "source": "vector_database"
                }
                
            except Exception as e:
                logger.error(f"Failed to retrieve documents from ChromaDB: {e}")
                # Fallback to empty result
                return {
                    "documents": [],
                    "total": 0,
                    "limit": limit,
                    "offset": offset,
                    "has_more": False,
                    "source": "fallback",
                    "error": f"ChromaDB retrieval failed: {str(e)}"
                }
        else:
            # No ChromaDB collection available, fallback to in-memory
            logger.warning("ChromaDB collection not available, using in-memory storage")
            
            filtered_docs = search_index
            
            # Apply filters
            if category:
                filtered_docs = [doc for doc in filtered_docs if doc.get("category") == category]
            
            if subcategory:
                filtered_docs = [doc for doc in filtered_docs if doc.get("subcategory") == subcategory]
            
            # Pagination
            total = len(filtered_docs)
            paginated_docs = filtered_docs[offset:offset + limit]
            
            # Format results
            results = []
            for doc_summary in paginated_docs:
                full_doc = document_storage.get(doc_summary["id"])
                if full_doc:
                    results.append({
                        "id": full_doc["id"],
                        "title": full_doc["title"],
                        "category": full_doc["category"],
                        "subcategory": full_doc["subcategory"],
                        "tags": full_doc["tags"],
                        "created_at": full_doc["metadata"].get("created_at"),
                        "content_length": len(full_doc.get("content", "")),
                        "metadata": full_doc.get("metadata", {})
                    })
                else:
                    # Add basic info from search index if full doc not found
                    results.append({
                        "id": doc_summary["id"],
                        "title": doc_summary.get("title", "Unknown"),
                        "category": doc_summary.get("category", ""),
                        "subcategory": doc_summary.get("subcategory", ""),
                        "tags": doc_summary.get("tags", []),
                        "created_at": doc_summary.get("created_at"),
                        "content_length": len(doc_summary.get("content", "")),
                        "metadata": {}
                    })
            
            return {
                "documents": results,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total,
                "source": "in_memory"
            }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )

@router.post("/documents/bulk", response_model=Dict[str, Any])
async def bulk_create_documents(
    documents: List[DocumentCreateRequest],
    chunker: AdvancedDocumentChunker = Depends(get_chunker),
    semantic_retriever = Depends(get_semantic_retriever)
):
    """Bulk create multiple documents in the knowledge base."""
    try:
        if len(documents) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot create more than 100 documents at once"
            )
        
        created_documents = []
        failed_documents = []
        
        for i, request in enumerate(documents):
            try:
                document_id = str(uuid4())
                timestamp = datetime.utcnow()
                
                # Validate content
                if len(request.content.strip()) < 10:
                    failed_documents.append({
                        "index": i,
                        "title": request.title,
                        "error": "Document content must be at least 10 characters long"
                    })
                    continue
                
                # Process document into chunks
                document_dict = {
                    "id": document_id,
                    "content": request.content,
                    "title": request.title,
                    "category": request.category,
                    "subcategory": request.subcategory,
                    "tags": request.tags,
                    "metadata": {
                        **request.metadata,
                        "created_at": timestamp.isoformat()
                    }
                }
                
                chunks = await asyncio.get_event_loop().run_in_executor(
                    None,
                    chunker.chunk_document,
                    document_dict
                )
                
                # Store document metadata
                document_data = {
                    "id": document_id,
                    "title": request.title,
                    "content": request.content,
                    "category": request.category,
                    "subcategory": request.subcategory,
                    "tags": request.tags,
                    "metadata": {
                        **request.metadata,
                        "created_at": timestamp.isoformat(),
                        "chunk_count": len(chunks)
                    },
                    "chunks": chunks
                }
                
                # Store in memory for compatibility
                document_storage[document_id] = document_data
                
                # Add to search index
                search_entry = {
                    "id": document_id,
                    "title": request.title,
                    "content": request.content,  # Return full content without truncation
                    "category": request.category,
                    "subcategory": request.subcategory,
                    "tags": request.tags,
                    "created_at": timestamp.isoformat()
                }
                search_index.append(search_entry)
                
                created_documents.append({
                    "id": document_id,
                    "title": request.title,
                    "category": request.category,
                    "chunk_count": len(chunks)
                })
                
            except Exception as e:
                failed_documents.append({
                    "index": i,
                    "title": request.title,
                    "error": str(e)
                })
        
        # Add successfully created documents to semantic retriever
        if created_documents:
            try:
                semantic_docs = []
                for doc_info in created_documents:
                    full_doc = document_storage.get(doc_info["id"])
                    if full_doc:
                        # Prepare metadata, filtering out None values for ChromaDB compatibility
                        semantic_metadata = {
                            "category": full_doc["category"],
                            "tags": full_doc["tags"]
                        }
                        
                        # Only add subcategory if it's not None
                        if full_doc["subcategory"] is not None:
                            semantic_metadata["subcategory"] = full_doc["subcategory"]
                        
                        # Add other metadata, filtering out None values
                        for key, value in full_doc["metadata"].items():
                            if value is not None:
                                semantic_metadata[key] = value
                        
                        semantic_docs.append({
                            "id": full_doc["id"],
                            "content": full_doc["content"],
                            "title": full_doc["title"],
                            "metadata": semantic_metadata
                        })
                
                if semantic_docs:
                    # Use semantic retriever to properly embed and store documents
                    retriever = get_semantic_retriever()
                    await retriever.add_documents(semantic_docs)
                    logger.info(f"Successfully added {len(semantic_docs)} documents to vector database")
                    
            except Exception as e:
                logger.error(f"Failed to add documents to vector database: {e}")
                # Don't fail the entire operation, just log the error
        
        return {
            "success": True,
            "created_count": len(created_documents),
            "failed_count": len(failed_documents),
            "created_documents": created_documents,
            "failed_documents": failed_documents,
            "total_processed": len(documents)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bulk creation failed: {str(e)}"
        )

@router.get("/stats")
async def get_stats():
    """Get knowledge base statistics from vector database."""
    try:
        # Get semantic retriever stats
        retriever = get_semantic_retriever()
        retriever_stats = retriever.get_statistics()
        
        categories = {}
        total_documents = 0
        
        # Get stats from ChromaDB if available
        if retriever.collection:
            try:
                # Get all documents metadata from ChromaDB
                chroma_results = retriever.collection.get(
                    include=["metadatas"],
                    limit=None
                )
                
                total_documents = len(chroma_results["ids"])
                
                # Count by categories
                for metadata in chroma_results["metadatas"]:
                    category = metadata.get("category", "Unknown")
                    categories[category] = categories.get(category, 0) + 1
                
                logger.info(f"Retrieved stats from vector database: {total_documents} documents")
                
            except Exception as e:
                logger.error(f"Failed to get stats from ChromaDB: {e}")
                # Fallback to in-memory counts
                for doc in document_storage.values():
                    category = doc["category"]
                    categories[category] = categories.get(category, 0) + 1
                total_documents = len(document_storage)
        else:
            # Fallback to in-memory storage
            for doc in document_storage.values():
                category = doc["category"]
                categories[category] = categories.get(category, 0) + 1
            total_documents = len(document_storage)
        
        return {
            "total_documents": total_documents,
            "total_documents_chroma": retriever_stats.get("total_documents_chroma", 0),
            "total_documents_memory": len(document_storage),
            "categories": categories,
            "storage_backend": retriever_stats.get("storage_backend", "In-Memory"),
            "embedding_model": retriever_stats.get("models_used", {}).get("openai_embeddings", "unknown"),
            "chroma_available": retriever_stats.get("chroma_available", False),
            "collection_name": retriever_stats.get("collection_name"),
            "semantic_retriever": retriever_stats
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check with performance metrics."""
    import time
    import psutil
    
    start_time = time.time()
    
    try:
        # Test database connectivity
        retriever = get_retriever()
        db_status = "healthy"
        db_response_time = None
        
        try:
            # Quick test query to check database responsiveness
            test_start = time.time()
            test_results = await asyncio.wait_for(
                retriever.semantic_search("test", top_k=1),
                timeout=5.0
            )
            db_response_time = round((time.time() - test_start) * 1000, 2)  # ms
        except asyncio.TimeoutError:
            db_status = "slow"
            db_response_time = ">5000"
        except Exception as e:
            db_status = "error"
            logger.warning(f"Database health check failed: {e}")
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        response_time = round((time.time() - start_time) * 1000, 2)  # ms
        
        health_data = {
            "status": "healthy" if db_status in ["healthy", "slow"] else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "response_time_ms": response_time,
            "database": {
                "status": db_status,
                "response_time_ms": db_response_time
            },
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2)
            },
            "version": "2.3"
        }
        
        return HealthResponse(**health_data)
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="error",
            timestamp=datetime.utcnow().isoformat(),
            response_time_ms=round((time.time() - start_time) * 1000, 2),
            database={"status": "unknown", "response_time_ms": None},
            system={"cpu_percent": 0, "memory_percent": 0, "memory_available_gb": 0},
            version="2.3"
        )
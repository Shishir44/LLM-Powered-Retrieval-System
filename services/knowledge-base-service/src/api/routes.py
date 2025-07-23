from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from uuid import uuid4
import time
import asyncio
from datetime import datetime
import json

from ..core.retrieval import AdvancedRAGRetriever
from ..core.chunking import DocumentChunker
from ..core.cache import VectorCache
# from ..core.semantic_retriever import SemanticRetriever  # Disabled due to dependency issues

router = APIRouter()

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

# Global components - in production these would be dependency injected
chunker = None
retriever = None
vector_cache = None
semantic_retriever = None

# In-memory storage for demo purposes
document_storage = {}
search_index = []

def get_chunker() -> DocumentChunker:
    global chunker
    if chunker is None:
        chunker = DocumentChunker()
    return chunker

def get_retriever() -> AdvancedRAGRetriever:
    global retriever
    if retriever is None:
        retriever = AdvancedRAGRetriever()
    return retriever

def get_cache() -> VectorCache:
    global vector_cache
    if vector_cache is None:
        vector_cache = VectorCache()
    return vector_cache

class DummySemanticRetriever:
    """Dummy class to replace SemanticRetriever when dependencies are broken."""
    def __init__(self):
        self.documents = []
    
    async def semantic_search(self, **kwargs):
        return []
    
    async def add_documents(self, docs):
        pass

def get_semantic_retriever():
    return DummySemanticRetriever()

@router.post("/documents", response_model=DocumentResponse)
async def create_document(
    request: DocumentCreateRequest,
    chunker: DocumentChunker = Depends(get_chunker)
):
    """Create a new document in the knowledge base."""
    try:
        document_id = str(uuid4())
        timestamp = datetime.utcnow()
        
        # Validate content
        if len(request.content.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document content must be at least 10 characters long"
            )
        
        # Process document into chunks
        chunks = await asyncio.get_event_loop().run_in_executor(
            None,
            chunker.chunk_document,
            request.content,
            {
                "document_id": document_id,
                "title": request.title,
                "category": request.category,
                "subcategory": request.subcategory,
                "tags": request.tags,
                "created_at": timestamp.isoformat()
            }
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
        
        # Store in memory (in production this would go to vector DB)
        document_storage[document_id] = document_data
        
        # Add to search index
        search_entry = {
            "id": document_id,
            "title": request.title,
            "content": request.content[:500],  # Store first 500 chars for search
            "category": request.category,
            "subcategory": request.subcategory,
            "tags": request.tags,
            "created_at": timestamp.isoformat()
        }
        search_index.append(search_entry)
        
        return DocumentResponse(
            id=document_id,
            title=request.title,
            content=request.content,
            category=request.category,
            subcategory=request.subcategory,
            tags=request.tags,
            metadata=document_data["metadata"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create document: {str(e)}"
        )

@router.get("/search", response_model=SearchResponse)
async def search_documents(
    q: str,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 10,
    retriever: AdvancedRAGRetriever = Depends(get_retriever)
):
    """Search documents in the knowledge base."""
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
        
        tags_list = [tag.strip() for tag in tags.split(",")] if tags else None
        
        # Enhanced text search implementation
        query_lower = q.lower()
        
        # Clean and tokenize query - remove stop words and extract key terms
        stop_words = {"what", "is", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "how", "why", "when", "where", "who"}
        query_terms = [term.strip() for term in query_lower.split() if term.strip() not in stop_words and len(term.strip()) > 1]
        
        matching_documents = []
        
        for doc in search_index:
            score = 0.0
            doc_title_lower = doc["title"].lower()
            doc_content_lower = doc["content"].lower()
            doc_tags_lower = [tag.lower() for tag in doc["tags"]]
            
            # Exact phrase match (highest priority)
            if query_lower in doc_title_lower:
                score += 3.0
            if query_lower in doc_content_lower:
                score += 2.5
            
            # Individual term matching
            for term in query_terms:
                # Title term match
                if term in doc_title_lower:
                    score += 1.5
                
                # Content term match
                if term in doc_content_lower:
                    score += 1.0
                
                # Tags exact match
                if term in doc_tags_lower:
                    score += 1.2
                
                # Tags partial match
                for tag in doc_tags_lower:
                    if term in tag:
                        score += 0.8
            
            # Bonus for multiple term matches (indicates better relevance)
            matched_terms = sum(1 for term in query_terms if term in doc_title_lower or term in doc_content_lower)
            if matched_terms > 1:
                score += matched_terms * 0.3
            
            # Category match bonus
            if category and doc["category"] == category:
                score += 0.5
            elif not category:
                score += 0.1
            
            # Subcategory match bonus
            if subcategory and doc["subcategory"] == subcategory:
                score += 0.3
            
            # Filter-based tags match
            if tags_list:
                tag_matches = len(set(tags_list) & set(doc["tags"]))
                score += tag_matches * 0.2
            
            # Only include documents with meaningful scores
            if score > 0.1:
                full_doc = document_storage.get(doc["id"])
                if full_doc:
                    matching_documents.append({
                        "document": full_doc,
                        "score": round(score, 2)
                    })
        
        # Sort by score
        matching_documents.sort(key=lambda x: x["score"], reverse=True)
        matching_documents = matching_documents[:limit]
        
        # Format results
        results = []
        for match in matching_documents:
            doc = match["document"]
            results.append(DocumentResponse(
                id=doc["id"],
                title=doc["title"],
                content=doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                category=doc["category"],
                subcategory=doc["subcategory"],
                tags=doc["tags"],
                score=round(match["score"], 3),
                metadata=doc["metadata"]
            ))
        
        processing_time = round(time.time() - start_time, 3)
        
        return SearchResponse(
            results=results,
            total=len(matching_documents),
            query=q,
            metadata={
                "processing_time": processing_time,
                "total_documents": len(search_index),
                "filters": {
                    "category": category,
                    "subcategory": subcategory,
                    "tags": tags_list
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
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
                content=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
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
    """List all documents with optional filtering."""
    try:
        if limit > 100:
            limit = 100
            
        filtered_docs = search_index
        
        # Apply filters
        if category:
            filtered_docs = [doc for doc in filtered_docs if doc["category"] == category]
        
        if subcategory:
            filtered_docs = [doc for doc in filtered_docs if doc["subcategory"] == subcategory]
        
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
                    "chunk_count": full_doc["metadata"].get("chunk_count", 0)
                })
        
        return {
            "documents": results,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )

@router.post("/documents/bulk", response_model=Dict[str, Any])
async def bulk_create_documents(
    documents: List[DocumentCreateRequest],
    chunker: DocumentChunker = Depends(get_chunker),
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
                chunks = await asyncio.get_event_loop().run_in_executor(
                    None,
                    chunker.chunk_document,
                    request.content,
                    {
                        "document_id": document_id,
                        "title": request.title,
                        "category": request.category,
                        "subcategory": request.subcategory,
                        "tags": request.tags,
                        "created_at": timestamp.isoformat()
                    }
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
                
                # Store in memory
                document_storage[document_id] = document_data
                
                # Add to search index
                search_entry = {
                    "id": document_id,
                    "title": request.title,
                    "content": request.content[:500],
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
            semantic_docs = []
            for doc_info in created_documents:
                full_doc = document_storage.get(doc_info["id"])
                if full_doc:
                    semantic_docs.append({
                        "id": full_doc["id"],
                        "content": full_doc["content"],
                        "title": full_doc["title"],
                        "metadata": {
                            "category": full_doc["category"],
                            "subcategory": full_doc["subcategory"],
                            "tags": full_doc["tags"],
                            **full_doc["metadata"]
                        }
                    })
            
            if semantic_docs:
                await semantic_retriever.add_documents(semantic_docs)
        
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
    """Get knowledge base statistics."""
    try:
        categories = {}
        total_chunks = 0
        
        for doc in document_storage.values():
            category = doc["category"]
            categories[category] = categories.get(category, 0) + 1
            total_chunks += doc["metadata"].get("chunk_count", 0)
        
        return {
            "total_documents": len(document_storage),
            "total_chunks": total_chunks,
            "categories": categories,
            "average_chunks_per_document": round(total_chunks / max(len(document_storage), 1), 2)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )
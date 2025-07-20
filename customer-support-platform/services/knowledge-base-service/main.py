"""
Knowledge Base Service for Customer Support Platform

This service handles document storage, vector search, and RAG (Retrieval-Augmented Generation)
operations using LangChain for the customer support platform.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.bm25 import BM25Retriever

import pinecone

from shared.config.settings import get_knowledge_base_service_settings
from shared.auth.jwt_handler import JWTHandler
from shared.database.models import KnowledgeBase
from shared.database.connection import get_database_session
from shared.monitoring.metrics import MetricsCollector

# Configuration
settings = get_knowledge_base_service_settings()

# FastAPI app
app = FastAPI(
    title="Knowledge Base Service",
    description="LangChain-based knowledge base and RAG service for customer support",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication
jwt_handler = JWTHandler(settings.jwt.secret_key, settings.jwt.algorithm)


# Pydantic models
class DocumentCreateRequest(BaseModel):
    """Request model for creating a document."""
    
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    category: str = Field(..., description="Document category")
    subcategory: Optional[str] = Field(None, description="Document subcategory")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DocumentSearchRequest(BaseModel):
    """Request model for searching documents."""
    
    query: str = Field(..., description="Search query")
    category: Optional[str] = Field(None, description="Filter by category")
    subcategory: Optional[str] = Field(None, description="Filter by subcategory")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    limit: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    include_content: bool = Field(default=True, description="Include full content in results")


class DocumentResponse(BaseModel):
    """Response model for document operations."""
    
    id: str
    title: str
    content: Optional[str] = None
    summary: Optional[str] = None
    category: str
    subcategory: Optional[str] = None
    tags: List[str]
    score: Optional[float] = None
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Response model for search operations."""
    
    results: List[DocumentResponse]
    total: int
    query: str
    metadata: Dict[str, Any] = {}


# Vector store management
class VectorStoreManager:
    """Manages vector store operations using LangChain."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.openai.embedding_model,
            openai_api_key=settings.openai.api_key
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Initialize vector store based on configuration
        self.vector_store = self._initialize_vector_store()
        
        # Initialize BM25 retriever for hybrid search
        self.bm25_retriever = None
        self.all_documents = []
    
    def _initialize_vector_store(self):
        """Initialize the vector store based on configuration."""
        
        if settings.vector_store.type == "pinecone":
            pinecone.init(
                api_key=settings.vector_store.pinecone_api_key,
                environment=settings.vector_store.pinecone_environment
            )
            
            # Create or get index
            index_name = settings.vector_store.pinecone_index_name
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=1536,  # OpenAI embedding dimension
                    metric="cosine"
                )
            
            return Pinecone.from_existing_index(
                index_name=index_name,
                embedding=self.embeddings
            )
        
        elif settings.vector_store.type == "weaviate":
            import weaviate
            
            client = weaviate.Client(
                url=settings.vector_store.weaviate_url,
                auth_client_secret=weaviate.AuthApiKey(
                    api_key=settings.vector_store.weaviate_api_key
                )
            )
            
            from langchain.vectorstores import Weaviate
            return Weaviate(client, "Document", "content", embedding=self.embeddings)
        
        elif settings.vector_store.type == "chroma":
            from langchain.vectorstores import Chroma
            return Chroma(embedding_function=self.embeddings)
        
        else:
            raise ValueError(f"Unsupported vector store type: {settings.vector_store.type}")
    
    def _update_bm25_retriever(self):
        """Update BM25 retriever with current documents."""
        if self.all_documents:
            self.bm25_retriever = BM25Retriever.from_documents(
                documents=self.all_documents,
                k=settings.max_search_results
            )
    
    async def add_document(
        self,
        document_id: str,
        title: str,
        content: str,
        category: str,
        subcategory: Optional[str] = None,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add a document to the vector store."""
        
        try:
            # Prepare metadata
            doc_metadata = {
                "id": document_id,
                "title": title,
                "category": category,
                "subcategory": subcategory,
                "tags": tags or [],
                **(metadata or {})
            }
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Create documents
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc_metadata,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                )
                documents.append(doc)
            
            # Add to vector store
            vector_ids = await self.vector_store.aadd_documents(documents)
            
            # Update BM25 retriever
            self.all_documents.extend(documents)
            self._update_bm25_retriever()
            
            logging.info(f"Added document {document_id} with {len(chunks)} chunks")
            return document_id
            
        except Exception as e:
            logging.error(f"Error adding document to vector store: {e}")
            raise
    
    async def search_documents(
        self,
        query: str,
        k: int = 10,
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search documents using hybrid retrieval."""
        
        try:
            # Build metadata filter
            metadata_filter = {}
            if category:
                metadata_filter["category"] = category
            if subcategory:
                metadata_filter["subcategory"] = subcategory
            if tags:
                metadata_filter["tags"] = {"$in": tags}
            
            # Vector search
            vector_results = await self.vector_store.asimilarity_search_with_score(
                query=query,
                k=k,
                filter=metadata_filter if metadata_filter else None
            )
            
            # Hybrid search if BM25 is available
            if self.bm25_retriever:
                # Create ensemble retriever
                vector_retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": k, "filter": metadata_filter}
                )
                
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[vector_retriever, self.bm25_retriever],
                    weights=[0.7, 0.3]  # Favor vector search
                )
                
                hybrid_results = await ensemble_retriever.aget_relevant_documents(query)
                
                # Combine and deduplicate results
                results = []
                seen_ids = set()
                
                for doc in hybrid_results:
                    doc_id = doc.metadata.get("id")
                    if doc_id and doc_id not in seen_ids:
                        results.append({
                            "id": doc_id,
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "score": getattr(doc, "score", 0.0)
                        })
                        seen_ids.add(doc_id)
                
                return results[:k]
            
            else:
                # Return vector search results
                return [
                    {
                        "id": doc.metadata.get("id"),
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score)
                    }
                    for doc, score in vector_results
                ]
                
        except Exception as e:
            logging.error(f"Error searching documents: {e}")
            raise
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector store."""
        
        try:
            # Delete from vector store
            await self.vector_store.adelete(ids=[document_id])
            
            # Remove from BM25 retriever documents
            self.all_documents = [
                doc for doc in self.all_documents
                if doc.metadata.get("id") != document_id
            ]
            self._update_bm25_retriever()
            
            logging.info(f"Deleted document {document_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting document: {e}")
            return False


# Service implementation
class KnowledgeBaseService:
    """Knowledge base service implementation."""
    
    def __init__(self):
        self.vector_manager = VectorStoreManager()
        self.metrics = MetricsCollector()
    
    async def create_document(
        self,
        title: str,
        content: str,
        category: str,
        subcategory: Optional[str] = None,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        db_session = None
    ) -> DocumentResponse:
        """Create a new document."""
        
        start_time = time.time()
        
        try:
            # Generate document ID
            document_id = str(uuid4())
            
            # Create database record
            kb_document = KnowledgeBase(
                id=document_id,
                title=title,
                content=content,
                category=category,
                subcategory=subcategory,
                tags=tags or [],
                metadata=metadata or {},
                vector_id=document_id,
                embedding_model=settings.openai.embedding_model
            )
            
            db_session.add(kb_document)
            await db_session.commit()
            
            # Add to vector store
            await self.vector_manager.add_document(
                document_id=document_id,
                title=title,
                content=content,
                category=category,
                subcategory=subcategory,
                tags=tags,
                metadata=metadata
            )
            
            # Record metrics
            self.metrics.record_document_created(
                category=category,
                processing_time=time.time() - start_time
            )
            
            return DocumentResponse(
                id=document_id,
                title=title,
                content=content,
                category=category,
                subcategory=subcategory,
                tags=tags or [],
                metadata=metadata or {}
            )
            
        except Exception as e:
            logging.error(f"Error creating document: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create document"
            )
    
    async def search_documents(
        self,
        query: str,
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        include_content: bool = True,
        db_session = None
    ) -> SearchResponse:
        """Search documents."""
        
        start_time = time.time()
        
        try:
            # Search in vector store
            vector_results = await self.vector_manager.search_documents(
                query=query,
                k=limit,
                category=category,
                subcategory=subcategory,
                tags=tags
            )
            
            # Get document IDs from vector results
            document_ids = [result["id"] for result in vector_results if result.get("id")]
            
            # Fetch full documents from database
            if document_ids:
                db_documents = await db_session.execute(
                    select(KnowledgeBase).where(KnowledgeBase.id.in_(document_ids))
                )
                db_docs_dict = {doc.id: doc for doc in db_documents.scalars().all()}
            else:
                db_docs_dict = {}
            
            # Build response
            results = []
            for vector_result in vector_results:
                doc_id = vector_result.get("id")
                db_doc = db_docs_dict.get(doc_id)
                
                if db_doc:
                    results.append(DocumentResponse(
                        id=db_doc.id,
                        title=db_doc.title,
                        content=db_doc.content if include_content else None,
                        summary=db_doc.summary,
                        category=db_doc.category,
                        subcategory=db_doc.subcategory,
                        tags=db_doc.tags,
                        score=vector_result.get("score"),
                        metadata=db_doc.metadata
                    ))
            
            # Record metrics
            self.metrics.record_search_performed(
                query=query,
                results_count=len(results),
                processing_time=time.time() - start_time
            )
            
            return SearchResponse(
                results=results,
                total=len(results),
                query=query,
                metadata={
                    "processing_time": time.time() - start_time,
                    "vector_results": len(vector_results)
                }
            )
            
        except Exception as e:
            logging.error(f"Error searching documents: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to search documents"
            )
    
    async def get_document(self, document_id: str, db_session = None) -> DocumentResponse:
        """Get a specific document."""
        
        try:
            document = await db_session.get(KnowledgeBase, document_id)
            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Document not found"
                )
            
            return DocumentResponse(
                id=document.id,
                title=document.title,
                content=document.content,
                summary=document.summary,
                category=document.category,
                subcategory=document.subcategory,
                tags=document.tags,
                metadata=document.metadata
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Error getting document: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get document"
            )
    
    async def delete_document(self, document_id: str, db_session = None) -> bool:
        """Delete a document."""
        
        try:
            # Delete from database
            document = await db_session.get(KnowledgeBase, document_id)
            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Document not found"
                )
            
            await db_session.delete(document)
            await db_session.commit()
            
            # Delete from vector store
            await self.vector_manager.delete_document(document_id)
            
            # Record metrics
            self.metrics.record_document_deleted(document_id)
            
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Error deleting document: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete document"
            )


# Initialize service
knowledge_base_service = KnowledgeBaseService()


# Dependencies
async def get_current_user(request: Request):
    """Get current user from JWT token."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header"
        )
    
    token = auth_header.split(" ")[1]
    try:
        payload = jwt_handler.decode_token(token)
        return payload
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


# API endpoints
@app.post("/documents", response_model=DocumentResponse)
async def create_document(
    request: DocumentCreateRequest,
    user: dict = Depends(get_current_user),
    db_session = Depends(get_database_session)
):
    """
    Create a new document in the knowledge base.
    
    This endpoint allows you to add new documents to the knowledge base.
    The document will be automatically chunked and embedded for vector search.
    """
    
    return await knowledge_base_service.create_document(
        title=request.title,
        content=request.content,
        category=request.category,
        subcategory=request.subcategory,
        tags=request.tags,
        metadata=request.metadata,
        db_session=db_session
    )


@app.get("/search", response_model=SearchResponse)
async def search_documents(
    q: str,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 10,
    include_content: bool = True,
    user: dict = Depends(get_current_user),
    db_session = Depends(get_database_session)
):
    """
    Search documents in the knowledge base.
    
    This endpoint performs hybrid search combining vector similarity and keyword matching.
    Results are ranked by relevance score.
    """
    
    # Parse tags if provided
    tags_list = tags.split(",") if tags else None
    
    return await knowledge_base_service.search_documents(
        query=q,
        category=category,
        subcategory=subcategory,
        tags=tags_list,
        limit=limit,
        include_content=include_content,
        db_session=db_session
    )


@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    user: dict = Depends(get_current_user),
    db_session = Depends(get_database_session)
):
    """
    Get a specific document by ID.
    
    Returns the complete document information including content and metadata.
    """
    
    return await knowledge_base_service.get_document(
        document_id=document_id,
        db_session=db_session
    )


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    user: dict = Depends(get_current_user),
    db_session = Depends(get_database_session)
):
    """
    Delete a document from the knowledge base.
    
    This removes the document from both the database and vector store.
    """
    
    success = await knowledge_base_service.delete_document(
        document_id=document_id,
        db_session=db_session
    )
    
    return {"message": "Document deleted successfully", "success": success}


@app.get("/categories")
async def get_categories(
    user: dict = Depends(get_current_user),
    db_session = Depends(get_database_session)
):
    """
    Get all available document categories.
    
    Returns a list of categories and their document counts.
    """
    
    try:
        result = await db_session.execute(
            select(KnowledgeBase.category, func.count(KnowledgeBase.id))
            .group_by(KnowledgeBase.category)
            .order_by(KnowledgeBase.category)
        )
        
        categories = [
            {"category": category, "count": count}
            for category, count in result.fetchall()
        ]
        
        return {"categories": categories}
        
    except Exception as e:
        logging.error(f"Error getting categories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get categories"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "knowledge-base-service"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level="info"
    )
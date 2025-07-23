from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import asyncio
from datetime import datetime
import json

@dataclass
class VectorDocument:
    """Document representation for vector storage."""
    id: str
    content: str
    title: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    namespace: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class VectorSearchResult:
    """Search result from vector database."""
    document: VectorDocument
    score: float
    metadata: Dict[str, Any]

class VectorDatabaseInterface(ABC):
    """Abstract interface for vector databases."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the vector database."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the vector database."""
        pass
    
    @abstractmethod
    async def create_index(self, name: str, dimension: int, metric: str = "cosine") -> bool:
        """Create a new index."""
        pass
    
    @abstractmethod
    async def delete_index(self, name: str) -> bool:
        """Delete an index."""
        pass
    
    @abstractmethod
    async def upsert_documents(self, index_name: str, documents: List[VectorDocument]) -> bool:
        """Insert or update documents in the index."""
        pass
    
    @abstractmethod
    async def search(self, 
                    index_name: str, 
                    query_vector: np.ndarray, 
                    top_k: int = 10,
                    filters: Optional[Dict[str, Any]] = None,
                    namespace: Optional[str] = None) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def delete_documents(self, index_name: str, document_ids: List[str]) -> bool:
        """Delete documents from the index."""
        pass
    
    @abstractmethod
    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get statistics about an index."""
        pass

class PineconeVectorDatabase(VectorDatabaseInterface):
    """Pinecone vector database implementation."""
    
    def __init__(self, api_key: str, environment: str):
        self.api_key = api_key
        self.environment = environment
        self.pinecone_client = None
        self.indexes = {}
        self.logger = logging.getLogger(__name__)
    
    async def connect(self) -> bool:
        """Connect to Pinecone."""
        try:
            import pinecone
            
            pinecone.init(
                api_key=self.api_key,
                environment=self.environment
            )
            
            self.pinecone_client = pinecone
            self.logger.info("Successfully connected to Pinecone")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Pinecone: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Pinecone."""
        self.pinecone_client = None
        self.indexes = {}
        self.logger.info("Disconnected from Pinecone")
    
    async def create_index(self, name: str, dimension: int, metric: str = "cosine") -> bool:
        """Create a Pinecone index."""
        try:
            if not self.pinecone_client:
                await self.connect()
            
            if name not in self.pinecone_client.list_indexes():
                self.pinecone_client.create_index(
                    name=name,
                    dimension=dimension,
                    metric=metric,
                    pods=1,
                    replicas=1,
                    pod_type="p1.x1"
                )
                
                # Wait for index to be ready
                await asyncio.sleep(10)
            
            self.indexes[name] = self.pinecone_client.Index(name)
            self.logger.info(f"Pinecone index '{name}' created/connected successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create Pinecone index '{name}': {e}")
            return False
    
    async def delete_index(self, name: str) -> bool:
        """Delete a Pinecone index."""
        try:
            if name in self.pinecone_client.list_indexes():
                self.pinecone_client.delete_index(name)
                
            if name in self.indexes:
                del self.indexes[name]
            
            self.logger.info(f"Pinecone index '{name}' deleted successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete Pinecone index '{name}': {e}")
            return False
    
    async def upsert_documents(self, index_name: str, documents: List[VectorDocument]) -> bool:
        """Upsert documents to Pinecone index."""
        try:
            if index_name not in self.indexes:
                self.logger.error(f"Index '{index_name}' not found")
                return False
            
            index = self.indexes[index_name]
            
            # Prepare vectors for upsert
            vectors = []
            for doc in documents:
                metadata = {
                    **doc.metadata,
                    "title": doc.title,
                    "content": doc.content[:1000],  # Limit content size for metadata
                    "namespace": doc.namespace or "default",
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                    "updated_at": doc.updated_at.isoformat() if doc.updated_at else None
                }
                
                vectors.append({
                    "id": doc.id,
                    "values": doc.embedding.tolist(),
                    "metadata": metadata
                })
            
            # Batch upsert (Pinecone handles large batches)
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                index.upsert(vectors=batch, namespace=documents[0].namespace or "default")
            
            self.logger.info(f"Successfully upserted {len(documents)} documents to Pinecone index '{index_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upsert documents to Pinecone index '{index_name}': {e}")
            return False
    
    async def search(self, 
                    index_name: str, 
                    query_vector: np.ndarray, 
                    top_k: int = 10,
                    filters: Optional[Dict[str, Any]] = None,
                    namespace: Optional[str] = None) -> List[VectorSearchResult]:
        """Search Pinecone index."""
        try:
            if index_name not in self.indexes:
                self.logger.error(f"Index '{index_name}' not found")
                return []
            
            index = self.indexes[index_name]
            
            # Perform search
            search_response = index.query(
                vector=query_vector.tolist(),
                top_k=top_k,
                include_metadata=True,
                namespace=namespace or "default",
                filter=filters
            )
            
            # Convert to VectorSearchResult
            results = []
            for match in search_response.matches:
                metadata = match.metadata
                
                # Reconstruct document
                document = VectorDocument(
                    id=match.id,
                    content=metadata.get("content", ""),
                    title=metadata.get("title", ""),
                    embedding=np.array([]),  # Don't store embeddings in results
                    metadata={k: v for k, v in metadata.items() 
                             if k not in ["content", "title", "namespace", "created_at", "updated_at"]},
                    namespace=metadata.get("namespace"),
                    created_at=datetime.fromisoformat(metadata["created_at"]) if metadata.get("created_at") else None,
                    updated_at=datetime.fromisoformat(metadata["updated_at"]) if metadata.get("updated_at") else None
                )
                
                results.append(VectorSearchResult(
                    document=document,
                    score=match.score,
                    metadata=metadata
                ))
            
            self.logger.info(f"Pinecone search returned {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search Pinecone index '{index_name}': {e}")
            return []
    
    async def delete_documents(self, index_name: str, document_ids: List[str]) -> bool:
        """Delete documents from Pinecone index."""
        try:
            if index_name not in self.indexes:
                self.logger.error(f"Index '{index_name}' not found")
                return False
            
            index = self.indexes[index_name]
            index.delete(ids=document_ids)
            
            self.logger.info(f"Successfully deleted {len(document_ids)} documents from Pinecone index '{index_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete documents from Pinecone index '{index_name}': {e}")
            return False
    
    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get Pinecone index statistics."""
        try:
            if index_name not in self.indexes:
                return {"error": f"Index '{index_name}' not found"}
            
            index = self.indexes[index_name]
            stats = index.describe_index_stats()
            
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {}
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get Pinecone index stats for '{index_name}': {e}")
            return {"error": str(e)}

class WeaviateVectorDatabase(VectorDatabaseInterface):
    """Weaviate vector database implementation."""
    
    def __init__(self, url: str, auth_config: Optional[Dict[str, Any]] = None):
        self.url = url
        self.auth_config = auth_config or {}
        self.client = None
        self.logger = logging.getLogger(__name__)
    
    async def connect(self) -> bool:
        """Connect to Weaviate."""
        try:
            import weaviate
            
            auth_client_secret = None
            if self.auth_config.get("api_key"):
                auth_client_secret = weaviate.AuthApiKey(api_key=self.auth_config["api_key"])
            
            self.client = weaviate.Client(
                url=self.url,
                auth_client_secret=auth_client_secret
            )
            
            # Test connection
            if self.client.is_ready():
                self.logger.info("Successfully connected to Weaviate")
                return True
            else:
                self.logger.error("Weaviate is not ready")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to Weaviate: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Weaviate."""
        self.client = None
        self.logger.info("Disconnected from Weaviate")
    
    async def create_index(self, name: str, dimension: int, metric: str = "cosine") -> bool:
        """Create a Weaviate class (index)."""
        try:
            if not self.client:
                await self.connect()
            
            class_name = name.capitalize()  # Weaviate class names should be capitalized
            
            # Check if class already exists
            if self.client.schema.exists(class_name):
                self.logger.info(f"Weaviate class '{class_name}' already exists")
                return True
            
            # Define class schema
            class_schema = {
                "class": class_name,
                "description": f"Document class for {name}",
                "vectorizer": "none",  # We provide our own vectors
                "vectorIndexConfig": {
                    "distance": metric
                },
                "properties": [
                    {
                        "name": "title",
                        "dataType": ["text"],
                        "description": "Document title"
                    },
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Document content"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["text"],
                        "description": "Additional metadata as JSON"
                    },
                    {
                        "name": "namespace",
                        "dataType": ["text"],
                        "description": "Document namespace"
                    },
                    {
                        "name": "createdAt",
                        "dataType": ["date"],
                        "description": "Creation timestamp"
                    }
                ]
            }
            
            self.client.schema.create_class(class_schema)
            self.logger.info(f"Weaviate class '{class_name}' created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create Weaviate class '{name}': {e}")
            return False
    
    async def delete_index(self, name: str) -> bool:
        """Delete a Weaviate class."""
        try:
            class_name = name.capitalize()
            
            if self.client.schema.exists(class_name):
                self.client.schema.delete_class(class_name)
            
            self.logger.info(f"Weaviate class '{class_name}' deleted successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete Weaviate class '{name}': {e}")
            return False
    
    async def upsert_documents(self, index_name: str, documents: List[VectorDocument]) -> bool:
        """Upsert documents to Weaviate."""
        try:
            class_name = index_name.capitalize()
            
            # Check if class exists
            if not self.client.schema.exists(class_name):
                await self.create_index(index_name, len(documents[0].embedding))
            
            # Prepare batch
            with self.client.batch as batch:
                batch.batch_size = 100
                
                for doc in documents:
                    properties = {
                        "title": doc.title,
                        "content": doc.content,
                        "metadata": json.dumps(doc.metadata),
                        "namespace": doc.namespace or "default",
                        "createdAt": doc.created_at.isoformat() if doc.created_at else datetime.now().isoformat()
                    }
                    
                    batch.add_data_object(
                        data_object=properties,
                        class_name=class_name,
                        uuid=doc.id,
                        vector=doc.embedding.tolist()
                    )
            
            self.logger.info(f"Successfully upserted {len(documents)} documents to Weaviate class '{class_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upsert documents to Weaviate class '{index_name}': {e}")
            return False
    
    async def search(self, 
                    index_name: str, 
                    query_vector: np.ndarray, 
                    top_k: int = 10,
                    filters: Optional[Dict[str, Any]] = None,
                    namespace: Optional[str] = None) -> List[VectorSearchResult]:
        """Search Weaviate index."""
        try:
            class_name = index_name.capitalize()
            
            # Build query
            query_builder = (
                self.client.query
                .get(class_name, ["title", "content", "metadata", "namespace", "createdAt"])
                .with_near_vector({"vector": query_vector.tolist()})
                .with_limit(top_k)
                .with_additional(["certainty", "id"])
            )
            
            # Add filters if provided
            if filters:
                where_filter = self._build_weaviate_filter(filters)
                if where_filter:
                    query_builder = query_builder.with_where(where_filter)
            
            # Add namespace filter
            if namespace:
                namespace_filter = {
                    "path": ["namespace"],
                    "operator": "Equal",
                    "valueText": namespace
                }
                query_builder = query_builder.with_where(namespace_filter)
            
            # Execute query
            result = query_builder.do()
            
            # Convert to VectorSearchResult
            results = []
            if "data" in result and "Get" in result["data"] and class_name in result["data"]["Get"]:
                for item in result["data"]["Get"][class_name]:
                    metadata = json.loads(item.get("metadata", "{}"))
                    
                    document = VectorDocument(
                        id=item["_additional"]["id"],
                        content=item.get("content", ""),
                        title=item.get("title", ""),
                        embedding=np.array([]),  # Don't store embeddings in results
                        metadata=metadata,
                        namespace=item.get("namespace"),
                        created_at=datetime.fromisoformat(item["createdAt"]) if item.get("createdAt") else None
                    )
                    
                    # Convert certainty to score (Weaviate uses certainty, we want similarity score)
                    certainty = item["_additional"].get("certainty", 0.0)
                    score = certainty  # Could transform this if needed
                    
                    results.append(VectorSearchResult(
                        document=document,
                        score=score,
                        metadata=metadata
                    ))
            
            self.logger.info(f"Weaviate search returned {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search Weaviate class '{index_name}': {e}")
            return []
    
    def _build_weaviate_filter(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build Weaviate where filter from filter dict."""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated filter building
        if not filters:
            return None
        
        conditions = []
        for key, value in filters.items():
            if isinstance(value, str):
                conditions.append({
                    "path": [key],
                    "operator": "Equal",
                    "valueText": value
                })
            elif isinstance(value, (int, float)):
                conditions.append({
                    "path": [key],
                    "operator": "Equal",
                    "valueNumber": value
                })
        
        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {
                "operator": "And",
                "operands": conditions
            }
        
        return None
    
    async def delete_documents(self, index_name: str, document_ids: List[str]) -> bool:
        """Delete documents from Weaviate."""
        try:
            class_name = index_name.capitalize()
            
            for doc_id in document_ids:
                self.client.data_object.delete(doc_id, class_name)
            
            self.logger.info(f"Successfully deleted {len(document_ids)} documents from Weaviate class '{class_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete documents from Weaviate class '{index_name}': {e}")
            return False
    
    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get Weaviate class statistics."""
        try:
            class_name = index_name.capitalize()
            
            # Get aggregate count
            result = (
                self.client.query
                .aggregate(class_name)
                .with_meta_count()
                .do()
            )
            
            count = 0
            if "data" in result and "Aggregate" in result["data"] and class_name in result["data"]["Aggregate"]:
                meta = result["data"]["Aggregate"][class_name][0].get("meta", {})
                count = meta.get("count", 0)
            
            return {
                "total_vector_count": count,
                "class_name": class_name,
                "exists": self.client.schema.exists(class_name)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get Weaviate class stats for '{index_name}': {e}")
            return {"error": str(e)}

class ChromaVectorDatabase(VectorDatabaseInterface):
    """ChromaDB vector database implementation for development."""
    
    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory
        self.client = None
        self.collections = {}
        self.logger = logging.getLogger(__name__)
    
    async def connect(self) -> bool:
        """Connect to ChromaDB."""
        try:
            import chromadb
            
            if self.persist_directory:
                self.client = chromadb.PersistentClient(path=self.persist_directory)
            else:
                self.client = chromadb.Client()
            
            self.logger.info("Successfully connected to ChromaDB")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to ChromaDB: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from ChromaDB."""
        self.client = None
        self.collections = {}
        self.logger.info("Disconnected from ChromaDB")
    
    async def create_index(self, name: str, dimension: int, metric: str = "cosine") -> bool:
        """Create a ChromaDB collection."""
        try:
            if not self.client:
                await self.connect()
            
            # Map metric names
            distance_mapping = {
                "cosine": "cosine",
                "euclidean": "l2",
                "dot": "ip"
            }
            
            distance_function = distance_mapping.get(metric, "cosine")
            
            try:
                collection = self.client.create_collection(
                    name=name,
                    metadata={"hnsw:space": distance_function}
                )
            except Exception:
                # Collection might already exist
                collection = self.client.get_collection(name)
            
            self.collections[name] = collection
            self.logger.info(f"ChromaDB collection '{name}' created/connected successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create ChromaDB collection '{name}': {e}")
            return False
    
    async def delete_index(self, name: str) -> bool:
        """Delete a ChromaDB collection."""
        try:
            self.client.delete_collection(name)
            
            if name in self.collections:
                del self.collections[name]
            
            self.logger.info(f"ChromaDB collection '{name}' deleted successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete ChromaDB collection '{name}': {e}")
            return False
    
    async def upsert_documents(self, index_name: str, documents: List[VectorDocument]) -> bool:
        """Upsert documents to ChromaDB collection."""
        try:
            if index_name not in self.collections:
                await self.create_index(index_name, len(documents[0].embedding))
            
            collection = self.collections[index_name]
            
            # Prepare data
            ids = [doc.id for doc in documents]
            embeddings = [doc.embedding.tolist() for doc in documents]
            documents_data = [doc.content for doc in documents]
            metadatas = []
            
            for doc in documents:
                metadata = {
                    **doc.metadata,
                    "title": doc.title,
                    "namespace": doc.namespace or "default",
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                    "updated_at": doc.updated_at.isoformat() if doc.updated_at else None
                }
                # ChromaDB metadata values must be strings, numbers, or booleans
                cleaned_metadata = {}
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        if v is not None:
                            cleaned_metadata[k] = v
                    else:
                        cleaned_metadata[k] = str(v)
                
                metadatas.append(cleaned_metadata)
            
            # Upsert to collection
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents_data,
                metadatas=metadatas
            )
            
            self.logger.info(f"Successfully upserted {len(documents)} documents to ChromaDB collection '{index_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upsert documents to ChromaDB collection '{index_name}': {e}")
            return False
    
    async def search(self, 
                    index_name: str, 
                    query_vector: np.ndarray, 
                    top_k: int = 10,
                    filters: Optional[Dict[str, Any]] = None,
                    namespace: Optional[str] = None) -> List[VectorSearchResult]:
        """Search ChromaDB collection."""
        try:
            if index_name not in self.collections:
                self.logger.error(f"Collection '{index_name}' not found")
                return []
            
            collection = self.collections[index_name]
            
            # Build where clause
            where = {}
            if filters:
                where.update(filters)
            if namespace:
                where["namespace"] = namespace
            
            # Perform search
            results = collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=top_k,
                where=where if where else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert to VectorSearchResult
            search_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    
                    document = VectorDocument(
                        id=results["ids"][0][i],
                        content=results["documents"][0][i],
                        title=metadata.get("title", ""),
                        embedding=np.array([]),  # Don't store embeddings in results
                        metadata={k: v for k, v in metadata.items() 
                                 if k not in ["title", "namespace", "created_at", "updated_at"]},
                        namespace=metadata.get("namespace"),
                        created_at=datetime.fromisoformat(metadata["created_at"]) if metadata.get("created_at") else None,
                        updated_at=datetime.fromisoformat(metadata["updated_at"]) if metadata.get("updated_at") else None
                    )
                    
                    # Convert distance to similarity score
                    distance = results["distances"][0][i] if results["distances"] else 1.0
                    score = 1.0 / (1.0 + distance)  # Convert distance to similarity
                    
                    search_results.append(VectorSearchResult(
                        document=document,
                        score=score,
                        metadata=metadata
                    ))
            
            self.logger.info(f"ChromaDB search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Failed to search ChromaDB collection '{index_name}': {e}")
            return []
    
    async def delete_documents(self, index_name: str, document_ids: List[str]) -> bool:
        """Delete documents from ChromaDB collection."""
        try:
            if index_name not in self.collections:
                self.logger.error(f"Collection '{index_name}' not found")
                return False
            
            collection = self.collections[index_name]
            collection.delete(ids=document_ids)
            
            self.logger.info(f"Successfully deleted {len(document_ids)} documents from ChromaDB collection '{index_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete documents from ChromaDB collection '{index_name}': {e}")
            return False
    
    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get ChromaDB collection statistics."""
        try:
            if index_name not in self.collections:
                return {"error": f"Collection '{index_name}' not found"}
            
            collection = self.collections[index_name]
            count = collection.count()
            
            return {
                "total_vector_count": count,
                "collection_name": index_name
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get ChromaDB collection stats for '{index_name}': {e}")
            return {"error": str(e)}

class VectorDatabaseManager:
    """Manager for different vector database implementations."""
    
    def __init__(self):
        self.databases: Dict[str, VectorDatabaseInterface] = {}
        self.active_database: Optional[str] = None
        self.logger = logging.getLogger(__name__)
    
    def register_database(self, name: str, database: VectorDatabaseInterface):
        """Register a vector database implementation."""
        self.databases[name] = database
        self.logger.info(f"Registered vector database: {name}")
    
    def set_active_database(self, name: str) -> bool:
        """Set the active vector database."""
        if name in self.databases:
            self.active_database = name
            self.logger.info(f"Set active vector database: {name}")
            return True
        else:
            self.logger.error(f"Vector database '{name}' not found")
            return False
    
    def get_active_database(self) -> Optional[VectorDatabaseInterface]:
        """Get the active vector database instance."""
        if self.active_database:
            return self.databases.get(self.active_database)
        return None
    
    async def initialize_production_setup(self, config: Dict[str, Any]) -> bool:
        """Initialize production vector database setup."""
        try:
            # Setup based on configuration
            db_type = config.get("type", "chroma")
            
            if db_type == "pinecone":
                pinecone_db = PineconeVectorDatabase(
                    api_key=config["api_key"],
                    environment=config["environment"]
                )
                self.register_database("pinecone", pinecone_db)
                await pinecone_db.connect()
                self.set_active_database("pinecone")
                
            elif db_type == "weaviate":
                weaviate_db = WeaviateVectorDatabase(
                    url=config["url"],
                    auth_config=config.get("auth", {})
                )
                self.register_database("weaviate", weaviate_db)
                await weaviate_db.connect()
                self.set_active_database("weaviate")
                
            elif db_type == "chroma":
                chroma_db = ChromaVectorDatabase(
                    persist_directory=config.get("persist_directory")
                )
                self.register_database("chroma", chroma_db)
                await chroma_db.connect()
                self.set_active_database("chroma")
            
            self.logger.info(f"Successfully initialized {db_type} vector database")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector database: {e}")
            return False
    
    async def get_database_info(self) -> Dict[str, Any]:
        """Get information about registered databases."""
        info = {
            "registered_databases": list(self.databases.keys()),
            "active_database": self.active_database,
            "database_status": {}
        }
        
        for name, db in self.databases.items():
            try:
                # Try to get some basic info to test connectivity
                if hasattr(db, 'client') and db.client:
                    info["database_status"][name] = "connected"
                else:
                    info["database_status"][name] = "disconnected"
            except Exception:
                info["database_status"][name] = "error"
        
        return info
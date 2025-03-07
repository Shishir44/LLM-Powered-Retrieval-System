from typing import Dict, List, Optional
import os
import json
from llama_index.core.vector_stores.types import (
    MetadataFilter, 
    MetadataFilters,
    FilterOperator,
    FilterCondition,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult
)
from llama_index.core.schema import TextNode, NodeWithScore, NodeRelationship, RelatedNodeInfo, ObjectType

class CustomVectorStore(VectorStore):
    def __init__(self):
        self._data: Dict[str, dict] = {}
        self._metadata: Dict[str, dict] = {}
        self.stores_text = True  # Since we store text in self._data

    @property
    def stores_text(self) -> bool:
        """Flag to indicate if the vector store stores text."""
        return True

    @stores_text.setter 
    def stores_text(self, value: bool):
        """Setter for stores_text property."""
        # We always store text, so this is just to satisfy the interface
        pass

    def query(
        self,
        query: VectorStoreQuery,
    ) -> VectorStoreQueryResult:
        """Query the vector store."""
        print("\nExecuting vector store query...")
        print(f"Vector store size: {len(self._data)}")
        print(f"Query filters: {query.filters}")
        
        # Filter nodes based on metadata
        candidate_ids = list(self._data.keys())
        if query.filters:
            filtered_ids = []
            for node_id in candidate_ids:
                if self.evaluate_filters(self._metadata[node_id], query.filters):
                    filtered_ids.append(node_id)
                    print(f"Node {node_id} passed filters")
            candidate_ids = filtered_ids

        print(f"Candidates after filtering: {len(candidate_ids)}")

        # Compute similarities
        similarities = []
        query_embedding = query.query_embedding
        for node_id in candidate_ids:
            vector = self._data[node_id]["vector"]
            if isinstance(vector, list):
                vector = vector
            similarity = self.compute_similarity(query_embedding, vector)
            similarities.append((node_id, similarity))

        # Sort by similarity and take top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k_results = similarities[:query.similarity_top_k]

        # Create response nodes
        nodes = []
        similarities_list = []
        ids_list = []
        
        for node_id, score in top_k_results:
            # Create base node
            base_node = TextNode(
                text=self._data[node_id]["text"],
                metadata=self._metadata[node_id],
                embedding=self._data[node_id]["vector"],
                id_=node_id,
                relationships={},
            )
            
            # Simply append the node without wrapping in NodeWithScore
            nodes.append(base_node)
            similarities_list.append(score)
            ids_list.append(node_id)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities_list,
            ids=ids_list
        )

    def add(self, nodes: List[TextNode]) -> List[str]:
        """Add nodes to the vector store."""
        print("\nAdding nodes to vector store...")
        ids = []
        for node in nodes:
            print(f"Adding node {node.id_}")
            print(f"Node metadata: {node.metadata}")
            
            self._data[node.id_] = {
                "vector": node.embedding,
                "text": node.text,
                "node_type": ObjectType.TEXT,  # Store node type
            }
            self._metadata[node.id_] = node.metadata
            ids.append(node.id_)
            
        print(f"Vector store size after addition: {len(self._data)}")
        return ids

    def delete(self, node_ids: List[str]) -> None:
        """Delete nodes from the vector store."""
        for node_id in node_ids:
            if node_id in self._data:
                del self._data[node_id]
            if node_id in self._metadata:
                del self._metadata[node_id]

    def persist(self, persist_path: str, fs=None):
        """Persist vector store to disk."""
        if not os.path.exists(persist_path):
            os.makedirs(persist_path)
            
        vector_file = os.path.join(persist_path, "vectors.json")
        metadata_file = os.path.join(persist_path, "metadata.json")
        
        try:
            # Convert vectors to lists for JSON serialization
            serializable_data = {}
            for node_id, node_data in self._data.items():
                serializable_data[node_id] = {
                    "vector": node_data["vector"] if isinstance(node_data["vector"], list) else node_data["vector"].tolist(),
                    "text": node_data["text"]
                }
            
            with open(vector_file, 'w') as f:
                json.dump(serializable_data, f)
                
            with open(metadata_file, 'w') as f:
                json.dump(self._metadata, f)
                
            print(f"Persisted {len(self._data)} vectors to {persist_path}")
            
        except Exception as e:
            print(f"Error persisting vector store: {str(e)}")
            raise

    def load(self, persist_path: str, fs=None):
        """Load vector store from disk."""
        vector_file = os.path.join(persist_path, "vectors.json")
        metadata_file = os.path.join(persist_path, "metadata.json")
        
        if os.path.exists(vector_file) and os.path.exists(metadata_file):
            try:
                with open(vector_file, 'r') as f:
                    self._data = json.load(f)
                    
                with open(metadata_file, 'r') as f:
                    self._metadata = json.load(f)
                    
                print(f"Loaded {len(self._data)} vectors from {persist_path}")
            except Exception as e:
                print(f"Error loading vector store: {str(e)}")
                self._data = {}
                self._metadata = {}
                raise
        else:
            print(f"No existing vector store found in {persist_path}")
            self._data = {}
            self._metadata = {}

    def evaluate_filters(self, metadata: Dict, filters: MetadataFilters) -> bool:
        """Evaluate metadata filters."""
        if not filters.filters:
            return True

        results = []
        for filter_item in filters.filters:
            if isinstance(filter_item, MetadataFilters):
                result = self.evaluate_filters(metadata, filter_item)
            else:
                result = self.evaluate_filter(metadata, filter_item)
            results.append(result)

        if filters.condition == FilterCondition.AND:
            return all(results)
        else:
            return any(results)

    def evaluate_filter(self, metadata: Dict, filter_item: MetadataFilter) -> bool:
        """
        Evaluate a single metadata filter against a node's metadata.
        
        Args:
            metadata (Dict): The metadata dictionary of the node being evaluated
            filter_item (MetadataFilter): The filter to apply, containing key, value, and operator
            
        Returns:
            bool: True if the metadata matches the filter criteria, False otherwise
        """
        # First check if the filter key exists in metadata
        if filter_item.key not in metadata:
            return False
            
        # Get the value from metadata that we'll compare against
        metadata_value = metadata[filter_item.key]
        filter_value = filter_item.value
        
        try:
            # Handle different operator types
            if filter_item.operator == FilterOperator.EQ or filter_item.operator == "==":
                # Direct equality comparison
                return metadata_value == filter_value
                
            elif filter_item.operator == FilterOperator.NE or filter_item.operator == "!=":
                # Not equal comparison
                return metadata_value != filter_value
                
            elif filter_item.operator == FilterOperator.GT or filter_item.operator == ">":
                # Greater than comparison
                # Convert to float if both values are numeric strings
                if isinstance(metadata_value, str) and isinstance(filter_value, str):
                    if metadata_value.replace('.','',1).isdigit() and filter_value.replace('.','',1).isdigit():
                        return float(metadata_value) > float(filter_value)
                return metadata_value > filter_value
                
            elif filter_item.operator == FilterOperator.GTE or filter_item.operator == ">=":
                # Greater than or equal comparison
                if isinstance(metadata_value, str) and isinstance(filter_value, str):
                    if metadata_value.replace('.','',1).isdigit() and filter_value.replace('.','',1).isdigit():
                        return float(metadata_value) >= float(filter_value)
                return metadata_value >= filter_value
                
            elif filter_item.operator == FilterOperator.LT or filter_item.operator == "<":
                # Less than comparison
                if isinstance(metadata_value, str) and isinstance(filter_value, str):
                    if metadata_value.replace('.','',1).isdigit() and filter_value.replace('.','',1).isdigit():
                        return float(metadata_value) < float(filter_value)
                return metadata_value < filter_value
                
            elif filter_item.operator == FilterOperator.LTE or filter_item.operator == "<=":
                # Less than or equal comparison
                if isinstance(metadata_value, str) and isinstance(filter_value, str):
                    if metadata_value.replace('.','',1).isdigit() and filter_value.replace('.','',1).isdigit():
                        return float(metadata_value) <= float(filter_value)
                return metadata_value <= filter_value
                
            elif filter_item.operator == FilterOperator.CONTAINS or filter_item.operator == "contains":
                # Handle CONTAINS operator for different data types
                if isinstance(metadata_value, (list, tuple)):
                    # If metadata value is a list/tuple, check if filter value is in it
                    return filter_value in metadata_value
                elif isinstance(metadata_value, str) and isinstance(filter_value, str):
                    # If both are strings, check if filter value is a substring
                    return filter_value in metadata_value
                elif isinstance(metadata_value, dict):
                    # If metadata value is a dict, check if filter value is a key
                    return filter_value in metadata_value
                return False
                
            elif filter_item.operator == "IN" or filter_item.operator == "in":
                # Handle IN operator (new addition)
                if isinstance(filter_value, (list, tuple)):
                    # If filter value is a list/tuple, check if metadata value is in it
                    return metadata_value in filter_value
                elif isinstance(metadata_value, (list, tuple)):
                    # If metadata value is a list/tuple, check if filter value is in it
                    return filter_value in metadata_value
                else:
                    # For non-collections, fall back to equality check
                    return metadata_value == filter_value
                    
            else:
                # Log unsupported operator for debugging
                print(f"Warning: Unsupported operator {filter_item.operator}")
                return False
            
        except Exception as e:
            # Log any errors during comparison for debugging
            print(f"Error evaluating filter: {str(e)}")
            print(f"Metadata value: {metadata_value} ({type(metadata_value)})")
            print(f"Filter value: {filter_value} ({type(filter_value)})")
            print(f"Operator: {filter_item.operator}")
            return False

    def compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0
            
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
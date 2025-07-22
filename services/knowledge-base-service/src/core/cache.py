from typing import Dict, Any, Optional, List
import json
import hashlib
import time
from dataclasses import dataclass, asdict
from collections import OrderedDict
import asyncio

@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    ttl: float
    access_count: int = 0
    last_accessed: float = None
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl <= 0:  # Never expires
            return False
        return time.time() - self.created_at > self.ttl
    
    def access(self):
        """Mark entry as accessed."""
        self.access_count += 1
        self.last_accessed = time.time()

class VectorCache:
    """In-memory cache with TTL, LRU eviction, and statistics."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0
        }
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate a cache key from data."""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True, default=str)
        
        hash_obj = hashlib.md5(content.encode())
        return f"{prefix}:{hash_obj.hexdigest()[:16]}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            self.stats["misses"] += 1
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if entry.is_expired:
            self.delete(key)
            self.stats["misses"] += 1
            return None
        
        # Update access stats
        entry.access()
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        
        self.stats["hits"] += 1
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        ttl = ttl if ttl is not None else self.default_ttl
        current_time = time.time()
        
        # If key exists, update it
        if key in self.cache:
            entry = self.cache[key]
            entry.value = value
            entry.created_at = current_time
            entry.ttl = ttl
            entry.last_accessed = current_time
        else:
            # Check if we need to evict
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                ttl=ttl,
                last_accessed=current_time
            )
            self.cache[key] = entry
        
        # Move to end
        self.cache.move_to_end(key)
        self.stats["sets"] += 1
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if key in self.cache:
            del self.cache[key]
            self.stats["deletes"] += 1
            return True
        return False
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self.cache:
            # Remove the first (oldest) entry
            self.cache.popitem(last=False)
            self.stats["evictions"] += 1
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count."""
        expired_keys = []
        current_time = time.time()
        
        for key, entry in self.cache.items():
            if entry.is_expired:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.delete(key)
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.stats["hits"] / max(self.stats["hits"] + self.stats["misses"], 1)
        
        return {
            **self.stats,
            "hit_rate": round(hit_rate, 3),
            "size": len(self.cache),
            "max_size": self.max_size,
            "fill_percentage": round(len(self.cache) / self.max_size * 100, 1)
        }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Estimate memory usage of cache."""
        total_size = 0
        
        for entry in self.cache.values():
            # Rough estimation
            total_size += len(str(entry.key)) + len(str(entry.value))
        
        return {
            "estimated_bytes": total_size,
            "estimated_mb": round(total_size / (1024 * 1024), 2),
            "entries": len(self.cache)
        }
    
    # Specific cache methods for common use cases
    
    def cache_search_results(self, query: str, filters: Dict[str, Any], results: List[Any], ttl: float = 300) -> None:
        """Cache search results."""
        key = self._generate_key("search", {"query": query, "filters": filters})
        self.set(key, results, ttl)
    
    def get_cached_search_results(self, query: str, filters: Dict[str, Any]) -> Optional[List[Any]]:
        """Get cached search results."""
        key = self._generate_key("search", {"query": query, "filters": filters})
        return self.get(key)
    
    def cache_document_chunks(self, document_id: str, chunks: List[Any], ttl: float = 3600) -> None:
        """Cache document chunks."""
        key = self._generate_key("chunks", document_id)
        self.set(key, chunks, ttl)
    
    def get_cached_document_chunks(self, document_id: str) -> Optional[List[Any]]:
        """Get cached document chunks."""
        key = self._generate_key("chunks", document_id)
        return self.get(key)
    
    def cache_embeddings(self, text: str, embeddings: List[float], ttl: float = 86400) -> None:
        """Cache text embeddings."""
        key = self._generate_key("embeddings", text)
        self.set(key, embeddings, ttl)
    
    def get_cached_embeddings(self, text: str) -> Optional[List[float]]:
        """Get cached text embeddings."""
        key = self._generate_key("embeddings", text)
        return self.get(key)
    
    def cache_rag_context(self, query: str, context: Dict[str, Any], ttl: float = 600) -> None:
        """Cache RAG context for queries."""
        key = self._generate_key("rag_context", query)
        self.set(key, context, ttl)
    
    def get_cached_rag_context(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached RAG context."""
        key = self._generate_key("rag_context", query)
        return self.get(key)
    
    def invalidate_document_cache(self, document_id: str) -> int:
        """Invalidate all cache entries related to a document."""
        keys_to_delete = []
        
        for key in self.cache.keys():
            if f":{document_id}" in key or document_id in key:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            self.delete(key)
        
        return len(keys_to_delete)


# Global cache instance
vector_cache = VectorCache()
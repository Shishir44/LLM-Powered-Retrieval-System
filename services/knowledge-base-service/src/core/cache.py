"""
Enhanced Vector Cache for Large-Scale RAG Performance
Provides query result caching, embedding caching, and batch operations.
"""
import json
import hashlib
import time
import logging
from typing import Dict, List, Optional, Any, Set
import redis
import pickle
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Cache performance statistics."""
    query_hits: int = 0
    query_misses: int = 0
    embedding_hits: int = 0
    embedding_misses: int = 0
    total_operations: int = 0
    cache_size_mb: float = 0.0
    
    @property
    def query_hit_rate(self) -> float:
        total_queries = self.query_hits + self.query_misses
        return self.query_hits / total_queries if total_queries > 0 else 0.0
    
    @property
    def embedding_hit_rate(self) -> float:
        total_embeddings = self.embedding_hits + self.embedding_misses
        return self.embedding_hits / total_embeddings if total_embeddings > 0 else 0.0

class VectorCache:
    """Enhanced caching system for large-scale RAG operations."""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, 
                 redis_db: int = 0, redis_password: str = None):
        """Initialize the cache with Redis backend."""
        self.logger = logging.getLogger(__name__)
        self.stats = CacheStats()
        
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=False,  # Handle binary data
                socket_connect_timeout=5,
                socket_timeout=5,
                connection_pool_class_kwargs={
                    'max_connections': 20,  # Connection pool for performance
                }
            )
            # Test connection
            self.redis_client.ping()
            self.logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
            
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
            self.redis_client = None
            self._memory_cache = {}
    
    def _get_cache_key(self, prefix: str, data: str) -> str:
        """Generate consistent cache key."""
        hash_obj = hashlib.sha256(data.encode('utf-8'))
        return f"{prefix}:{hash_obj.hexdigest()[:16]}"
    
    async def get_query_results(self, query: str, top_k: int = 5, filters: Dict = None) -> Optional[List[Dict]]:
        """Get cached query results."""
        self.stats.total_operations += 1
        
        # Create cache key based on query parameters
        cache_data = json.dumps({
            "query": query.strip().lower(),
            "top_k": top_k,
            "filters": filters or {}
        }, sort_keys=True)
        
        cache_key = self._get_cache_key("query", cache_data)
        
        try:
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    results = pickle.loads(cached_data)
                    self.stats.query_hits += 1
                    self.logger.debug(f"Query cache hit for: {query[:50]}...")
                    return results
            else:
                # In-memory fallback
                if cache_key in self._memory_cache:
                    cache_entry = self._memory_cache[cache_key]
                    if time.time() - cache_entry['timestamp'] < 3600:  # 1 hour TTL
                        self.stats.query_hits += 1
                        return cache_entry['data']
            
            self.stats.query_misses += 1
            return None
            
        except Exception as e:
            self.logger.error(f"Cache retrieval error: {e}")
            self.stats.query_misses += 1
            return None
    
    async def cache_query_results(self, query: str, top_k: int, filters: Dict, results: List[Dict], 
                                ttl: int = 3600) -> bool:
        """Cache query results with TTL."""
        if not results:
            return False
            
        try:
            cache_data = json.dumps({
                "query": query.strip().lower(),
                "top_k": top_k,
                "filters": filters or {}
            }, sort_keys=True)
            
            cache_key = self._get_cache_key("query", cache_data)
            
            if self.redis_client:
                serialized_results = pickle.dumps(results)
                self.redis_client.setex(cache_key, ttl, serialized_results)
            else:
                # In-memory fallback with manual TTL
                self._memory_cache[cache_key] = {
                    'data': results,
                    'timestamp': time.time()
                }
                # Cleanup old entries periodically
                await self._cleanup_memory_cache()
            
            self.logger.debug(f"Cached query results for: {query[:50]}...")
            return True
            
        except Exception as e:
            self.logger.error(f"Cache storage error: {e}")
            return False
    
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text."""
        self.stats.total_operations += 1
        
        cache_key = self._get_cache_key("embedding", text.strip())
        
        try:
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    embedding = pickle.loads(cached_data)
                    self.stats.embedding_hits += 1
                    return embedding
            else:
                if cache_key in self._memory_cache:
                    cache_entry = self._memory_cache[cache_key]
                    if time.time() - cache_entry['timestamp'] < 86400:  # 24 hour TTL for embeddings
                        self.stats.embedding_hits += 1
                        return cache_entry['data']
            
            self.stats.embedding_misses += 1
            return None
            
        except Exception as e:
            self.logger.error(f"Embedding cache retrieval error: {e}")
            self.stats.embedding_misses += 1
            return None
    
    async def cache_embedding(self, text: str, embedding: List[float], ttl: int = 86400) -> bool:
        """Cache embedding with 24-hour TTL by default."""
        try:
            cache_key = self._get_cache_key("embedding", text.strip())
            
            if self.redis_client:
                serialized_embedding = pickle.dumps(embedding)
                self.redis_client.setex(cache_key, ttl, serialized_embedding)
            else:
                self._memory_cache[cache_key] = {
                    'data': embedding,
                    'timestamp': time.time()
                }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Embedding cache storage error: {e}")
            return False
    
    async def batch_get_embeddings(self, texts: List[str]) -> Dict[str, Optional[List[float]]]:
        """Get multiple embeddings from cache efficiently."""
        results = {}
        
        if self.redis_client:
            # Use Redis pipeline for efficient batch operations
            pipeline = self.redis_client.pipeline()
            cache_keys = []
            
            for text in texts:
                cache_key = self._get_cache_key("embedding", text.strip())
                cache_keys.append((text, cache_key))
                pipeline.get(cache_key)
            
            try:
                cached_data = pipeline.execute()
                
                for (text, cache_key), data in zip(cache_keys, cached_data):
                    if data:
                        try:
                            embedding = pickle.loads(data)
                            results[text] = embedding
                            self.stats.embedding_hits += 1
                        except:
                            results[text] = None
                            self.stats.embedding_misses += 1
                    else:
                        results[text] = None
                        self.stats.embedding_misses += 1
                        
            except Exception as e:
                self.logger.error(f"Batch embedding retrieval error: {e}")
                for text in texts:
                    results[text] = None
                    self.stats.embedding_misses += 1
        else:
            # Fallback to individual lookups for in-memory cache
            for text in texts:
                results[text] = await self.get_embedding(text)
        
        return results
    
    async def batch_cache_embeddings(self, text_embedding_pairs: List[tuple], ttl: int = 86400) -> bool:
        """Cache multiple embeddings efficiently."""
        if not text_embedding_pairs:
            return True
            
        try:
            if self.redis_client:
                # Use Redis pipeline for efficient batch operations
                pipeline = self.redis_client.pipeline()
                
                for text, embedding in text_embedding_pairs:
                    cache_key = self._get_cache_key("embedding", text.strip())
                    serialized_embedding = pickle.dumps(embedding)
                    pipeline.setex(cache_key, ttl, serialized_embedding)
                
                pipeline.execute()
                self.logger.debug(f"Batch cached {len(text_embedding_pairs)} embeddings")
            else:
                # In-memory fallback
                timestamp = time.time()
                for text, embedding in text_embedding_pairs:
                    cache_key = self._get_cache_key("embedding", text.strip())
                    self._memory_cache[cache_key] = {
                        'data': embedding,
                        'timestamp': timestamp
                    }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Batch embedding cache error: {e}")
            return False
    
    async def _cleanup_memory_cache(self):
        """Clean up expired entries from memory cache."""
        if not hasattr(self, '_memory_cache'):
            return
            
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._memory_cache.items():
            # Different TTLs for different types
            if 'embedding' in key:
                ttl = 86400  # 24 hours
            else:
                ttl = 3600   # 1 hour
                
            if current_time - entry['timestamp'] > ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._memory_cache[key]
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def get_cache_stats(self) -> CacheStats:
        """Get comprehensive cache statistics."""
        if self.redis_client:
            try:
                # Get Redis memory usage
                info = self.redis_client.info('memory')
                self.stats.cache_size_mb = info.get('used_memory', 0) / (1024 * 1024)
            except:
                pass
        else:
            # Estimate memory cache size
            import sys
            total_size = sum(sys.getsizeof(v) for v in self._memory_cache.values())
            self.stats.cache_size_mb = total_size / (1024 * 1024)
        
        return self.stats
    
    async def clear_cache(self, pattern: str = None) -> bool:
        """Clear cache entries, optionally by pattern."""
        try:
            if self.redis_client:
                if pattern:
                    keys = self.redis_client.keys(f"*{pattern}*")
                    if keys:
                        self.redis_client.delete(*keys)
                        self.logger.info(f"Cleared {len(keys)} cache entries matching '{pattern}'")
                else:
                    self.redis_client.flushdb()
                    self.logger.info("Cleared all cache entries")
            else:
                if pattern:
                    keys_to_delete = [k for k in self._memory_cache.keys() if pattern in k]
                    for key in keys_to_delete:
                        del self._memory_cache[key]
                    self.logger.info(f"Cleared {len(keys_to_delete)} cache entries matching '{pattern}'")
                else:
                    self._memory_cache.clear()
                    self.logger.info("Cleared all memory cache entries")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Cache clearing error: {e}")
            return False
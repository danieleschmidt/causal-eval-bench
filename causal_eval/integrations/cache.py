"""
Caching integration using Redis for performance optimization.
"""

import os
import json
import asyncio
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
import hashlib
import logging

import redis.asyncio as redis
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class CacheManager:
    """Abstract cache manager interface."""
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        raise NotImplementedError
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        raise NotImplementedError
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        raise NotImplementedError
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        raise NotImplementedError


class RedisCache(CacheManager):
    """Redis-based cache implementation."""
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 3600,
        prefix: str = "causal_eval:"
    ):
        """Initialize Redis cache."""
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.default_ttl = default_ttl
        self.prefix = prefix
        self.client: Optional[Redis] = None
    
    async def connect(self):
        """Connect to Redis."""
        if not self.client:
            try:
                self.client = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                # Test connection
                await self.client.ping()
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.client:
            await self.client.close()
            self.client = None
            logger.info("Disconnected from Redis cache")
    
    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self.client:
            await self.connect()
        
        try:
            cached_value = await self.client.get(self._make_key(key))
            if cached_value:
                return json.loads(cached_value)
            return None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        if not self.client:
            await self.connect()
        
        try:
            serialized_value = json.dumps(value, default=str)
            ttl = ttl or self.default_ttl
            
            await self.client.setex(
                self._make_key(key),
                ttl,
                serialized_value
            )
            return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        if not self.client:
            await self.connect()
        
        try:
            result = await self.client.delete(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        if not self.client:
            await self.connect()
        
        try:
            result = await self.client.exists(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries with prefix."""
        if not self.client:
            await self.connect()
        
        try:
            pattern = f"{self.prefix}*"
            keys = await self.client.keys(pattern)
            if keys:
                await self.client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a numeric value in cache."""
        if not self.client:
            await self.connect()
        
        try:
            return await self.client.incrby(self._make_key(key), amount)
        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.client:
            await self.connect()
        
        try:
            info = await self.client.info()
            pattern = f"{self.prefix}*"
            keys = await self.client.keys(pattern)
            
            return {
                "total_keys": len(keys),
                "memory_usage": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                )
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate."""
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0


class EvaluationCache:
    """Specialized cache for evaluation results."""
    
    def __init__(self, cache_manager: CacheManager):
        """Initialize evaluation cache."""
        self.cache = cache_manager
    
    def _make_evaluation_key(
        self,
        task_type: str,
        domain: str,
        difficulty: str,
        prompt_hash: str
    ) -> str:
        """Create cache key for evaluation."""
        return f"eval:{task_type}:{domain}:{difficulty}:{prompt_hash}"
    
    def _hash_prompt(self, prompt: str) -> str:
        """Create hash for prompt content."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
    
    async def get_evaluation_result(
        self,
        task_type: str,
        domain: str,
        difficulty: str,
        prompt: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached evaluation result."""
        prompt_hash = self._hash_prompt(prompt)
        key = self._make_evaluation_key(task_type, domain, difficulty, prompt_hash)
        
        result = await self.cache.get(key)
        if result:
            logger.info(f"Cache hit for evaluation: {task_type}/{domain}/{difficulty}")
        
        return result
    
    async def cache_evaluation_result(
        self,
        task_type: str,
        domain: str,
        difficulty: str,
        prompt: str,
        result: Dict[str, Any],
        ttl: int = 86400  # 24 hours
    ) -> bool:
        """Cache evaluation result."""
        prompt_hash = self._hash_prompt(prompt)
        key = self._make_evaluation_key(task_type, domain, difficulty, prompt_hash)
        
        # Add cache metadata
        cached_result = {
            **result,
            "_cached_at": datetime.utcnow().isoformat(),
            "_cache_key": key
        }
        
        success = await self.cache.set(key, cached_result, ttl)
        if success:
            logger.info(f"Cached evaluation result: {task_type}/{domain}/{difficulty}")
        
        return success
    
    async def invalidate_domain_cache(self, domain: str) -> int:
        """Invalidate all cached results for a domain."""
        if isinstance(self.cache, RedisCache):
            pattern = f"eval:*:{domain}:*"
            try:
                if not self.cache.client:
                    await self.cache.connect()
                
                keys = await self.cache.client.keys(f"{self.cache.prefix}{pattern}")
                if keys:
                    await self.cache.client.delete(*keys)
                    logger.info(f"Invalidated {len(keys)} cache entries for domain: {domain}")
                    return len(keys)
            except Exception as e:
                logger.error(f"Failed to invalidate domain cache: {e}")
        
        return 0


class PromptCache:
    """Specialized cache for generated prompts."""
    
    def __init__(self, cache_manager: CacheManager):
        """Initialize prompt cache."""
        self.cache = cache_manager
    
    def _make_prompt_key(self, task_type: str, domain: str, difficulty: str) -> str:
        """Create cache key for prompt."""
        return f"prompt:{task_type}:{domain}:{difficulty}"
    
    async def get_prompt(
        self,
        task_type: str,
        domain: str,
        difficulty: str
    ) -> Optional[str]:
        """Get cached prompt."""
        key = self._make_prompt_key(task_type, domain, difficulty)
        result = await self.cache.get(key)
        
        if result and isinstance(result, dict):
            return result.get("prompt")
        return result
    
    async def cache_prompt(
        self,
        task_type: str,
        domain: str,
        difficulty: str,
        prompt: str,
        ttl: int = 3600  # 1 hour
    ) -> bool:
        """Cache generated prompt."""
        key = self._make_prompt_key(task_type, domain, difficulty)
        
        cached_prompt = {
            "prompt": prompt,
            "generated_at": datetime.utcnow().isoformat(),
            "task_type": task_type,
            "domain": domain,
            "difficulty": difficulty
        }
        
        return await self.cache.set(key, cached_prompt, ttl)


class InMemoryCache(CacheManager):
    """In-memory cache implementation for testing."""
    
    def __init__(self, default_ttl: int = 3600):
        """Initialize in-memory cache."""
        self.data: Dict[str, Any] = {}
        self.expiry: Dict[str, datetime] = {}
        self.default_ttl = default_ttl
    
    def _is_expired(self, key: str) -> bool:
        """Check if key is expired."""
        if key in self.expiry:
            return datetime.utcnow() > self.expiry[key]
        return False
    
    def _cleanup_expired(self):
        """Remove expired keys."""
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, expiry_time in self.expiry.items()
            if current_time > expiry_time
        ]
        
        for key in expired_keys:
            self.data.pop(key, None)
            self.expiry.pop(key, None)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        self._cleanup_expired()
        
        if key in self.data and not self._is_expired(key):
            return self.data[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        self.data[key] = value
        
        ttl = ttl or self.default_ttl
        self.expiry[key] = datetime.utcnow() + timedelta(seconds=ttl)
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from memory cache."""
        deleted = key in self.data
        self.data.pop(key, None)
        self.expiry.pop(key, None)
        return deleted
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        self._cleanup_expired()
        return key in self.data and not self._is_expired(key)
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        self.data.clear()
        self.expiry.clear()
        return True


# Factory functions
def create_redis_cache(redis_url: Optional[str] = None, prefix: str = "causal_eval:") -> RedisCache:
    """Create Redis cache instance."""
    return RedisCache(redis_url=redis_url, prefix=prefix)


def create_in_memory_cache() -> InMemoryCache:
    """Create in-memory cache instance for testing."""
    return InMemoryCache()


def create_evaluation_cache(redis_url: Optional[str] = None) -> EvaluationCache:
    """Create evaluation cache with Redis backend."""
    cache_manager = create_redis_cache(redis_url)
    return EvaluationCache(cache_manager)


def create_prompt_cache(redis_url: Optional[str] = None) -> PromptCache:
    """Create prompt cache with Redis backend."""
    cache_manager = create_redis_cache(redis_url)
    return PromptCache(cache_manager)
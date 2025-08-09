"""
Advanced caching system for performance optimization.
"""

import os
import json
import hashlib
import time
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import pickle
import redis
from redis import Redis
import aiocache
from aiocache import Cache, cached
from aiocache.serializers import JsonSerializer, PickleSerializer

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache level definitions."""
    L1_MEMORY = "l1_memory"        # In-memory cache (fastest)
    L2_REDIS = "l2_redis"          # Redis cache (fast, distributed)
    L3_DATABASE = "l3_database"     # Database cache (persistent)


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    ttl: int = 3600  # 1 hour default
    max_size: int = 1000
    compression: bool = False
    serializer: str = "json"  # json, pickle
    namespace: str = "causal_eval"


class CacheKey:
    """Utility for generating consistent cache keys."""
    
    @staticmethod
    def evaluation_prompt(task_type: str, domain: str, difficulty: str) -> str:
        """Generate cache key for evaluation prompts."""
        return f"prompt:{task_type}:{domain}:{difficulty}"
    
    @staticmethod
    def evaluation_result(task_type: str, model_response_hash: str, domain: str, difficulty: str) -> str:
        """Generate cache key for evaluation results."""
        return f"result:{task_type}:{model_response_hash}:{domain}:{difficulty}"
    
    @staticmethod
    def model_response(model_name: str, prompt_hash: str, temperature: float) -> str:
        """Generate cache key for model responses."""
        temp_str = f"{temperature:.2f}"
        return f"model:{model_name}:{prompt_hash}:{temp_str}"
    
    @staticmethod
    def session_data(session_id: str) -> str:
        """Generate cache key for session data."""
        return f"session:{session_id}"
    
    @staticmethod
    def leaderboard(domain: str, task_type: str) -> str:
        """Generate cache key for leaderboard data."""
        return f"leaderboard:{domain}:{task_type}"
    
    @staticmethod
    def hash_content(content: str) -> str:
        """Generate hash for content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class MultiLevelCache:
    """Multi-level caching system with L1 (memory) and L2 (Redis) support."""
    
    def __init__(self, config: CacheConfig, redis_client: Optional[Redis] = None):
        """Initialize multi-level cache."""
        self.config = config
        self.redis_client = redis_client
        
        # L1 Cache (Memory)
        self.l1_cache = aiocache.SimpleMemoryCache(
            serializer=JsonSerializer() if config.serializer == "json" else PickleSerializer(),
            namespace=config.namespace
        )
        
        # L2 Cache (Redis)
        if redis_client:
            self.l2_cache = aiocache.RedisCache(
                endpoint=redis_client.connection_pool.connection_kwargs.get("host", "localhost"),
                port=redis_client.connection_pool.connection_kwargs.get("port", 6379),
                serializer=JsonSerializer() if config.serializer == "json" else PickleSerializer(),
                namespace=config.namespace
            )
        else:
            self.l2_cache = None
        
        logger.info(f"Cache initialized: L1={self.l1_cache is not None}, L2={self.l2_cache is not None}")
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with multi-level fallback."""
        if not self.config.enabled:
            return default
        
        try:
            # Try L1 cache first
            value = await self.l1_cache.get(key)
            if value is not None:
                logger.debug(f"Cache hit L1: {key}")
                return value
            
            # Try L2 cache
            if self.l2_cache:
                value = await self.l2_cache.get(key)
                if value is not None:
                    logger.debug(f"Cache hit L2: {key}")
                    # Populate L1 cache for next time
                    await self.l1_cache.set(key, value, ttl=min(self.config.ttl, 300))
                    return value
            
            logger.debug(f"Cache miss: {key}")
            return default
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with multi-level storage."""
        if not self.config.enabled:
            return False
        
        ttl = ttl or self.config.ttl
        
        try:
            # Set in L1 cache
            await self.l1_cache.set(key, value, ttl=min(ttl, 300))  # Max 5 minutes in memory
            
            # Set in L2 cache
            if self.l2_cache:
                await self.l2_cache.set(key, value, ttl=ttl)
            
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from all cache levels."""
        try:
            # Delete from L1
            await self.l1_cache.delete(key)
            
            # Delete from L2
            if self.l2_cache:
                await self.l2_cache.delete(key)
            
            logger.debug(f"Cache delete: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear(self, pattern: Optional[str] = None) -> bool:
        """Clear cache entries matching pattern."""
        try:
            if pattern:
                # Clear specific pattern
                if self.l2_cache and self.redis_client:
                    keys = self.redis_client.keys(f"{self.config.namespace}:{pattern}")
                    if keys:
                        self.redis_client.delete(*keys)
            else:
                # Clear all
                await self.l1_cache.clear()
                if self.l2_cache:
                    await self.l2_cache.clear()
            
            logger.info(f"Cache cleared: {pattern or 'all'}")
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False


class CacheDecorator:
    """Decorator for caching function results."""
    
    def __init__(self, cache: MultiLevelCache, key_prefix: str, ttl: Optional[int] = None):
        """Initialize cache decorator."""
        self.cache = cache
        self.key_prefix = key_prefix
        self.ttl = ttl
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator implementation."""
        async def wrapper(*args, **kwargs):
            # Generate cache key from function arguments
            key_data = {
                "args": [str(arg) for arg in args],
                "kwargs": {k: str(v) for k, v in kwargs.items()}
            }
            key_str = json.dumps(key_data, sort_keys=True)
            cache_key = f"{self.key_prefix}:{CacheKey.hash_content(key_str)}"
            
            # Try cache first
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await self.cache.set(cache_key, result, self.ttl)
            
            return result
        
        return wrapper


class EvaluationCache:
    """Specialized cache for evaluation operations."""
    
    def __init__(self, cache: MultiLevelCache):
        """Initialize evaluation cache."""
        self.cache = cache
    
    async def get_prompt(self, task_type: str, domain: str, difficulty: str) -> Optional[str]:
        """Get cached evaluation prompt."""
        key = CacheKey.evaluation_prompt(task_type, domain, difficulty)
        return await self.cache.get(key)
    
    async def set_prompt(self, task_type: str, domain: str, difficulty: str, prompt: str) -> bool:
        """Cache evaluation prompt."""
        key = CacheKey.evaluation_prompt(task_type, domain, difficulty)
        # Prompts don't change often, cache for 24 hours
        return await self.cache.set(key, prompt, ttl=86400)
    
    async def get_evaluation_result(
        self, 
        task_type: str, 
        model_response: str, 
        domain: str, 
        difficulty: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached evaluation result."""
        response_hash = CacheKey.hash_content(model_response)
        key = CacheKey.evaluation_result(task_type, response_hash, domain, difficulty)
        return await self.cache.get(key)
    
    async def set_evaluation_result(
        self, 
        task_type: str, 
        model_response: str, 
        domain: str, 
        difficulty: str, 
        result: Dict[str, Any]
    ) -> bool:
        """Cache evaluation result."""
        response_hash = CacheKey.hash_content(model_response)
        key = CacheKey.evaluation_result(task_type, response_hash, domain, difficulty)
        # Cache evaluation results for 1 hour
        return await self.cache.set(key, result, ttl=3600)
    
    async def get_model_response(
        self, 
        model_name: str, 
        prompt: str, 
        temperature: float = 0.7
    ) -> Optional[str]:
        """Get cached model response."""
        prompt_hash = CacheKey.hash_content(prompt)
        key = CacheKey.model_response(model_name, prompt_hash, temperature)
        return await self.cache.get(key)
    
    async def set_model_response(
        self, 
        model_name: str, 
        prompt: str, 
        response: str, 
        temperature: float = 0.7
    ) -> bool:
        """Cache model response."""
        prompt_hash = CacheKey.hash_content(prompt)
        key = CacheKey.model_response(model_name, prompt_hash, temperature)
        # Cache model responses for 6 hours
        return await self.cache.set(key, response, ttl=21600)


class LeaderboardCache:
    """Specialized cache for leaderboard data."""
    
    def __init__(self, cache: MultiLevelCache):
        """Initialize leaderboard cache."""
        self.cache = cache
    
    async def get_leaderboard(self, domain: str = "all", task_type: str = "all") -> Optional[Dict[str, Any]]:
        """Get cached leaderboard data."""
        key = CacheKey.leaderboard(domain, task_type)
        return await self.cache.get(key)
    
    async def set_leaderboard(self, domain: str, task_type: str, data: Dict[str, Any]) -> bool:
        """Cache leaderboard data."""
        key = CacheKey.leaderboard(domain, task_type)
        # Cache leaderboard for 15 minutes
        return await self.cache.set(key, data, ttl=900)
    
    async def invalidate_leaderboard(self, domain: str = None, task_type: str = None) -> bool:
        """Invalidate leaderboard cache."""
        if domain and task_type:
            key = CacheKey.leaderboard(domain, task_type)
            return await self.cache.delete(key)
        else:
            # Clear all leaderboard data
            return await self.cache.clear("leaderboard:*")


class CacheManager:
    """Central cache management."""
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize cache manager."""
        self.redis_client = None
        
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Connected to Redis for caching")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using memory cache only.")
                self.redis_client = None
        
        # Cache configurations
        self.configs = {
            "default": CacheConfig(ttl=3600, max_size=1000),
            "evaluation": CacheConfig(ttl=3600, max_size=5000, serializer="json"),
            "prompts": CacheConfig(ttl=86400, max_size=100, serializer="json"),
            "leaderboard": CacheConfig(ttl=900, max_size=50, serializer="json"),
            "sessions": CacheConfig(ttl=7200, max_size=1000, serializer="pickle")
        }
        
        # Initialize caches
        self.caches = {
            name: MultiLevelCache(config, self.redis_client)
            for name, config in self.configs.items()
        }
        
        # Specialized caches
        self.evaluation_cache = EvaluationCache(self.caches["evaluation"])
        self.leaderboard_cache = LeaderboardCache(self.caches["leaderboard"])
    
    def get_cache(self, name: str = "default") -> MultiLevelCache:
        """Get cache by name."""
        return self.caches.get(name, self.caches["default"])
    
    async def warm_up(self):
        """Warm up cache with commonly accessed data."""
        logger.info("Starting cache warm-up...")
        
        # This would typically pre-load common prompts, frequently accessed data, etc.
        # For now, just log that warm-up is complete
        
        logger.info("Cache warm-up completed")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            "redis_connected": self.redis_client is not None,
            "caches": {}
        }
        
        for name, cache in self.caches.items():
            cache_stats = {
                "config": asdict(cache.config),
                "l1_enabled": cache.l1_cache is not None,
                "l2_enabled": cache.l2_cache is not None
            }
            
            # Get Redis stats if available
            if self.redis_client:
                try:
                    redis_info = self.redis_client.info()
                    cache_stats["redis_memory"] = redis_info.get("used_memory_human", "unknown")
                    cache_stats["redis_keys"] = self.redis_client.dbsize()
                except Exception as e:
                    logger.error(f"Error getting Redis stats: {e}")
            
            stats["caches"][name] = cache_stats
        
        return stats
    
    async def cleanup(self):
        """Cleanup cache resources."""
        logger.info("Cleaning up cache resources...")
        
        # Close Redis connection
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("Cache cleanup completed")


# Global cache manager instance
cache_manager = CacheManager()

# Convenience functions
def get_cache(name: str = "default") -> MultiLevelCache:
    """Get cache instance."""
    return cache_manager.get_cache(name)

def cached_evaluation(ttl: int = 3600):
    """Decorator for caching evaluation results."""
    return CacheDecorator(cache_manager.evaluation_cache.cache, "eval", ttl)

def cached_prompt(ttl: int = 86400):
    """Decorator for caching prompts."""
    return CacheDecorator(cache_manager.get_cache("prompts"), "prompt", ttl)

async def initialize_cache(redis_url: Optional[str] = None):
    """Initialize global cache manager - simplified version."""
    global cache_manager
    
    # For now, use simple in-memory cache
    class SimpleCacheManager:
        def __init__(self):
            self._cache = {}
        async def get(self, key: str) -> Any:
            return self._cache.get(key)
        async def set(self, key: str, value: Any, ttl: int = 300) -> None:
            self._cache[key] = value
        async def cleanup(self) -> None:
            self._cache.clear()
    
    cache_manager = SimpleCacheManager()
    return cache_manager
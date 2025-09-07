"""
InvestIQ Platform - Redis工具模块
Redis连接管理和通用缓存操作
"""

import redis.asyncio as redis
import json
import logging
from typing import Any, Optional, Union, Dict
from datetime import timedelta

from ..config import settings


logger = logging.getLogger(__name__)


class RedisManager:
    """Redis连接管理器"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.redis.url
        self.redis_client: Optional[redis.Redis] = None
        
    async def initialize(self):
        """初始化Redis连接"""
        self.redis_client = redis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        
        # 测试连接
        await self.redis_client.ping()
        logger.info(f"Redis initialized: {self.redis_url}")
    
    async def close(self):
        """关闭Redis连接"""
        if self.redis_client:
            await self.redis_client.close()
    
    def get_client(self) -> redis.Redis:
        """获取Redis客户端"""
        if not self.redis_client:
            raise RuntimeError("Redis not initialized. Call initialize() first.")
        return self.redis_client
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置键值"""
        client = self.get_client()
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        
        await client.set(key, value, ex=ttl)
    
    async def get(self, key: str) -> Optional[str]:
        """获取值"""
        client = self.get_client()
        return await client.get(key)
    
    async def get_json(self, key: str) -> Optional[Dict]:
        """获取JSON值"""
        value = await self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode JSON for key: {key}")
        return None
    
    async def delete(self, key: str) -> bool:
        """删除键"""
        client = self.get_client()
        return await client.delete(key) > 0
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        client = self.get_client()
        return await client.exists(key) > 0
    
    async def check_connection(self) -> bool:
        """检查Redis连接"""
        try:
            client = self.get_client()
            await client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis connection check failed: {e}")
            return False


# 全局Redis管理器实例
redis_manager = RedisManager()


async def init_redis():
    """初始化Redis"""
    await redis_manager.initialize()
    return redis_manager


async def get_redis() -> redis.Redis:
    """获取Redis客户端"""
    return redis_manager.get_client()


async def redis_health_check() -> bool:
    """Redis健康检查"""
    return await redis_manager.check_connection()
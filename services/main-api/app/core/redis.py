"""
Redis客户端 - 精简版本
"""

import aioredis
from typing import Optional

from .config import settings


class RedisClient:
    """Redis客户端封装"""
    
    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None
    
    async def connect(self):
        """连接Redis"""
        if not self._redis:
            self._redis = aioredis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
            )
    
    async def ping(self) -> bool:
        """检查连接"""
        if not self._redis:
            await self.connect()
        
        try:
            await self._redis.ping()
            return True
        except Exception:
            return False
    
    async def get(self, key: str) -> Optional[str]:
        """获取值"""
        if not self._redis:
            await self.connect()
        return await self._redis.get(key)
    
    async def set(self, key: str, value: str, expire: int = None) -> bool:
        """设置值"""
        if not self._redis:
            await self.connect()
        return await self._redis.set(key, value, ex=expire)
    
    async def delete(self, key: str) -> bool:
        """删除键"""
        if not self._redis:
            await self.connect()
        return await self._redis.delete(key) > 0
    
    async def close(self):
        """关闭连接"""
        if self._redis:
            await self._redis.close()
            self._redis = None


# 全局Redis客户端实例
redis_client = RedisClient()
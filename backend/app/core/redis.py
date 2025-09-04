"""
InvestIQ Platform - Redis配置模块
管理Redis连接和缓存操作
"""

import json
import logging
from typing import Any, Optional, Union
import redis.asyncio as redis
from backend.app.core.config import settings


class RedisManager:
    """Redis管理器"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.connection_pool: Optional[redis.ConnectionPool] = None
    
    async def connect(self):
        """建立Redis连接"""
        try:
            self.connection_pool = redis.ConnectionPool.from_url(
                settings.REDIS_URL,
                max_connections=20,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
            )
            
            self.redis_client = redis.Redis(
                connection_pool=self.connection_pool,
                decode_responses=True,
            )
            
            # 测试连接
            await self.redis_client.ping()
            logging.info("Redis connection established")
            
        except Exception as e:
            logging.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def close(self):
        """关闭Redis连接"""
        if self.redis_client:
            await self.redis_client.close()
        if self.connection_pool:
            await self.connection_pool.disconnect()
        logging.info("Redis connection closed")
    
    async def ping(self) -> bool:
        """检查Redis连接状态"""
        try:
            if self.redis_client:
                await self.redis_client.ping()
                return True
            return False
        except Exception:
            return False
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """设置缓存值"""
        try:
            if not self.redis_client:
                await self.connect()
            
            # 序列化值
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            elif not isinstance(value, str):
                value = str(value)
            
            ttl = ttl or settings.REDIS_CACHE_TTL
            await self.redis_client.setex(key, ttl, value)
            return True
            
        except Exception as e:
            logging.error(f"Redis set error: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        try:
            if not self.redis_client:
                await self.connect()
            
            value = await self.redis_client.get(key)
            if value is None:
                return None
            
            # 尝试反序列化JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logging.error(f"Redis get error: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """删除缓存键"""
        try:
            if not self.redis_client:
                await self.connect()
            
            result = await self.redis_client.delete(key)
            return result > 0
            
        except Exception as e:
            logging.error(f"Redis delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            if not self.redis_client:
                await self.connect()
            
            result = await self.redis_client.exists(key)
            return result > 0
            
        except Exception as e:
            logging.error(f"Redis exists error: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """设置键的过期时间"""
        try:
            if not self.redis_client:
                await self.connect()
            
            result = await self.redis_client.expire(key, ttl)
            return result
            
        except Exception as e:
            logging.error(f"Redis expire error: {e}")
            return False
    
    async def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """递增计数器"""
        try:
            if not self.redis_client:
                await self.connect()
            
            result = await self.redis_client.incrby(key, amount)
            return result
            
        except Exception as e:
            logging.error(f"Redis incr error: {e}")
            return None
    
    async def decr(self, key: str, amount: int = 1) -> Optional[int]:
        """递减计数器"""
        try:
            if not self.redis_client:
                await self.connect()
            
            result = await self.redis_client.decrby(key, amount)
            return result
            
        except Exception as e:
            logging.error(f"Redis decr error: {e}")
            return None
    
    async def hset(self, name: str, mapping: dict) -> bool:
        """设置哈希表"""
        try:
            if not self.redis_client:
                await self.connect()
            
            # 序列化值
            serialized_mapping = {}
            for k, v in mapping.items():
                if isinstance(v, (dict, list)):
                    serialized_mapping[k] = json.dumps(v, ensure_ascii=False)
                else:
                    serialized_mapping[k] = str(v)
            
            await self.redis_client.hset(name, mapping=serialized_mapping)
            return True
            
        except Exception as e:
            logging.error(f"Redis hset error: {e}")
            return False
    
    async def hget(self, name: str, key: str) -> Optional[Any]:
        """获取哈希表字段值"""
        try:
            if not self.redis_client:
                await self.connect()
            
            value = await self.redis_client.hget(name, key)
            if value is None:
                return None
            
            # 尝试反序列化JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logging.error(f"Redis hget error: {e}")
            return None
    
    async def hgetall(self, name: str) -> dict:
        """获取哈希表所有字段"""
        try:
            if not self.redis_client:
                await self.connect()
            
            data = await self.redis_client.hgetall(name)
            
            # 反序列化值
            result = {}
            for k, v in data.items():
                try:
                    result[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    result[k] = v
            
            return result
            
        except Exception as e:
            logging.error(f"Redis hgetall error: {e}")
            return {}
    
    async def lpush(self, name: str, *values) -> Optional[int]:
        """向列表左侧推入元素"""
        try:
            if not self.redis_client:
                await self.connect()
            
            # 序列化值
            serialized_values = []
            for value in values:
                if isinstance(value, (dict, list)):
                    serialized_values.append(json.dumps(value, ensure_ascii=False))
                else:
                    serialized_values.append(str(value))
            
            result = await self.redis_client.lpush(name, *serialized_values)
            return result
            
        except Exception as e:
            logging.error(f"Redis lpush error: {e}")
            return None
    
    async def rpop(self, name: str) -> Optional[Any]:
        """从列表右侧弹出元素"""
        try:
            if not self.redis_client:
                await self.connect()
            
            value = await self.redis_client.rpop(name)
            if value is None:
                return None
            
            # 尝试反序列化JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logging.error(f"Redis rpop error: {e}")
            return None
    
    async def llen(self, name: str) -> int:
        """获取列表长度"""
        try:
            if not self.redis_client:
                await self.connect()
            
            result = await self.redis_client.llen(name)
            return result
            
        except Exception as e:
            logging.error(f"Redis llen error: {e}")
            return 0
    
    async def sadd(self, name: str, *values) -> Optional[int]:
        """向集合添加元素"""
        try:
            if not self.redis_client:
                await self.connect()
            
            # 序列化值
            serialized_values = []
            for value in values:
                if isinstance(value, (dict, list)):
                    serialized_values.append(json.dumps(value, ensure_ascii=False))
                else:
                    serialized_values.append(str(value))
            
            result = await self.redis_client.sadd(name, *serialized_values)
            return result
            
        except Exception as e:
            logging.error(f"Redis sadd error: {e}")
            return None
    
    async def sismember(self, name: str, value: Any) -> bool:
        """检查元素是否在集合中"""
        try:
            if not self.redis_client:
                await self.connect()
            
            # 序列化值
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            else:
                value = str(value)
            
            result = await self.redis_client.sismember(name, value)
            return result
            
        except Exception as e:
            logging.error(f"Redis sismember error: {e}")
            return False
    
    async def smembers(self, name: str) -> set:
        """获取集合所有成员"""
        try:
            if not self.redis_client:
                await self.connect()
            
            members = await self.redis_client.smembers(name)
            
            # 反序列化值
            result = set()
            for member in members:
                try:
                    result.add(json.loads(member))
                except (json.JSONDecodeError, TypeError):
                    result.add(member)
            
            return result
            
        except Exception as e:
            logging.error(f"Redis smembers error: {e}")
            return set()
    
    async def keys(self, pattern: str = "*") -> list:
        """获取匹配模式的键"""
        try:
            if not self.redis_client:
                await self.connect()
            
            keys = await self.redis_client.keys(pattern)
            return keys
            
        except Exception as e:
            logging.error(f"Redis keys error: {e}")
            return []
    
    async def flushdb(self) -> bool:
        """清空当前数据库"""
        try:
            if not self.redis_client:
                await self.connect()
            
            await self.redis_client.flushdb()
            return True
            
        except Exception as e:
            logging.error(f"Redis flushdb error: {e}")
            return False
    
    async def info(self) -> dict:
        """获取Redis服务器信息"""
        try:
            if not self.redis_client:
                await self.connect()
            
            info = await self.redis_client.info()
            return info
            
        except Exception as e:
            logging.error(f"Redis info error: {e}")
            return {}


# 创建全局Redis客户端实例
redis_client = RedisManager()


# 缓存装饰器
def cache_result(key_prefix: str, ttl: Optional[int] = None):
    """缓存函数结果的装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{key_prefix}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # 尝试从缓存获取
            cached_result = await redis_client.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数并缓存结果
            result = await func(*args, **kwargs)
            await redis_client.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


# 速率限制
async def check_rate_limit(
    key: str, 
    limit: int, 
    window: int = 3600
) -> tuple[bool, int]:
    """
    检查速率限制
    
    Args:
        key: 限制键（通常是用户ID或IP）
        limit: 限制次数
        window: 时间窗口（秒）
    
    Returns:
        (是否允许, 剩余次数)
    """
    try:
        current = await redis_client.incr(f"rate_limit:{key}")
        
        if current == 1:
            # 第一次请求，设置过期时间
            await redis_client.expire(f"rate_limit:{key}", window)
        
        remaining = max(0, limit - current)
        allowed = current <= limit
        
        return allowed, remaining
        
    except Exception as e:
        logging.error(f"Rate limit check error: {e}")
        # 出错时允许请求
        return True, limit

"""
InvestIQ Platform - 幂等性服务
实现幂等键管理和结果缓存
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Column, String, DateTime, Text, Integer, select, delete, and_, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

from backend.app.core.config import settings
from backend.app.core.exceptions import IdempotencyException
from backend.app.core.logging import get_logger
from backend.app.models.base import Base, TimestampMixin


logger = get_logger(__name__)


class IdempotencyRecord(Base, TimestampMixin):
    """幂等性记录表"""
    
    __tablename__ = "idempotency_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    idempotency_key = Column(String(128), nullable=False, unique=True, index=True, comment="幂等键")
    request_hash = Column(String(64), nullable=False, comment="请求内容哈希")
    request_body = Column(JSONB, nullable=False, comment="请求体")
    response_data = Column(JSONB, nullable=False, comment="响应数据")
    status_code = Column(Integer, nullable=False, default=200, comment="状态码")
    endpoint = Column(String(200), nullable=False, comment="API端点")
    method = Column(String(10), nullable=False, comment="HTTP方法")
    user_id = Column(String(100), nullable=True, comment="用户ID")
    expires_at = Column(DateTime, nullable=False, comment="过期时间")
    
    __table_args__ = (
        Index("ix_idempotency_records_key", "idempotency_key"),
        Index("ix_idempotency_records_expires", "expires_at"),
        {"comment": "幂等性记录表"}
    )


class IdempotencyService:
    """幂等性服务"""
    
    def __init__(self, db_session: Optional[AsyncSession] = None):
        self.db_session = db_session
        self.default_ttl = timedelta(hours=24)  # 默认24小时TTL
        
    def _calculate_request_hash(self, request_data: Dict[str, Any]) -> str:
        """计算请求数据哈希"""
        # 排序字典确保一致性
        json_data = json.dumps(request_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(json_data.encode('utf-8')).hexdigest()
    
    async def get_result(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        """获取幂等性结果"""
        if not self.db_session:
            return None
            
        try:
            # 查询幂等性记录
            stmt = select(IdempotencyRecord).where(
                and_(
                    IdempotencyRecord.idempotency_key == idempotency_key,
                    IdempotencyRecord.expires_at > datetime.utcnow()
                )
            )
            
            result = await self.db_session.execute(stmt)
            record = result.scalar_one_or_none()
            
            if record:
                logger.info(f"Found idempotency result for key: {idempotency_key}")
                return record.response_data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get idempotency result: {e}")
            return None
    
    async def store_result(
        self,
        idempotency_key: str,
        response_data: Dict[str, Any],
        request_data: Dict[str, Any],
        endpoint: str = "unknown",
        method: str = "POST",
        user_id: Optional[str] = None,
        status_code: int = 200,
        ttl: Optional[timedelta] = None
    ):
        """存储幂等性结果"""
        if not self.db_session:
            return
            
        try:
            # 计算请求哈希
            request_hash = self._calculate_request_hash(request_data)
            
            # 设置过期时间
            expires_at = datetime.utcnow() + (ttl or self.default_ttl)
            
            # 创建幂等性记录
            record = IdempotencyRecord(
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                request_body=request_data,
                response_data=response_data,
                status_code=status_code,
                endpoint=endpoint,
                method=method,
                user_id=user_id,
                expires_at=expires_at
            )
            
            self.db_session.add(record)
            await self.db_session.commit()
            
            logger.info(f"Stored idempotency result for key: {idempotency_key}")
            
        except Exception as e:
            await self.db_session.rollback()
            logger.error(f"Failed to store idempotency result: {e}")
            raise IdempotencyException(f"存储幂等性结果失败: {e}")
    
    async def validate_request_consistency(
        self,
        idempotency_key: str,
        request_data: Dict[str, Any]
    ) -> bool:
        """验证请求一致性"""
        if not self.db_session:
            return True
            
        try:
            # 查询现有记录
            stmt = select(IdempotencyRecord).where(
                and_(
                    IdempotencyRecord.idempotency_key == idempotency_key,
                    IdempotencyRecord.expires_at > datetime.utcnow()
                )
            )
            
            result = await self.db_session.execute(stmt)
            record = result.scalar_one_or_none()
            
            if not record:
                return True  # 没有现有记录，可以继续
            
            # 计算当前请求哈希
            current_hash = self._calculate_request_hash(request_data)
            
            # 检查哈希是否匹配
            if record.request_hash != current_hash:
                logger.warning(f"Request hash mismatch for idempotency key: {idempotency_key}")
                raise IdempotencyException(
                    f"幂等键 {idempotency_key} 的请求内容不一致"
                )
            
            return True
            
        except IdempotencyException:
            raise
        except Exception as e:
            logger.error(f"Failed to validate request consistency: {e}")
            return True  # 验证失败时允许继续（降级处理）
    
    async def cleanup_expired_records(self):
        """清理过期记录"""
        if not self.db_session:
            return
            
        try:
            # 删除过期记录
            stmt = delete(IdempotencyRecord).where(
                IdempotencyRecord.expires_at <= datetime.utcnow()
            )
            
            result = await self.db_session.execute(stmt)
            deleted_count = result.rowcount
            
            await self.db_session.commit()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired idempotency records")
            
        except Exception as e:
            await self.db_session.rollback()
            logger.error(f"Failed to cleanup expired idempotency records: {e}")


class IdempotencyManager:
    """幂等性管理器 - 装饰器和上下文管理"""
    
    def __init__(self, db_session: AsyncSession):
        self.service = IdempotencyService(db_session)
    
    async def __aenter__(self):
        return self.service
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def idempotent_endpoint(
        self,
        endpoint: str,
        method: str = "POST",
        ttl_hours: int = 24
    ):
        """幂等性端点装饰器"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # 提取幂等键和请求数据
                idempotency_key = kwargs.get('idempotency_key')
                if not idempotency_key:
                    # 没有幂等键，直接执行
                    return await func(*args, **kwargs)
                
                # 构建请求数据
                request_data = {
                    'args': args,
                    'kwargs': {k: v for k, v in kwargs.items() if k != 'idempotency_key'}
                }
                
                # 验证请求一致性
                await self.service.validate_request_consistency(idempotency_key, request_data)
                
                # 检查现有结果
                existing_result = await self.service.get_result(idempotency_key)
                if existing_result:
                    return existing_result
                
                # 执行操作
                result = await func(*args, **kwargs)
                
                # 存储结果
                await self.service.store_result(
                    idempotency_key=idempotency_key,
                    response_data=result,
                    request_data=request_data,
                    endpoint=endpoint,
                    method=method,
                    ttl=timedelta(hours=ttl_hours)
                )
                
                return result
                
            return wrapper
        return decorator


# 全局幂等性服务实例
async def create_idempotency_service(db_session: AsyncSession) -> IdempotencyService:
    """创建幂等性服务实例"""
    return IdempotencyService(db_session)
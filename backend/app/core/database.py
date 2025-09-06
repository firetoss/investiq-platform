"""
InvestIQ Platform - 数据库配置模块
管理PostgreSQL数据库连接和会话
"""

import logging
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import MetaData, event
from sqlalchemy.engine import Engine

from backend.app.core.config import settings

# 创建基础模型类
Base = declarative_base(
    metadata=MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s"
        }
    )
)

# 创建异步数据库引擎
if settings.ENVIRONMENT == "production":
    engine = create_async_engine(
        settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
        echo=settings.DEBUG,
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
        pool_pre_ping=True,
        pool_recycle=3600,  # 1小时回收连接
        poolclass=QueuePool,
    )
else:
    engine = create_async_engine(
        settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
        echo=settings.DEBUG,
        pool_pre_ping=True,
        poolclass=NullPool,
    )

# 创建会话工厂
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=True,
    autocommit=False,
)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """设置数据库连接参数"""
    if "postgresql" in str(dbapi_connection):
        # PostgreSQL特定设置
        cursor = dbapi_connection.cursor()
        cursor.execute("SET timezone TO 'Asia/Shanghai'")
        cursor.execute("SET statement_timeout = '30s'")
        cursor.execute("SET lock_timeout = '10s'")
        cursor.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    获取数据库会话的依赖注入函数
    用于FastAPI的Depends
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logging.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


async def create_tables():
    """创建所有数据库表"""
    try:
        async with engine.begin() as conn:
            # 导入所有模型以确保它们被注册
            from backend.app.models import (
                industry,
                equity,
                portfolio,
                alert,
                evidence,
                audit,
                user,  # 虽然是单用户，但保留用户模型用于审计
            )
            
            # 创建所有表
            await conn.run_sync(Base.metadata.create_all)
            logging.info("Database tables created successfully")
            
    except Exception as e:
        logging.error(f"Error creating database tables: {e}")
        raise


async def drop_tables():
    """删除所有数据库表 (仅用于开发/测试)"""
    if settings.ENVIRONMENT == "production":
        raise ValueError("Cannot drop tables in production environment")
    
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            logging.warning("All database tables dropped")
    except Exception as e:
        logging.error(f"Error dropping database tables: {e}")
        raise


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self.engine = engine
        self.session_factory = AsyncSessionLocal
    
    async def health_check(self) -> bool:
        """数据库健康检查"""
        try:
            async with self.session_factory() as session:
                result = await session.execute("SELECT 1")
                return result.scalar() == 1
        except Exception as e:
            logging.error(f"Database health check failed: {e}")
            return False
    
    async def get_connection_info(self) -> dict:
        """获取数据库连接信息"""
        try:
            async with self.session_factory() as session:
                # 获取数据库版本
                version_result = await session.execute("SELECT version()")
                version = version_result.scalar()
                
                # 获取连接数
                connections_result = await session.execute(
                    "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
                )
                active_connections = connections_result.scalar()
                
                # 获取数据库大小
                size_result = await session.execute(
                    "SELECT pg_size_pretty(pg_database_size(current_database()))"
                )
                database_size = size_result.scalar()
                
                return {
                    "version": version,
                    "active_connections": active_connections,
                    "database_size": database_size,
                    "pool_size": self.engine.pool.size(),
                    "checked_out_connections": self.engine.pool.checkedout(),
                }
        except Exception as e:
            logging.error(f"Error getting database info: {e}")
            return {"error": str(e)}
    
    async def execute_raw_sql(self, sql: str, params: Optional[dict] = None) -> list:
        """执行原始SQL查询 (谨慎使用)"""
        if settings.ENVIRONMENT == "production":
            raise ValueError("Raw SQL execution not allowed in production")
        
        try:
            async with self.session_factory() as session:
                result = await session.execute(sql, params or {})
                return result.fetchall()
        except Exception as e:
            logging.error(f"Error executing raw SQL: {e}")
            raise
    
    async def backup_database(self, backup_path: str) -> bool:
        """数据库备份 (需要pg_dump)"""
        import subprocess
        import asyncio
        
        try:
            # 解析数据库URL
            from urllib.parse import urlparse
            parsed = urlparse(settings.DATABASE_URL)
            
            cmd = [
                "pg_dump",
                f"--host={parsed.hostname}",
                f"--port={parsed.port or 5432}",
                f"--username={parsed.username}",
                f"--dbname={parsed.path[1:]}",  # 去掉开头的 /
                "--format=custom",
                "--no-password",
                f"--file={backup_path}",
            ]
            
            # 设置密码环境变量
            env = {"PGPASSWORD": parsed.password}
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logging.info(f"Database backup created: {backup_path}")
                return True
            else:
                logging.error(f"Database backup failed: {stderr.decode()}")
                return False
                
        except Exception as e:
            logging.error(f"Error creating database backup: {e}")
            return False
    
    async def restore_database(self, backup_path: str) -> bool:
        """数据库恢复 (需要pg_restore)"""
        if settings.ENVIRONMENT == "production":
            raise ValueError("Database restore not allowed in production without explicit confirmation")
        
        import subprocess
        import asyncio
        
        try:
            # 解析数据库URL
            from urllib.parse import urlparse
            parsed = urlparse(settings.DATABASE_URL)
            
            cmd = [
                "pg_restore",
                f"--host={parsed.hostname}",
                f"--port={parsed.port or 5432}",
                f"--username={parsed.username}",
                f"--dbname={parsed.path[1:]}",
                "--clean",
                "--if-exists",
                "--no-password",
                backup_path,
            ]
            
            # 设置密码环境变量
            env = {"PGPASSWORD": parsed.password}
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logging.info(f"Database restored from: {backup_path}")
                return True
            else:
                logging.error(f"Database restore failed: {stderr.decode()}")
                return False
                
        except Exception as e:
            logging.error(f"Error restoring database: {e}")
            return False


# 创建全局数据库管理器实例
db_manager = DatabaseManager()


# 时态表支持函数
async def create_temporal_table_trigger(table_name: str):
    """为表创建时态触发器"""
    trigger_sql = f"""
    CREATE OR REPLACE FUNCTION {table_name}_temporal_trigger()
    RETURNS TRIGGER AS $$
    BEGIN
        IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
            NEW.updated_at = NOW();
            RETURN NEW;
        END IF;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql;
    
    DROP TRIGGER IF EXISTS {table_name}_temporal_trigger ON {table_name};
    CREATE TRIGGER {table_name}_temporal_trigger
        BEFORE INSERT OR UPDATE ON {table_name}
        FOR EACH ROW EXECUTE FUNCTION {table_name}_temporal_trigger();
    """
    
    async with engine.begin() as conn:
        await conn.execute(trigger_sql)


# 数据库迁移支持
class MigrationManager:
    """数据库迁移管理器"""
    
    def __init__(self):
        self.engine = engine
    
    async def get_current_version(self) -> Optional[str]:
        """获取当前数据库版本"""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    "SELECT version FROM alembic_version LIMIT 1"
                )
                return result.scalar()
        except Exception:
            return None
    
    async def check_migration_status(self) -> dict:
        """检查迁移状态"""
        current_version = await self.get_current_version()
        return {
            "current_version": current_version,
            "needs_migration": current_version is None,
        }


# 创建全局迁移管理器实例
migration_manager = MigrationManager()

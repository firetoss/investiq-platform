"""
InvestIQ Platform - 数据库工具模块
SQLAlchemy连接管理和通用数据库操作
"""

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
import logging
import time
from typing import Generator, Optional, Dict, Any

from ..config import settings
from ..models.base import Base


logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库连接管理器"""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.database.url
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self.scoped_session_factory: Optional[scoped_session] = None
        
    def initialize(self, echo: bool = False, pool_size: int = 5, max_overflow: int = 10):
        """初始化数据库连接"""
        
        # 创建引擎
        self.engine = create_engine(
            self.database_url,
            echo=echo or settings.app.debug,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # 连接健康检查
            pool_recycle=3600,   # 连接回收时间 (1小时)
        )
        
        # 设置查询日志事件监听器
        if settings.app.debug:
            self._setup_query_logging()
        
        # 创建会话工厂
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # 创建作用域会话
        self.scoped_session_factory = scoped_session(self.SessionLocal)
        
        logger.info(f"Database initialized: {self.database_url}")
    
    def _setup_query_logging(self):
        """设置查询日志"""
        
        @event.listens_for(self.engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()
        
        @event.listens_for(self.engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total = time.time() - context._query_start_time
            if total > 0.1:  # 只记录慢查询 (>100ms)
                logger.warning(f"Slow query ({total:.3f}s): {statement[:100]}...")
    
    def create_all_tables(self):
        """创建所有表"""
        if self.engine:
            Base.metadata.create_all(bind=self.engine)
            logger.info("All database tables created")
    
    def drop_all_tables(self):
        """删除所有表"""
        if self.engine:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("All database tables dropped")
    
    def get_session(self) -> Session:
        """获取数据库会话"""
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self.SessionLocal()
    
    def get_scoped_session(self) -> scoped_session:
        """获取作用域会话"""
        if not self.scoped_session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self.scoped_session_factory
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """数据库会话上下文管理器"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def check_connection(self) -> bool:
        """检查数据库连接"""
        try:
            with self.session_scope() as session:
                session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""
        if not self.engine:
            return {"status": "not_initialized"}
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute("SELECT version()").fetchone()
                return {
                    "status": "connected",
                    "database_version": result[0] if result else "unknown",
                    "pool_size": self.engine.pool.size(),
                    "checked_in": self.engine.pool.checkedin(),
                    "overflow": self.engine.pool.overflow(),
                    "checked_out": self.engine.pool.checkedout()
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


# 全局数据库管理器实例
db_manager = DatabaseManager()


def init_database(echo: bool = False):
    """初始化数据库"""
    db_manager.initialize(echo=echo)
    return db_manager


def get_db() -> Generator[Session, None, None]:
    """FastAPI依赖注入：获取数据库会话"""
    with db_manager.session_scope() as session:
        yield session


def get_db_session() -> Session:
    """直接获取数据库会话"""
    return db_manager.get_session()


class Repository:
    """基础仓库类"""
    
    def __init__(self, model, session: Session):
        self.model = model
        self.session = session
    
    def get_by_id(self, id: Any):
        """根据ID获取记录"""
        return self.session.query(self.model).filter(self.model.id == id).first()
    
    def get_all(self, skip: int = 0, limit: int = 100):
        """获取所有记录"""
        return self.session.query(self.model).offset(skip).limit(limit).all()
    
    def create(self, obj_in: Dict[str, Any]):
        """创建记录"""
        db_obj = self.model(**obj_in)
        self.session.add(db_obj)
        self.session.flush()
        self.session.refresh(db_obj)
        return db_obj
    
    def update(self, db_obj, obj_in: Dict[str, Any]):
        """更新记录"""
        for field, value in obj_in.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        
        self.session.flush()
        self.session.refresh(db_obj)
        return db_obj
    
    def delete(self, id: Any):
        """删除记录"""
        db_obj = self.get_by_id(id)
        if db_obj:
            self.session.delete(db_obj)
            self.session.flush()
        return db_obj
    
    def count(self):
        """统计记录数"""
        return self.session.query(self.model).count()
    
    def exists(self, **kwargs):
        """检查记录是否存在"""
        query = self.session.query(self.model)
        for key, value in kwargs.items():
            if hasattr(self.model, key):
                query = query.filter(getattr(self.model, key) == value)
        return query.first() is not None


class TransactionalService:
    """事务服务基类"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def commit(self):
        """提交事务"""
        self.session.commit()
    
    def rollback(self):
        """回滚事务"""
        self.session.rollback()
    
    def flush(self):
        """刷新会话"""
        self.session.flush()


def check_database_health() -> Dict[str, Any]:
    """数据库健康检查"""
    return db_manager.get_connection_info()


async def database_health_check() -> bool:
    """异步数据库健康检查"""
    return db_manager.check_connection()
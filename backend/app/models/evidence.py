"""
InvestIQ Platform - 证据和审计数据模型
证据链、审计日志和用户模型
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Text, JSON, Boolean, Index, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
import enum
import hashlib

from backend.app.core.database import Base


class EvidenceType(enum.Enum):
    """证据类型"""
    POLICY = "policy"           # 政策文件
    ORDER = "order"             # 订单/招标
    FINANCIAL = "financial"     # 财务数据
    NEWS = "news"               # 新闻报道
    ANNOUNCEMENT = "announcement"  # 公司公告
    RESEARCH = "research"       # 研究报告
    MARKET_DATA = "market_data" # 市场数据
    OTHER = "other"             # 其他


class EvidenceSource(enum.Enum):
    """证据来源"""
    OFFICIAL = "official"       # 官方来源 (权重 1.0)
    AUTHORIZED = "authorized"   # 授权来源 (权重 0.9)
    PUBLIC = "public"          # 公开抓取 (权重 0.8)


class EvidenceItem(Base):
    """证据项表"""
    
    __tablename__ = "evidence_items"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 关联实体
    entity_type = Column(String(50), nullable=False, comment="实体类型 (industry/equity/portfolio)")
    entity_id = Column(UUID(as_uuid=True), nullable=False, comment="实体ID")
    
    # 证据基本信息
    evidence_type = Column(Enum(EvidenceType), nullable=False, comment="证据类型")
    title = Column(String(500), nullable=False, comment="证据标题")
    description = Column(Text, comment="证据描述")
    
    # 来源信息
    source = Column(Enum(EvidenceSource), nullable=False, comment="证据来源")
    source_url = Column(String(1000), comment="来源URL")
    source_name = Column(String(200), comment="来源名称")
    
    # 内容信息
    content = Column(Text, comment="证据内容")
    content_hash = Column(String(64), comment="内容哈希 (SHA-256)")
    file_path = Column(String(500), comment="文件路径 (MinIO)")
    file_size = Column(Integer, comment="文件大小 (字节)")
    file_type = Column(String(50), comment="文件类型")
    
    # 元数据
    metadata = Column(JSON, comment="元数据")
    tags = Column(JSON, comment="标签列表")
    keywords = Column(JSON, comment="关键词列表")
    
    # 质量评估
    confidence_score = Column(Float, comment="置信度评分 (0-100)")
    relevance_score = Column(Float, comment="相关性评分 (0-100)")
    quality_score = Column(Float, comment="质量评分 (0-100)")
    
    # 时间信息
    evidence_date = Column(Date, comment="证据日期")
    published_at = Column(DateTime(timezone=True), comment="发布时间")
    collected_at = Column(DateTime(timezone=True), server_default=func.now(), comment="收集时间")
    
    # 状态信息
    is_verified = Column(Boolean, default=False, comment="是否已验证")
    is_active = Column(Boolean, default=True, comment="是否活跃")
    verified_by = Column(String(100), comment="验证人")
    verified_at = Column(DateTime(timezone=True), comment="验证时间")
    
    # 生命周期
    expires_at = Column(DateTime(timezone=True), comment="过期时间")
    archived_at = Column(DateTime(timezone=True), comment="归档时间")
    
    # 时间戳
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # 索引
    __table_args__ = (
        Index('idx_evidence_entity', 'entity_type', 'entity_id'),
        Index('idx_evidence_type', 'evidence_type'),
        Index('idx_evidence_source', 'source'),
        Index('idx_evidence_date', 'evidence_date'),
        Index('idx_evidence_collected', 'collected_at'),
        Index('idx_evidence_hash', 'content_hash'),
        Index('idx_evidence_active', 'is_active'),
        Index('idx_evidence_verified', 'is_verified'),
    )
    
    def __repr__(self):
        return f"<EvidenceItem(type='{self.evidence_type.value}', title='{self.title[:50]}...')>"
    
    def calculate_content_hash(self) -> str:
        """计算内容哈希"""
        if self.content:
            return hashlib.sha256(self.content.encode('utf-8')).hexdigest()
        return ""
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False
    
    def get_source_weight(self) -> float:
        """获取来源权重"""
        weights = {
            EvidenceSource.OFFICIAL: 1.0,
            EvidenceSource.AUTHORIZED: 0.9,
            EvidenceSource.PUBLIC: 0.8
        }
        return weights.get(self.source, 0.8)
    
    def calculate_overall_score(self) -> float:
        """计算综合评分"""
        scores = [
            self.confidence_score or 0,
            self.relevance_score or 0,
            self.quality_score or 0
        ]
        if not any(scores):
            return 0
        
        # 加权平均，考虑来源权重
        base_score = sum(scores) / len([s for s in scores if s > 0])
        source_weight = self.get_source_weight()
        
        return base_score * source_weight


class DecisionLog(Base):
    """决策日志表 - 不可变审计链"""
    
    __tablename__ = "decision_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 审计链信息
    sequence_number = Column(Integer, nullable=False, comment="序列号")
    previous_hash = Column(String(64), comment="前一条记录哈希")
    current_hash = Column(String(64), nullable=False, comment="当前记录哈希")
    
    # 操作信息
    user_id = Column(String(100), nullable=False, comment="操作用户")
    action = Column(String(100), nullable=False, comment="操作动作")
    resource = Column(String(100), comment="操作资源")
    resource_id = Column(String(100), comment="资源ID")
    
    # 操作数据
    payload = Column(JSON, comment="操作载荷")
    payload_hash = Column(String(64), comment="载荷哈希")
    before_state = Column(JSON, comment="操作前状态")
    after_state = Column(JSON, comment="操作后状态")
    
    # 上下文信息
    request_id = Column(String(100), comment="请求ID")
    session_id = Column(String(100), comment="会话ID")
    ip_address = Column(String(45), comment="IP地址")
    user_agent = Column(String(500), comment="用户代理")
    
    # 业务上下文
    business_context = Column(JSON, comment="业务上下文")
    risk_level = Column(String(20), comment="风险级别")
    
    # 时间信息
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), comment="时间戳")
    
    # 索引
    __table_args__ = (
        Index('idx_decision_log_sequence', 'sequence_number'),
        Index('idx_decision_log_user', 'user_id'),
        Index('idx_decision_log_action', 'action'),
        Index('idx_decision_log_resource', 'resource', 'resource_id'),
        Index('idx_decision_log_timestamp', 'timestamp'),
        Index('idx_decision_log_request', 'request_id'),
        Index('idx_decision_log_hash', 'current_hash'),
    )
    
    def __repr__(self):
        return f"<DecisionLog(seq={self.sequence_number}, user='{self.user_id}', action='{self.action}')>"
    
    def calculate_hash(self) -> str:
        """计算当前记录哈希"""
        hash_data = f"{self.sequence_number}:{self.user_id}:{self.action}:{self.payload_hash}:{self.timestamp.isoformat()}"
        if self.previous_hash:
            hash_data = f"{self.previous_hash}:{hash_data}"
        return hashlib.sha256(hash_data.encode('utf-8')).hexdigest()
    
    def verify_chain(self, previous_record: Optional['DecisionLog'] = None) -> bool:
        """验证审计链完整性"""
        # 验证当前记录哈希
        expected_hash = self.calculate_hash()
        if self.current_hash != expected_hash:
            return False
        
        # 验证与前一条记录的链接
        if previous_record:
            return self.previous_hash == previous_record.current_hash
        
        return True


class User(Base):
    """用户表 - 简化的单用户模式"""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), unique=True, nullable=False, comment="用户名")
    email = Column(String(200), comment="邮箱")
    full_name = Column(String(200), comment="全名")
    
    # 认证信息 (简化版)
    is_active = Column(Boolean, default=True, comment="是否活跃")
    last_login = Column(DateTime(timezone=True), comment="最后登录时间")
    login_count = Column(Integer, default=0, comment="登录次数")
    
    # 偏好设置
    preferences = Column(JSON, comment="用户偏好")
    timezone = Column(String(50), default="Asia/Shanghai", comment="时区")
    language = Column(String(10), default="zh-CN", comment="语言")
    
    # 通知设置
    notification_settings = Column(JSON, comment="通知设置")
    
    # 时间戳
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # 索引
    __table_args__ = (
        Index('idx_user_username', 'username'),
        Index('idx_user_email', 'email'),
        Index('idx_user_active', 'is_active'),
    )
    
    def __repr__(self):
        return f"<User(username='{self.username}', name='{self.full_name}')>"


class AuditSummary(Base):
    """审计摘要表"""
    
    __tablename__ = "audit_summaries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 摘要信息
    summary_date = Column(Date, nullable=False, comment="摘要日期")
    summary_type = Column(String(50), comment="摘要类型 (daily/weekly/monthly)")
    
    # 统计数据
    total_operations = Column(Integer, default=0, comment="总操作数")
    unique_users = Column(Integer, default=0, comment="独立用户数")
    
    # 操作分类统计
    operations_by_action = Column(JSON, comment="按动作分类的操作统计")
    operations_by_resource = Column(JSON, comment="按资源分类的操作统计")
    operations_by_user = Column(JSON, comment="按用户分类的操作统计")
    
    # 风险统计
    high_risk_operations = Column(Integer, default=0, comment="高风险操作数")
    failed_operations = Column(Integer, default=0, comment="失败操作数")
    
    # 审计链验证
    chain_integrity_verified = Column(Boolean, comment="审计链完整性验证结果")
    last_verified_sequence = Column(Integer, comment="最后验证的序列号")
    verification_errors = Column(JSON, comment="验证错误列表")
    
    # Merkle根
    merkle_root = Column(String(64), comment="Merkle树根哈希")
    
    # 时间信息
    period_start = Column(DateTime(timezone=True), comment="统计周期开始")
    period_end = Column(DateTime(timezone=True), comment="统计周期结束")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 索引
    __table_args__ = (
        Index('idx_audit_summary_date', 'summary_date'),
        Index('idx_audit_summary_type', 'summary_type'),
        Index('idx_audit_summary_period', 'period_start', 'period_end'),
    )
    
    def __repr__(self):
        return f"<AuditSummary(date='{self.summary_date}', type='{self.summary_type}', ops={self.total_operations})>"


class DataLineage(Base):
    """数据血缘表"""
    
    __tablename__ = "data_lineage"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 数据实体
    entity_type = Column(String(50), nullable=False, comment="实体类型")
    entity_id = Column(UUID(as_uuid=True), nullable=False, comment="实体ID")
    field_name = Column(String(100), comment="字段名称")
    
    # 来源信息
    source_type = Column(String(50), comment="来源类型")
    source_id = Column(String(200), comment="来源ID")
    source_system = Column(String(100), comment="来源系统")
    
    # 转换信息
    transformation_rule = Column(JSON, comment="转换规则")
    transformation_code = Column(Text, comment="转换代码")
    
    # 质量信息
    data_quality_score = Column(Float, comment="数据质量评分")
    completeness = Column(Float, comment="完整性")
    accuracy = Column(Float, comment="准确性")
    timeliness = Column(Float, comment="及时性")
    
    # 时间信息
    effective_from = Column(DateTime(timezone=True), comment="生效开始时间")
    effective_to = Column(DateTime(timezone=True), comment="生效结束时间")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 索引
    __table_args__ = (
        Index('idx_data_lineage_entity', 'entity_type', 'entity_id'),
        Index('idx_data_lineage_source', 'source_type', 'source_id'),
        Index('idx_data_lineage_effective', 'effective_from', 'effective_to'),
    )
    
    def __repr__(self):
        return f"<DataLineage(entity='{self.entity_type}:{self.entity_id}', source='{self.source_type}')>"

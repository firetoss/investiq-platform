"""
InvestIQ Platform - 不可变审计链模型
实现不可破坏的审计追踪和决策记录
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Index, CheckConstraint, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.hybrid import hybrid_property
import uuid
import hashlib
import json

from backend.app.models.base import Base


class AuditChain(Base):
    """不可变审计链"""
    
    __tablename__ = "audit_chains"
    
    # 主键和链式结构
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sequence_number = Column(Integer, nullable=False, comment="序列号")
    previous_hash = Column(String(64), nullable=True, comment="上一条记录哈希")
    current_hash = Column(String(64), nullable=False, comment="当前记录哈希")
    
    # 操作信息
    operation_type = Column(String(50), nullable=False, comment="操作类型")
    operation_id = Column(String(100), nullable=True, comment="操作ID")
    user_id = Column(String(100), nullable=False, comment="用户ID")
    user_role = Column(String(50), nullable=False, comment="用户角色")
    request_id = Column(String(100), nullable=True, comment="请求ID")
    
    # 实体信息
    entity_type = Column(String(50), nullable=False, comment="实体类型")
    entity_id = Column(String(100), nullable=False, comment="实体ID")
    
    # 变更内容
    action = Column(String(50), nullable=False, comment="动作")
    before_value = Column(JSONB, nullable=True, comment="变更前值")
    after_value = Column(JSONB, nullable=True, comment="变更后值")
    change_summary = Column(Text, nullable=True, comment="变更摘要")
    
    # 业务上下文
    business_context = Column(JSONB, nullable=True, comment="业务上下文")
    decision_rationale = Column(Text, nullable=True, comment="决策理由")
    risk_assessment = Column(JSONB, nullable=True, comment="风险评估")
    
    # 技术上下文
    system_metadata = Column(JSONB, nullable=False, comment="系统元数据")
    payload_size = Column(Integer, nullable=False, comment="载荷大小")
    payload_hash = Column(String(64), nullable=False, comment="载荷哈希")
    
    # 时间戳 (不可变)
    occurred_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="发生时间")
    recorded_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="记录时间")
    
    # 审计状态
    is_verified = Column(Boolean, nullable=False, default=False, comment="是否已验证")
    verification_hash = Column(String(64), nullable=True, comment="验证哈希")
    merkle_root = Column(String(64), nullable=True, comment="Merkle根")
    
    # 合规信息
    compliance_tags = Column(JSONB, nullable=True, comment="合规标签")
    retention_period = Column(Integer, nullable=False, default=1825, comment="保留期限(天)")
    sensitivity_level = Column(String(20), nullable=False, default="normal", comment="敏感级别")
    
    # 约束: 序列号必须递增
    __table_args__ = (
        Index("ix_audit_chains_sequence", "sequence_number"),
        Index("ix_audit_chains_entity", "entity_type", "entity_id"),
        Index("ix_audit_chains_user", "user_id"),
        Index("ix_audit_chains_occurred", "occurred_at"),
        Index("ix_audit_chains_hash", "current_hash"),
        Index("ix_audit_chains_operation", "operation_type", "action"),
        CheckConstraint("sequence_number >= 0", name="check_sequence_positive"),
        CheckConstraint("payload_size >= 0", name="check_payload_size_positive"),
        CheckConstraint("retention_period > 0", name="check_retention_positive"),
        {"comment": "不可变审计链表"}
    )
    
    @hybrid_property
    def chain_integrity_valid(self) -> bool:
        """检查链完整性"""
        # 在数据库查询中需要特殊处理
        return self.current_hash == self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """计算当前记录的哈希值"""
        # 构建哈希输入
        hash_data = {
            "sequence_number": self.sequence_number,
            "previous_hash": self.previous_hash,
            "operation_type": self.operation_type,
            "user_id": self.user_id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "action": self.action,
            "before_value": self.before_value,
            "after_value": self.after_value,
            "payload_hash": self.payload_hash,
            "occurred_at": self.occurred_at.isoformat() if self.occurred_at else None,
        }
        
        # 生成确定性JSON并计算SHA-256
        json_data = json.dumps(hash_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(json_data.encode('utf-8')).hexdigest()
    
    def verify_chain_link(self, previous_record: Optional['AuditChain']) -> bool:
        """验证与前一条记录的链接"""
        if not previous_record:
            # 第一条记录
            return self.sequence_number == 0 and self.previous_hash is None
        
        return (
            self.sequence_number == previous_record.sequence_number + 1 and
            self.previous_hash == previous_record.current_hash
        )


class DecisionLog(Base):
    """决策日志 (特化的审计记录)"""
    
    __tablename__ = "decision_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    audit_chain_id = Column(UUID(as_uuid=True), nullable=False, comment="关联的审计链ID")
    
    # 决策信息
    decision_type = Column(String(50), nullable=False, comment="决策类型")
    decision_category = Column(String(50), nullable=False, comment="决策分类")
    decision_outcome = Column(String(100), nullable=False, comment="决策结果")
    confidence_level = Column(Integer, nullable=False, comment="置信度(0-100)")
    
    # 四闸门相关
    gate_check_results = Column(JSONB, nullable=True, comment="闸门检查结果")
    gates_passed = Column(Integer, nullable=True, comment="通过的闸门数")
    gates_failed = Column(JSONB, nullable=True, comment="失败的闸门详情")
    
    # 评分相关
    industry_score = Column(Float, nullable=True, comment="行业评分")
    equity_score = Column(Float, nullable=True, comment="个股评分")
    valuation_percentile = Column(Float, nullable=True, comment="估值分位")
    liquidity_check_passed = Column(Boolean, nullable=True, comment="流动性检查是否通过")
    
    # 投资决策相关
    investment_thesis = Column(Text, nullable=True, comment="投资论点")
    target_allocation = Column(Float, nullable=True, comment="目标配置")
    risk_tolerance = Column(String(20), nullable=True, comment="风险承受度")
    time_horizon = Column(String(20), nullable=True, comment="投资期限")
    
    # 证据支持
    supporting_evidence = Column(JSONB, nullable=True, comment="支持证据")
    evidence_quality_score = Column(Float, nullable=True, comment="证据质量评分")
    conflicting_evidence = Column(JSONB, nullable=True, comment="冲突证据")
    
    # 风险因素
    identified_risks = Column(JSONB, nullable=True, comment="识别的风险")
    mitigation_strategies = Column(JSONB, nullable=True, comment="风险缓解策略")
    maximum_loss_tolerance = Column(Float, nullable=True, comment="最大损失容忍度")
    
    # 审批流程
    requires_approval = Column(Boolean, nullable=False, default=False, comment="是否需要审批")
    approver_role = Column(String(50), nullable=True, comment="审批人角色")
    approval_status = Column(String(20), nullable=False, default="auto_approved", comment="审批状态")
    approved_by = Column(String(100), nullable=True, comment="审批人")
    approved_at = Column(DateTime, nullable=True, comment="审批时间")
    
    # 执行跟踪
    execution_status = Column(String(20), nullable=False, default="pending", comment="执行状态")
    executed_at = Column(DateTime, nullable=True, comment="执行时间")
    execution_details = Column(JSONB, nullable=True, comment="执行详情")
    
    # 结果跟踪
    outcome_tracked = Column(Boolean, nullable=False, default=False, comment="是否跟踪结果")
    actual_outcome = Column(JSONB, nullable=True, comment="实际结果")
    performance_metrics = Column(JSONB, nullable=True, comment="绩效指标")
    lessons_learned = Column(Text, nullable=True, comment="经验教训")
    
    # 时间戳
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("ix_decision_logs_audit_chain", "audit_chain_id"),
        Index("ix_decision_logs_type", "decision_type", "decision_category"),
        Index("ix_decision_logs_outcome", "decision_outcome"),
        Index("ix_decision_logs_confidence", "confidence_level"),
        Index("ix_decision_logs_approval", "approval_status"),
        Index("ix_decision_logs_execution", "execution_status"),
        Index("ix_decision_logs_created", "created_at"),
        {"comment": "投资决策日志表"}
    )


class AuditVerification(Base):
    """审计验证记录"""
    
    __tablename__ = "audit_verifications"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    verification_batch_id = Column(String(100), nullable=False, comment="验证批次ID")
    
    # 验证范围
    start_sequence = Column(Integer, nullable=False, comment="开始序列号")
    end_sequence = Column(Integer, nullable=False, comment="结束序列号")
    total_records = Column(Integer, nullable=False, comment="总记录数")
    
    # 验证结果
    verification_status = Column(String(20), nullable=False, comment="验证状态")
    integrity_check_passed = Column(Boolean, nullable=False, comment="完整性检查是否通过")
    chain_continuity_verified = Column(Boolean, nullable=False, comment="链连续性是否验证")
    hash_consistency_verified = Column(Boolean, nullable=False, comment="哈希一致性是否验证")
    
    # Merkle树验证
    merkle_root = Column(String(64), nullable=False, comment="Merkle根")
    merkle_tree_valid = Column(Boolean, nullable=False, comment="Merkle树是否有效")
    merkle_proof_samples = Column(JSONB, nullable=True, comment="Merkle证明样本")
    
    # 异常记录
    anomalies_detected = Column(Integer, nullable=False, default=0, comment="检测到的异常数")
    anomaly_details = Column(JSONB, nullable=True, comment="异常详情")
    corrupted_records = Column(JSONB, nullable=True, comment="损坏记录")
    
    # 验证元数据
    verification_algorithm = Column(String(50), nullable=False, comment="验证算法")
    verification_duration_ms = Column(Integer, nullable=False, comment="验证耗时(毫秒)")
    verified_by = Column(String(100), nullable=False, comment="验证人/系统")
    verification_signature = Column(String(512), nullable=True, comment="验证签名")
    
    # 时间戳
    started_at = Column(DateTime, nullable=False, comment="开始时间")
    completed_at = Column(DateTime, nullable=False, comment="完成时间")
    
    __table_args__ = (
        Index("ix_audit_verifications_batch", "verification_batch_id"),
        Index("ix_audit_verifications_sequence_range", "start_sequence", "end_sequence"),
        Index("ix_audit_verifications_status", "verification_status"),
        Index("ix_audit_verifications_completed", "completed_at"),
        CheckConstraint("start_sequence <= end_sequence", name="check_sequence_range"),
        CheckConstraint("total_records > 0", name="check_total_records_positive"),
        CheckConstraint("anomalies_detected >= 0", name="check_anomalies_non_negative"),
        {"comment": "审计验证记录表"}
    )


class ComplianceEvent(Base):
    """合规事件记录"""
    
    __tablename__ = "compliance_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    audit_chain_id = Column(UUID(as_uuid=True), nullable=True, comment="关联审计链ID")
    
    # 事件基本信息
    event_type = Column(String(50), nullable=False, comment="事件类型")
    event_category = Column(String(50), nullable=False, comment="事件分类")
    severity_level = Column(String(20), nullable=False, comment="严重级别")
    compliance_domain = Column(String(50), nullable=False, comment="合规领域")
    
    # 事件详情
    title = Column(String(500), nullable=False, comment="事件标题")
    description = Column(Text, nullable=False, comment="事件描述")
    affected_entities = Column(JSONB, nullable=True, comment="受影响实体")
    impact_assessment = Column(JSONB, nullable=True, comment="影响评估")
    
    # 法规相关
    relevant_regulations = Column(JSONB, nullable=True, comment="相关法规")
    regulatory_requirements = Column(JSONB, nullable=True, comment="监管要求")
    compliance_status = Column(String(20), nullable=False, comment="合规状态")
    
    # 响应和处理
    response_required = Column(Boolean, nullable=False, default=True, comment="是否需要响应")
    response_deadline = Column(DateTime, nullable=True, comment="响应截止时间")
    assigned_to = Column(String(100), nullable=True, comment="分配给")
    current_status = Column(String(20), nullable=False, default="open", comment="当前状态")
    
    # 处理记录
    actions_taken = Column(JSONB, nullable=True, comment="已采取行动")
    remediation_plan = Column(JSONB, nullable=True, comment="整改计划")
    prevention_measures = Column(JSONB, nullable=True, comment="预防措施")
    
    # 报告相关
    requires_external_reporting = Column(Boolean, nullable=False, default=False, comment="是否需要外部报告")
    reported_to = Column(JSONB, nullable=True, comment="已报告给")
    reporting_deadline = Column(DateTime, nullable=True, comment="报告截止时间")
    
    # 时间戳
    detected_at = Column(DateTime, nullable=False, comment="检测时间")
    reported_at = Column(DateTime, nullable=True, comment="报告时间")
    resolved_at = Column(DateTime, nullable=True, comment="解决时间")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("ix_compliance_events_type", "event_type", "event_category"),
        Index("ix_compliance_events_severity", "severity_level"),
        Index("ix_compliance_events_domain", "compliance_domain"),
        Index("ix_compliance_events_status", "current_status"),
        Index("ix_compliance_events_detected", "detected_at"),
        Index("ix_compliance_events_deadline", "response_deadline"),
        {"comment": "合规事件记录表"}
    )
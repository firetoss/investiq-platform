"""
InvestIQ Platform - 审计链服务
实现不可变审计追踪和链式验证
"""

import logging
import hashlib
import json
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any, Union
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func, and_, or_
from sqlalchemy.orm import selectinload

from backend.app.core.config import settings
from backend.app.core.exceptions import AuditException, ComplianceException
from backend.app.core.logging import get_logger, log_performance
from backend.app.models.audit import AuditChain, DecisionLog, AuditVerification, ComplianceEvent
from backend.app.models.snapshots import (
    IndustryScoreSnapshot, 
    EquityScoreSnapshot,
    ValuationPercentileSnapshot,
    TechnicalIndicatorSnapshot,
    PortfolioSnapshot,
    EvidenceSnapshot,
    ConfigurationSnapshot
)


logger = get_logger(__name__)


class ImmutableAuditChain:
    """不可变审计链管理器"""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.hash_algorithm = "SHA-256"
    
    @log_performance("audit_chain_append")
    async def append_record(
        self,
        operation_type: str,
        entity_type: str,
        entity_id: str,
        action: str,
        user_id: str,
        user_role: str,
        before_value: Optional[Dict] = None,
        after_value: Optional[Dict] = None,
        business_context: Optional[Dict] = None,
        decision_rationale: Optional[str] = None,
        risk_assessment: Optional[Dict] = None,
        request_id: Optional[str] = None,
        compliance_tags: Optional[List[str]] = None
    ) -> AuditChain:
        """
        向审计链追加新记录
        
        Args:
            operation_type: 操作类型 (scoring, gatekeeper, portfolio, etc.)
            entity_type: 实体类型 (industry, equity, portfolio, etc.)
            entity_id: 实体ID
            action: 动作 (create, update, delete, approve, etc.)
            user_id: 用户ID
            user_role: 用户角色
            before_value: 变更前值
            after_value: 变更后值
            business_context: 业务上下文
            decision_rationale: 决策理由
            risk_assessment: 风险评估
            request_id: 请求ID (用于关联)
            compliance_tags: 合规标签
        
        Returns:
            创建的审计链记录
        """
        try:
            # 获取最新的序列号
            last_record = await self._get_last_record()
            sequence_number = 0 if not last_record else last_record.sequence_number + 1
            previous_hash = None if not last_record else last_record.current_hash
            
            # 计算载荷哈希
            payload = {
                "before_value": before_value,
                "after_value": after_value,
                "business_context": business_context,
                "risk_assessment": risk_assessment
            }
            payload_hash = self._calculate_payload_hash(payload)
            payload_size = len(json.dumps(payload, ensure_ascii=False))
            
            # 构建系统元数据
            system_metadata = {
                "timestamp": datetime.utcnow().isoformat(),
                "hash_algorithm": self.hash_algorithm,
                "payload_size_bytes": payload_size,
                "python_version": "3.11",
                "service_version": settings.APP_VERSION
            }
            
            # 创建审计记录
            audit_record = AuditChain(
                sequence_number=sequence_number,
                previous_hash=previous_hash,
                operation_type=operation_type,
                user_id=user_id,
                user_role=user_role,
                request_id=request_id,
                entity_type=entity_type,
                entity_id=entity_id,
                action=action,
                before_value=before_value,
                after_value=after_value,
                change_summary=self._generate_change_summary(action, before_value, after_value),
                business_context=business_context,
                decision_rationale=decision_rationale,
                risk_assessment=risk_assessment,
                system_metadata=system_metadata,
                payload_size=payload_size,
                payload_hash=payload_hash,
                compliance_tags=compliance_tags or [],
                occurred_at=datetime.utcnow(),
                recorded_at=datetime.utcnow()
            )
            
            # 计算当前记录哈希
            audit_record.current_hash = audit_record.calculate_hash()
            
            # 验证链完整性
            if not audit_record.verify_chain_link(last_record):
                raise AuditException("审计链完整性验证失败")
            
            # 保存到数据库
            self.db_session.add(audit_record)
            await self.db_session.commit()
            await self.db_session.refresh(audit_record)
            
            logger.info(
                "Audit record appended",
                sequence_number=sequence_number,
                operation_type=operation_type,
                entity_type=entity_type,
                entity_id=entity_id,
                action=action,
                user_id=user_id
            )
            
            return audit_record
            
        except Exception as e:
            await self.db_session.rollback()
            logger.error(f"Failed to append audit record: {e}")
            raise AuditException(f"审计记录追加失败: {e}")
    
    async def _get_last_record(self) -> Optional[AuditChain]:
        """获取最后一条审计记录"""
        stmt = select(AuditChain).order_by(desc(AuditChain.sequence_number)).limit(1)
        result = await self.db_session.execute(stmt)
        return result.scalar_one_or_none()
    
    def _calculate_payload_hash(self, payload: Dict) -> str:
        """计算载荷哈希"""
        json_data = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(json_data.encode('utf-8')).hexdigest()
    
    def _generate_change_summary(
        self, 
        action: str, 
        before_value: Optional[Dict], 
        after_value: Optional[Dict]
    ) -> str:
        """生成变更摘要"""
        if action == "create":
            return f"创建新{after_value.get('type', '实体') if after_value else '实体'}"
        elif action == "update":
            if before_value and after_value:
                changes = []
                for key in set(before_value.keys()) | set(after_value.keys()):
                    old_val = before_value.get(key)
                    new_val = after_value.get(key)
                    if old_val != new_val:
                        changes.append(f"{key}: {old_val} → {new_val}")
                return f"更新字段: {', '.join(changes[:5])}" + ("..." if len(changes) > 5 else "")
            return "更新实体"
        elif action == "delete":
            return f"删除{before_value.get('type', '实体') if before_value else '实体'}"
        elif action in ["approve", "reject"]:
            return f"{action}操作"
        else:
            return f"执行{action}操作"
    
    @log_performance("audit_chain_verify")
    async def verify_chain_integrity(
        self,
        start_sequence: Optional[int] = None,
        end_sequence: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        验证审计链完整性
        
        Args:
            start_sequence: 开始序列号 (None表示从头开始)
            end_sequence: 结束序列号 (None表示到最后)
        
        Returns:
            验证结果
        """
        try:
            # 构建查询
            stmt = select(AuditChain).order_by(AuditChain.sequence_number)
            if start_sequence is not None:
                stmt = stmt.where(AuditChain.sequence_number >= start_sequence)
            if end_sequence is not None:
                stmt = stmt.where(AuditChain.sequence_number <= end_sequence)
            
            result = await self.db_session.execute(stmt)
            records = result.scalars().all()
            
            if not records:
                return {
                    "status": "empty",
                    "message": "没有找到审计记录",
                    "verified_count": 0,
                    "errors": []
                }
            
            # 验证每条记录
            errors = []
            previous_record = None
            
            for record in records:
                # 验证记录哈希
                calculated_hash = record.calculate_hash()
                if calculated_hash != record.current_hash:
                    errors.append({
                        "sequence": record.sequence_number,
                        "type": "hash_mismatch",
                        "message": f"记录哈希不匹配: {record.current_hash} != {calculated_hash}"
                    })
                
                # 验证链连接
                if not record.verify_chain_link(previous_record):
                    errors.append({
                        "sequence": record.sequence_number,
                        "type": "chain_broken",
                        "message": f"链连接断裂: 序列号或前置哈希不匹配"
                    })
                
                previous_record = record
            
            # 生成验证结果
            total_verified = len(records)
            integrity_valid = len(errors) == 0
            
            result = {
                "status": "valid" if integrity_valid else "invalid",
                "message": "审计链完整性验证通过" if integrity_valid else "发现完整性问题",
                "verified_count": total_verified,
                "start_sequence": records[0].sequence_number,
                "end_sequence": records[-1].sequence_number,
                "errors": errors,
                "error_count": len(errors),
                "integrity_score": (total_verified - len(errors)) / total_verified if total_verified > 0 else 0
            }
            
            # 记录验证结果
            await self._record_verification_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Chain integrity verification failed: {e}")
            raise AuditException(f"审计链完整性验证失败: {e}")
    
    async def _record_verification_result(self, verification_result: Dict[str, Any]):
        """记录验证结果"""
        try:
            verification_record = AuditVerification(
                verification_batch_id=f"verify_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                start_sequence=verification_result.get("start_sequence", 0),
                end_sequence=verification_result.get("end_sequence", 0),
                total_records=verification_result["verified_count"],
                verification_status=verification_result["status"],
                integrity_check_passed=verification_result["status"] == "valid",
                chain_continuity_verified=len([e for e in verification_result["errors"] if e["type"] == "chain_broken"]) == 0,
                hash_consistency_verified=len([e for e in verification_result["errors"] if e["type"] == "hash_mismatch"]) == 0,
                merkle_root="",  # TODO: 实现Merkle树
                merkle_tree_valid=True,
                anomalies_detected=verification_result["error_count"],
                anomaly_details=verification_result["errors"],
                verification_algorithm=self.hash_algorithm,
                verification_duration_ms=0,  # TODO: 实际计算
                verified_by="system",
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )
            
            self.db_session.add(verification_record)
            await self.db_session.commit()
            
        except Exception as e:
            logger.warning(f"Failed to record verification result: {e}")
    
    async def get_audit_trail(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        user_id: Optional[str] = None,
        operation_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditChain]:
        """
        获取审计轨迹
        
        Args:
            entity_type: 实体类型过滤
            entity_id: 实体ID过滤
            user_id: 用户ID过滤
            operation_type: 操作类型过滤
            start_date: 开始时间
            end_date: 结束时间
            limit: 结果限制
        
        Returns:
            审计记录列表
        """
        try:
            stmt = select(AuditChain)
            
            # 构建过滤条件
            conditions = []
            if entity_type:
                conditions.append(AuditChain.entity_type == entity_type)
            if entity_id:
                conditions.append(AuditChain.entity_id == entity_id)
            if user_id:
                conditions.append(AuditChain.user_id == user_id)
            if operation_type:
                conditions.append(AuditChain.operation_type == operation_type)
            if start_date:
                conditions.append(AuditChain.occurred_at >= start_date)
            if end_date:
                conditions.append(AuditChain.occurred_at <= end_date)
            
            if conditions:
                stmt = stmt.where(and_(*conditions))
            
            stmt = stmt.order_by(desc(AuditChain.occurred_at)).limit(limit)
            
            result = await self.db_session.execute(stmt)
            return result.scalars().all()
            
        except Exception as e:
            logger.error(f"Failed to get audit trail: {e}")
            raise AuditException(f"获取审计轨迹失败: {e}")
    
    async def export_audit_proof(
        self,
        entity_type: str,
        entity_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        导出审计证明材料
        
        Args:
            entity_type: 实体类型
            entity_id: 实体ID
            start_date: 开始时间
            end_date: 结束时间
        
        Returns:
            审计证明材料
        """
        try:
            # 获取相关的审计记录
            audit_records = await self.get_audit_trail(
                entity_type=entity_type,
                entity_id=entity_id,
                start_date=start_date,
                end_date=end_date,
                limit=1000
            )
            
            # 构建证明材料
            proof = {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "export_timestamp": datetime.utcnow().isoformat(),
                "period": {
                    "start": start_date.isoformat() if start_date else None,
                    "end": end_date.isoformat() if end_date else None
                },
                "total_records": len(audit_records),
                "records": []
            }
            
            # 处理每条记录
            for record in audit_records:
                record_proof = {
                    "sequence_number": record.sequence_number,
                    "current_hash": record.current_hash,
                    "previous_hash": record.previous_hash,
                    "operation_type": record.operation_type,
                    "action": record.action,
                    "user_id": record.user_id,
                    "user_role": record.user_role,
                    "occurred_at": record.occurred_at.isoformat(),
                    "change_summary": record.change_summary,
                    "compliance_tags": record.compliance_tags
                }
                proof["records"].append(record_proof)
            
            # 计算证明材料哈希
            proof_json = json.dumps(proof, sort_keys=True, ensure_ascii=False)
            proof["proof_hash"] = hashlib.sha256(proof_json.encode('utf-8')).hexdigest()
            
            return proof
            
        except Exception as e:
            logger.error(f"Failed to export audit proof: {e}")
            raise AuditException(f"导出审计证明失败: {e}")


class DecisionAuditService:
    """决策审计服务"""
    
    def __init__(self, db_session: AsyncSession, audit_chain: ImmutableAuditChain):
        self.db_session = db_session
        self.audit_chain = audit_chain
    
    @log_performance("decision_log_create")
    async def log_investment_decision(
        self,
        decision_type: str,
        decision_outcome: str,
        user_id: str,
        user_role: str,
        confidence_level: int,
        gate_check_results: Optional[Dict] = None,
        investment_thesis: Optional[str] = None,
        target_allocation: Optional[float] = None,
        supporting_evidence: Optional[List[Dict]] = None,
        identified_risks: Optional[List[Dict]] = None,
        requires_approval: bool = False,
        approver_role: Optional[str] = None,
        entity_type: str = "decision",
        entity_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> DecisionLog:
        """
        记录投资决策
        
        Args:
            decision_type: 决策类型 (entry, exit, rebalance, etc.)
            decision_outcome: 决策结果 (approved, rejected, deferred)
            user_id: 决策人ID
            user_role: 决策人角色
            confidence_level: 置信度 (0-100)
            gate_check_results: 闸门检查结果
            investment_thesis: 投资论点
            target_allocation: 目标配置
            supporting_evidence: 支持证据
            identified_risks: 识别的风险
            requires_approval: 是否需要审批
            approver_role: 审批人角色
            entity_type: 实体类型
            entity_id: 实体ID
            request_id: 请求ID
        
        Returns:
            决策日志记录
        """
        try:
            # 创建审计链记录
            audit_record = await self.audit_chain.append_record(
                operation_type="investment_decision",
                entity_type=entity_type,
                entity_id=entity_id or f"decision_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                action=decision_type,
                user_id=user_id,
                user_role=user_role,
                after_value={
                    "decision_outcome": decision_outcome,
                    "confidence_level": confidence_level,
                    "target_allocation": target_allocation
                },
                business_context={
                    "investment_thesis": investment_thesis,
                    "requires_approval": requires_approval,
                    "approver_role": approver_role
                },
                decision_rationale=investment_thesis,
                risk_assessment={
                    "identified_risks": identified_risks or [],
                    "risk_count": len(identified_risks) if identified_risks else 0
                },
                request_id=request_id,
                compliance_tags=["investment_decision", decision_type]
            )
            
            # 创建决策日志记录
            decision_log = DecisionLog(
                audit_chain_id=audit_record.id,
                decision_type=decision_type,
                decision_category="investment",
                decision_outcome=decision_outcome,
                confidence_level=confidence_level,
                gate_check_results=gate_check_results,
                gates_passed=gate_check_results.get("summary", {}).get("passed_gates", 0) if gate_check_results else None,
                gates_failed=gate_check_results.get("failed_gates") if gate_check_results else None,
                industry_score=gate_check_results.get("context", {}).get("industry_score") if gate_check_results else None,
                equity_score=gate_check_results.get("context", {}).get("equity_score") if gate_check_results else None,
                valuation_percentile=gate_check_results.get("context", {}).get("valuation_percentile") if gate_check_results else None,
                investment_thesis=investment_thesis,
                target_allocation=target_allocation,
                supporting_evidence=supporting_evidence,
                evidence_quality_score=self._calculate_evidence_quality(supporting_evidence) if supporting_evidence else None,
                identified_risks=identified_risks,
                requires_approval=requires_approval,
                approver_role=approver_role,
                approval_status="pending" if requires_approval else "auto_approved"
            )
            
            self.db_session.add(decision_log)
            await self.db_session.commit()
            await self.db_session.refresh(decision_log)
            
            logger.info(
                "Investment decision logged",
                decision_type=decision_type,
                decision_outcome=decision_outcome,
                confidence_level=confidence_level,
                requires_approval=requires_approval,
                user_id=user_id
            )
            
            return decision_log
            
        except Exception as e:
            await self.db_session.rollback()
            logger.error(f"Failed to log investment decision: {e}")
            raise AuditException(f"记录投资决策失败: {e}")
    
    def _calculate_evidence_quality(self, supporting_evidence: List[Dict]) -> float:
        """计算证据质量评分"""
        if not supporting_evidence:
            return 0.0
        
        total_score = 0
        total_weight = 0
        
        for evidence in supporting_evidence:
            score = evidence.get("quality_score", 80)  # 默认80分
            weight = evidence.get("weight", 1.0)       # 默认权重1.0
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    async def approve_decision(
        self,
        decision_id: str,
        approver_id: str,
        approver_role: str,
        approval_notes: Optional[str] = None
    ) -> DecisionLog:
        """批准决策"""
        try:
            # 获取决策记录
            stmt = select(DecisionLog).where(DecisionLog.id == decision_id)
            result = await self.db_session.execute(stmt)
            decision = result.scalar_one_or_none()
            
            if not decision:
                raise AuditException(f"决策记录不存在: {decision_id}")
            
            if decision.approval_status != "pending":
                raise AuditException(f"决策已被处理: {decision.approval_status}")
            
            # 更新决策状态
            old_status = decision.approval_status
            decision.approval_status = "approved"
            decision.approved_by = approver_id
            decision.approved_at = datetime.utcnow()
            
            # 记录到审计链
            await self.audit_chain.append_record(
                operation_type="decision_approval",
                entity_type="decision",
                entity_id=str(decision_id),
                action="approve",
                user_id=approver_id,
                user_role=approver_role,
                before_value={"approval_status": old_status},
                after_value={
                    "approval_status": "approved",
                    "approved_by": approver_id,
                    "approved_at": datetime.utcnow().isoformat()
                },
                decision_rationale=approval_notes
            )
            
            await self.db_session.commit()
            await self.db_session.refresh(decision)
            
            return decision
            
        except Exception as e:
            await self.db_session.rollback()
            logger.error(f"Failed to approve decision: {e}")
            raise AuditException(f"批准决策失败: {e}")


# 全局审计服务实例工厂
async def create_audit_services(db_session: AsyncSession) -> Tuple[ImmutableAuditChain, DecisionAuditService]:
    """创建审计服务实例"""
    audit_chain = ImmutableAuditChain(db_session)
    decision_service = DecisionAuditService(db_session, audit_chain)
    return audit_chain, decision_service
"""
SQLAlchemy models for causal evaluation benchmark.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, Boolean, 
    JSON, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


class User(Base):
    """User model for authentication and tracking."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    api_key = Column(String(255), unique=True, nullable=True, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    evaluation_sessions = relationship("EvaluationSession", back_populates="user")
    leaderboard_entries = relationship("Leaderboard", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"


class EvaluationSession(Base):
    """Evaluation session tracking multiple task executions."""
    
    __tablename__ = "evaluation_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), unique=True, nullable=False, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=True)
    
    # Session configuration
    total_tasks = Column(Integer, nullable=False, default=0)
    completed_tasks = Column(Integer, nullable=False, default=0)
    failed_tasks = Column(Integer, nullable=False, default=0)
    
    # Session metadata
    config = Column(JSON, nullable=True)  # Session configuration
    status = Column(String(20), nullable=False, default="running", index=True)  # running, completed, failed, cancelled
    
    # Timing
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="evaluation_sessions")
    task_executions = relationship("TaskExecution", back_populates="session")
    
    __table_args__ = (
        Index("idx_session_model_status", "model_name", "status"),
        Index("idx_session_dates", "started_at", "completed_at"),
    )
    
    def __repr__(self):
        return f"<EvaluationSession(id={self.id}, model='{self.model_name}', status='{self.status}')>"


class TaskExecution(Base):
    """Individual task execution within an evaluation session."""
    
    __tablename__ = "task_executions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    execution_id = Column(String(36), unique=True, nullable=False, index=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(Integer, ForeignKey("evaluation_sessions.id"), nullable=False)
    
    # Task information
    task_type = Column(String(50), nullable=False, index=True)  # attribution, counterfactual, intervention
    domain = Column(String(50), nullable=False, index=True)
    difficulty = Column(String(20), nullable=False, index=True)
    
    # Task specifics
    prompt = Column(Text, nullable=False)
    expected_outcome = Column(Text, nullable=True)
    task_metadata = Column(JSON, nullable=True)  # Task-specific data
    
    # Execution details
    status = Column(String(20), nullable=False, default="pending", index=True)  # pending, running, completed, failed
    error_message = Column(Text, nullable=True)
    
    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    session = relationship("EvaluationSession", back_populates="task_executions")
    model_response = relationship("ModelResponse", back_populates="task_execution", uselist=False)
    evaluation_result = relationship("EvaluationResult", back_populates="task_execution", uselist=False)
    
    __table_args__ = (
        Index("idx_task_type_domain", "task_type", "domain"),
        Index("idx_task_status_dates", "status", "started_at"),
        Index("idx_task_session_type", "session_id", "task_type"),
    )
    
    def __repr__(self):
        return f"<TaskExecution(id={self.id}, type='{self.task_type}', domain='{self.domain}', status='{self.status}')>"


class ModelResponse(Base):
    """Model response to a causal reasoning task."""
    
    __tablename__ = "model_responses"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_execution_id = Column(Integer, ForeignKey("task_executions.id"), nullable=False, unique=True)
    
    # Response content
    raw_response = Column(Text, nullable=False)
    parsed_response = Column(JSON, nullable=True)  # Structured parsed response
    
    # Response metadata
    tokens_used = Column(Integer, nullable=True)
    api_cost = Column(Float, nullable=True)
    model_temperature = Column(Float, nullable=True)
    model_params = Column(JSON, nullable=True)
    
    # Response timing
    response_time_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    task_execution = relationship("TaskExecution", back_populates="model_response")
    
    def __repr__(self):
        return f"<ModelResponse(id={self.id}, task_execution_id={self.task_execution_id})>"


class EvaluationResult(Base):
    """Evaluation result and scoring for a task execution."""
    
    __tablename__ = "evaluation_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_execution_id = Column(Integer, ForeignKey("task_executions.id"), nullable=False, unique=True)
    
    # Overall scoring
    overall_score = Column(Float, nullable=False, index=True)
    confidence = Column(Float, nullable=True)
    
    # Component scores (task-specific)
    relationship_score = Column(Float, nullable=True)  # For attribution tasks
    reasoning_score = Column(Float, nullable=True)
    outcome_score = Column(Float, nullable=True)  # For counterfactual tasks
    effect_score = Column(Float, nullable=True)  # For intervention tasks
    magnitude_score = Column(Float, nullable=True)
    causal_chain_score = Column(Float, nullable=True)
    side_effects_score = Column(Float, nullable=True)
    time_frame_score = Column(Float, nullable=True)
    assumptions_score = Column(Float, nullable=True)
    confounder_score = Column(Float, nullable=True)
    
    # Evaluation details
    expected_answer = Column(Text, nullable=True)
    predicted_answer = Column(Text, nullable=True)
    explanation = Column(Text, nullable=True)
    
    # Detailed analysis
    evaluation_metadata = Column(JSON, nullable=True)  # Full evaluation details
    error_analysis = Column(JSON, nullable=True)  # Error pattern analysis
    
    # Evaluation timing
    evaluation_duration_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    task_execution = relationship("TaskExecution", back_populates="evaluation_result")
    
    __table_args__ = (
        Index("idx_result_score", "overall_score"),
        Index("idx_result_task_score", "task_execution_id", "overall_score"),
    )
    
    def __repr__(self):
        return f"<EvaluationResult(id={self.id}, score={self.overall_score:.3f})>"


class Leaderboard(Base):
    """Leaderboard entries for model performance tracking."""
    
    __tablename__ = "leaderboard"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    session_id = Column(Integer, ForeignKey("evaluation_sessions.id"), nullable=False)
    
    # Model information
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=True)
    model_description = Column(Text, nullable=True)
    
    # Performance metrics
    overall_score = Column(Float, nullable=False, index=True)
    attribution_score = Column(Float, nullable=True)
    counterfactual_score = Column(Float, nullable=True)
    intervention_score = Column(Float, nullable=True)
    causal_chain_score = Column(Float, nullable=True)
    confounding_score = Column(Float, nullable=True)
    
    # Aggregate statistics
    total_evaluations = Column(Integer, nullable=False)
    successful_evaluations = Column(Integer, nullable=False)
    average_confidence = Column(Float, nullable=True)
    
    # Domain-specific scores
    domain_scores = Column(JSON, nullable=True)  # {"medical": 0.85, "education": 0.72, ...}
    
    # Submission metadata
    submission_date = Column(DateTime(timezone=True), server_default=func.now())
    is_public = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Performance details
    evaluation_metadata = Column(JSON, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="leaderboard_entries")
    session = relationship("EvaluationSession")
    
    __table_args__ = (
        Index("idx_leaderboard_score", "overall_score"),
        Index("idx_leaderboard_model", "model_name", "overall_score"),
        Index("idx_leaderboard_date", "submission_date"),
        UniqueConstraint("session_id", name="unique_session_leaderboard"),
    )
    
    def __repr__(self):
        return f"<Leaderboard(id={self.id}, model='{self.model_name}', score={self.overall_score:.3f})>"


class TestCase(Base):
    """Generated test cases for causal reasoning evaluation."""
    
    __tablename__ = "test_cases"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    test_id = Column(String(36), unique=True, nullable=False, index=True, default=lambda: str(uuid.uuid4()))
    
    # Test classification
    task_type = Column(String(50), nullable=False, index=True)
    domain = Column(String(50), nullable=False, index=True)
    difficulty = Column(String(20), nullable=False, index=True)
    
    # Test content
    prompt = Column(Text, nullable=False)
    expected_answer = Column(Text, nullable=False)
    ground_truth = Column(JSON, nullable=False)  # Structured ground truth
    
    # Test metadata
    generation_method = Column(String(50), nullable=False)  # manual, template, adversarial
    quality_score = Column(Float, nullable=True)
    validation_status = Column(String(20), default="pending")  # pending, validated, rejected
    
    # Usage statistics
    times_used = Column(Integer, default=0, nullable=False)
    average_model_score = Column(Float, nullable=True)
    
    # Timing
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index("idx_test_type_domain", "task_type", "domain", "difficulty"),
        Index("idx_test_quality", "quality_score"),
    )
    
    def __repr__(self):
        return f"<TestCase(id={self.id}, type='{self.task_type}', domain='{self.domain}')>"


class AuditLog(Base):
    """Audit log for tracking system events and changes."""
    
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Event information
    event_type = Column(String(50), nullable=False, index=True)  # evaluation_started, task_completed, etc.
    entity_type = Column(String(50), nullable=True)  # session, task, user
    entity_id = Column(Integer, nullable=True)
    
    # Event details
    event_data = Column(JSON, nullable=True)
    message = Column(Text, nullable=True)
    
    # Context
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(500), nullable=True)
    
    # Timing
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    __table_args__ = (
        Index("idx_audit_event_time", "event_type", "timestamp"),
        Index("idx_audit_entity", "entity_type", "entity_id"),
    )
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, event='{self.event_type}', timestamp='{self.timestamp}')>"
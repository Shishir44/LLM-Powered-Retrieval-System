"""Database models for the customer support platform."""

from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    JSON,
    ForeignKey,
    Float,
    Index,
    UniqueConstraint,
    CheckConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, ARRAY
import uuid


Base = declarative_base()


class TimestampMixin:
    """Mixin for adding timestamp columns."""
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class User(Base, TimestampMixin):
    """User model for customer support platform."""
    
    __tablename__ = "users"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(100), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_staff: Mapped[bool] = mapped_column(Boolean, default=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Profile information
    phone: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    avatar_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    timezone: Mapped[str] = mapped_column(String(50), default="UTC")
    language: Mapped[str] = mapped_column(String(10), default="en")
    
    # Relationships
    conversations: Mapped[List["Conversation"]] = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"


class Conversation(Base, TimestampMixin):
    """Conversation model for tracking customer interactions."""
    
    __tablename__ = "conversations"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
    )
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Conversation metadata
    title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20),
        default="active",
        nullable=False,
    )  # active, resolved, escalated
    priority: Mapped[str] = mapped_column(
        String(10),
        default="medium",
        nullable=False,
    )  # low, medium, high, urgent
    
    # Analytics
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    satisfaction_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    resolution_time: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # seconds
    
    # Metadata
    tags: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="conversations")
    messages: Mapped[List["Message"]] = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )
    
    __table_args__ = (
        Index("idx_conversation_user_id", "user_id"),
        Index("idx_conversation_session_id", "session_id"),
        Index("idx_conversation_status", "status"),
        Index("idx_conversation_created_at", "created_at"),
        CheckConstraint(
            "status IN ('active', 'resolved', 'escalated')",
            name="check_conversation_status",
        ),
        CheckConstraint(
            "priority IN ('low', 'medium', 'high', 'urgent')",
            name="check_conversation_priority",
        ),
    )
    
    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, user_id={self.user_id}, status={self.status})>"


class Message(Base, TimestampMixin):
    """Message model for individual conversation messages."""
    
    __tablename__ = "messages"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    conversation_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id"),
        nullable=False,
    )
    
    # Message content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    message_type: Mapped[str] = mapped_column(
        String(20),
        default="user",
        nullable=False,
    )  # user, assistant, system
    
    # Processing metadata
    intent: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    sentiment: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Response metadata
    response_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # seconds
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Additional metadata
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship("Message", back_populates="messages")
    
    __table_args__ = (
        Index("idx_message_conversation_id", "conversation_id"),
        Index("idx_message_type", "message_type"),
        Index("idx_message_created_at", "created_at"),
        CheckConstraint(
            "message_type IN ('user', 'assistant', 'system')",
            name="check_message_type",
        ),
    )
    
    def __repr__(self) -> str:
        return f"<Message(id={self.id}, conversation_id={self.conversation_id}, type={self.message_type})>"


class KnowledgeBase(Base, TimestampMixin):
    """Knowledge base model for storing documents and information."""
    
    __tablename__ = "knowledge_base"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    # Document metadata
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Categorization
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    subcategory: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    tags: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    
    # Status and versioning
    status: Mapped[str] = mapped_column(
        String(20),
        default="active",
        nullable=False,
    )  # active, archived, draft
    version: Mapped[int] = mapped_column(Integer, default=1)
    
    # Analytics
    view_count: Mapped[int] = mapped_column(Integer, default=0)
    helpful_count: Mapped[int] = mapped_column(Integer, default=0)
    not_helpful_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Vector store information
    vector_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Metadata
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    
    __table_args__ = (
        Index("idx_knowledge_base_category", "category"),
        Index("idx_knowledge_base_status", "status"),
        Index("idx_knowledge_base_title", "title"),
        Index("idx_knowledge_base_tags", "tags"),
        CheckConstraint(
            "status IN ('active', 'archived', 'draft')",
            name="check_knowledge_base_status",
        ),
    )
    
    def __repr__(self) -> str:
        return f"<KnowledgeBase(id={self.id}, title={self.title[:50]}...)>"


class Analytics(Base, TimestampMixin):
    """Analytics model for storing metrics and events."""
    
    __tablename__ = "analytics"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    # Event information
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    event_name: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Associated entities
    user_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=True,
    )
    conversation_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id"),
        nullable=True,
    )
    
    # Metrics
    value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    properties: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    
    # Timestamps
    event_timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    
    __table_args__ = (
        Index("idx_analytics_event_type", "event_type"),
        Index("idx_analytics_event_name", "event_name"),
        Index("idx_analytics_user_id", "user_id"),
        Index("idx_analytics_conversation_id", "conversation_id"),
        Index("idx_analytics_event_timestamp", "event_timestamp"),
    )
    
    def __repr__(self) -> str:
        return f"<Analytics(id={self.id}, event_type={self.event_type}, event_name={self.event_name})>"


class APIKey(Base, TimestampMixin):
    """API key model for external integrations."""
    
    __tablename__ = "api_keys"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    # Key information
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    prefix: Mapped[str] = mapped_column(String(20), nullable=False)
    
    # Permissions
    permissions: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    rate_limit: Mapped[int] = mapped_column(Integer, default=100)  # requests per minute
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Usage tracking
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Metadata
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    
    __table_args__ = (
        Index("idx_api_key_prefix", "prefix"),
        Index("idx_api_key_is_active", "is_active"),
        Index("idx_api_key_expires_at", "expires_at"),
    )
    
    def __repr__(self) -> str:
        return f"<APIKey(id={self.id}, name={self.name}, prefix={self.prefix})>"


# Create indexes for better query performance
def create_indexes(engine):
    """Create additional indexes for better performance."""
    # Full-text search indexes
    # These would be created via raw SQL for PostgreSQL
    pass
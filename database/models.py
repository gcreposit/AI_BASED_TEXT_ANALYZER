"""
Database models for the Multilingual Topic Clustering System
"""

from sqlalchemy import Column, String, Text, DateTime, Float, Integer, JSON, Boolean, ForeignKey, Date
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime, date

class Base(DeclarativeBase):
    pass


class Topic(Base):
    """Topic cluster model"""
    __tablename__ = 'topics'

    # Existing fields
    id = Column(String(36), primary_key=True)
    title = Column(String(500), nullable=False, index=True)
    description = Column(Text)
    primary_language = Column(String(50), nullable=False, index=True)
    content_count = Column(Integer, default=0)
    confidence_score = Column(Float)
    representative_text = Column(Text)
    created_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # NEW: Topic status fields
    status = Column(String(20), nullable=True, index=True, default='active')
    status_changed_at = Column(DateTime, nullable=True)
    status_changed_by = Column(String(255), nullable=True)
    status_reason = Column(Text, nullable=True)

    # NEW: Temporal tracking fields
    first_incident_date = Column(Date, nullable=True, index=True)
    last_incident_date = Column(Date, nullable=True, index=True)
    incident_count = Column(Integer, nullable=True, default=0)

    # Relationships
    text_entries = relationship("TextEntry", back_populates="topic", cascade="all, delete-orphan")


class TextEntry(Base):
    """Individual text entry model"""
    __tablename__ = 'text_entries'

    id = Column(Integer, primary_key=True, autoincrement=True)
    original_text = Column(Text, nullable=False)
    processed_text = Column(Text)
    enhanced_text = Column(Text)

    # Language detection
    detected_language = Column(String(50), index=True)
    language_confidence = Column(Float)

    # Source information
    source_type = Column(String(100))
    source_url = Column(String(500))

    # Topic assignment
    topic_id = Column(String(36), ForeignKey('topics.id'), index=True)
    similarity_score = Column(Float)
    confidence_level = Column(String(50))

    # Extracted information
    extracted_entities = Column(JSON)
    sentiment_data = Column(JSON)

    # Metadata
    user_id = Column(String(255))
    processing_time_ms = Column(Integer)
    notes = Column(Text, nullable=True)  # ✅ ADD THIS LINE

    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    topic = relationship("Topic", back_populates="text_entries")


class ProcessingLog(Base):
    """Processing log for monitoring and debugging"""
    __tablename__ = 'processing_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    operation_type = Column(String(100), nullable=False, index=True)
    status = Column(String(50), nullable=False, index=True)
    processing_time_ms = Column(Integer)
    error_message = Column(Text)
    user_id = Column(String(255))
    metadata_info = Column(JSON)
    created_at = Column(DateTime, default=func.now(), nullable=False, index=True)


class SystemStats(Base):
    """
    SystemStats model for storing daily/hourly system statistics
    """
    __tablename__ = "system_stats"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    date = Column(DateTime, nullable=False, index=True)
    period_type = Column(String(10), nullable=False)  # hourly, daily

    # Counts
    total_texts_processed = Column(Integer, default=0)
    new_topics_created = Column(Integer, default=0)
    topics_merged = Column(Integer, default=0)

    # Performance metrics
    avg_processing_time_ms = Column(Float)
    avg_similarity_score = Column(Float)

    # Language distribution
    language_distribution = Column(JSON)

    # Error counts
    processing_errors = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    def __repr__(self):
        return f"<SystemStats(date='{self.date}', period='{self.period_type}')>"

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'date': self.date.isoformat() if self.date else None,
            'period_type': self.period_type,
            'total_texts_processed': self.total_texts_processed,
            'new_topics_created': self.new_topics_created,
            'topics_merged': self.topics_merged,
            'avg_processing_time_ms': self.avg_processing_time_ms,
            'avg_similarity_score': self.avg_similarity_score,
            'language_distribution': self.language_distribution,
            'processing_errors': self.processing_errors,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
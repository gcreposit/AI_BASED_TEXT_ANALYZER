"""
Database models for the Multilingual Topic Clustering System
"""

from sqlalchemy import Column, String, Text, DateTime, Float, Integer, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime

Base = declarative_base()


class Topic(Base):
    """
    Topic model representing clustered text topics
    """
    __tablename__ = "topics"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    primary_language = Column(String(10), nullable=False, index=True)
    content_count = Column(Integer, default=1, nullable=False)
    confidence_score = Column(Float, default=0.0)
    embedding_model = Column(String(100), default="BAAI/bge-m3")
    similarity_threshold = Column(Float, default=0.80)
    representative_text = Column(Text)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False, index=True)

    # Relationships
    text_entries = relationship("TextEntry", back_populates="topic", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Topic(id='{self.id}', title='{self.title}', language='{self.primary_language}')>"

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'primary_language': self.primary_language,
            'content_count': self.content_count,
            'confidence_score': self.confidence_score,
            'embedding_model': self.embedding_model,
            'similarity_threshold': self.similarity_threshold,
            'representative_text': self.representative_text,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class TextEntry(Base):
    """
    TextEntry model representing individual processed texts
    """
    __tablename__ = "text_entries"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    original_text = Column(Text, nullable=False)
    processed_text = Column(Text)
    enhanced_text = Column(Text)

    # Language information
    detected_language = Column(String(20), index=True)
    language_confidence = Column(Float)

    # Processing meta_data
    source_type = Column(String(50), index=True)
    similarity_score = Column(Float)
    confidence_level = Column(String(10))  # high, medium, low
    processing_time_ms = Column(Integer)
    boost_reasons = Column(JSON)  # List of reasons for similarity boost

    # NER extracted data
    extracted_entities = Column(JSON)
    sentiment_data = Column(JSON)

    # User information
    user_id = Column(String(100), index=True)

    # Foreign key
    topic_id = Column(String(36), ForeignKey("topics.id"), nullable=False, index=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)

    # Relationships
    topic = relationship("Topic", back_populates="text_entries")

    def __repr__(self):
        return f"<TextEntry(id='{self.id}', language='{self.detected_language}', topic_id='{self.topic_id}')>"

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'original_text': self.original_text,
            'processed_text': self.processed_text,
            'enhanced_text': self.enhanced_text,
            'detected_language': self.detected_language,
            'language_confidence': self.language_confidence,
            'source_type': self.source_type,
            'similarity_score': self.similarity_score,
            'confidence_level': self.confidence_level,
            'processing_time_ms': self.processing_time_ms,
            'boost_reasons': self.boost_reasons,
            'extracted_entities': self.extracted_entities,
            'sentiment_data': self.sentiment_data,
            'user_id': self.user_id,
            'topic_id': self.topic_id,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class ProcessingLog(Base):
    """
    ProcessingLog model for system monitoring and debugging
    """
    __tablename__ = "processing_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    operation_type = Column(String(50), nullable=False, index=True)  # clustering, ner_extraction, embedding_generation
    status = Column(String(20), nullable=False, index=True)  # success, failure, partial
    processing_time_ms = Column(Integer)
    error_message = Column(Text)
    metadata_info = Column(JSON)

    # User and session information
    user_id = Column(String(100), index=True)
    session_id = Column(String(100))

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)

    def __repr__(self):
        return f"<ProcessingLog(id='{self.id}', operation='{self.operation_type}', status='{self.status}')>"

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'operation_type': self.operation_type,
            'status': self.status,
            'processing_time_ms': self.processing_time_ms,
            'error_message': self.error_message,
            'metadata_info': self.metadata_info,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


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
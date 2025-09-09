"""
Pydantic schemas for API request/response models
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
from datetime import datetime


class TextInput(BaseModel):
    """Schema for text input requests"""
    text: str = Field(
        ...,
        min_length=3,
        max_length=10000,
        description="Text content to be processed and clustered"
    )
    source_type: str = Field(
        default="unknown",
        description="Source type of the text (social_media, whatsapp, news, etc.)"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional user identifier for tracking and analytics"
    )

    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

    @validator('source_type')
    def validate_source_type(cls, v):
        allowed_sources = [
            'social_media', 'whatsapp', 'news', 'blog',
            'email', 'chat', 'forum', 'other', 'unknown'
        ]
        if v not in allowed_sources:
            return 'other'  # Default to 'other' for unknown types
        return v


class SentimentData(BaseModel):
    """Schema for sentiment analysis results"""
    label: str = Field(..., description="Sentiment label (positive, negative, neutral)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class ExtractedEntities(BaseModel):
    """Schema for NER extracted entities"""
    person_names: List[str] = Field(default_factory=list)
    organisation_names: List[str] = Field(default_factory=list)
    location_names: List[str] = Field(default_factory=list)
    district_names: List[str] = Field(default_factory=list)
    thana_names: List[str] = Field(default_factory=list)
    incidents: List[str] = Field(default_factory=list)
    caste_names: List[str] = Field(default_factory=list)
    religion_names: List[str] = Field(default_factory=list)
    hashtags: List[str] = Field(default_factory=list)
    mention_ids: List[str] = Field(default_factory=list)
    events: List[str] = Field(default_factory=list)
    sentiment: Optional[SentimentData] = None
    contextual_understanding: str = Field(default="")


class TopicClusteringResponse(BaseModel):
    """Schema for topic clustering response"""
    input_text: str = Field(..., description="Original input text")
    processed_text: str = Field(..., description="Preprocessed text")
    enhanced_text: Optional[str] = Field(None, description="Enhanced text with entity context")
    detected_language: str = Field(..., description="Detected language")
    language_confidence: float = Field(..., ge=0.0, le=1.0, description="Language detection confidence")
    action: str = Field(..., description="Action taken (grouped, new_topic_created, etc.)")
    topic_title: str = Field(..., description="Generated or assigned topic title")
    topic_id: str = Field(..., description="Unique topic identifier")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score to matched topic")
    confidence: str = Field(..., description="Overall confidence level (high, medium, low)")
    source_type: str = Field(..., description="Source type of the text")
    embedding_model: str = Field(default="BAAI/bge-m3", description="Embedding model used")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    extracted_entities: ExtractedEntities = Field(..., description="Extracted entities and meta_data")
    boost_reasons: List[str] = Field(default_factory=list, description="Reasons for similarity boost")
    timestamp: float = Field(..., description="Processing timestamp")


class TopicInfo(BaseModel):
    """Schema for topic information"""
    id: str = Field(..., description="Unique topic identifier")
    title: str = Field(..., description="Topic title")
    description: str = Field(default="", description="Topic description")
    primary_language: str = Field(..., description="Primary language of the topic")
    content_count: int = Field(..., description="Number of texts in this topic")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Topic confidence score")
    created_at: datetime = Field(..., description="Topic creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class SystemStats(BaseModel):
    """Schema for system statistics"""
    total_topics: int = Field(..., description="Total number of topics")
    total_texts: int = Field(..., description="Total number of processed texts")
    language_distribution: Dict[str, int] = Field(..., description="Distribution of languages")
    processing_performance: Dict[str, float] = Field(..., description="Performance metrics")
    uptime_hours: float = Field(..., description="System uptime in hours")


class ErrorResponse(BaseModel):
    """Schema for error responses"""
    error: str = Field(..., description="Error message")
    detail: str = Field(..., description="Detailed error information")
    timestamp: float = Field(..., description="Error timestamp")

    @classmethod
    def create(cls, error: str, detail: str = ""):
        import time
        return cls(
            error=error,
            detail=detail,
            timestamp=time.time()
        )


class HealthStatus(BaseModel):
    """Schema for health check response"""
    status: str = Field(..., description="Overall system status")
    timestamp: float = Field(..., description="Health check timestamp")
    version: str = Field(..., description="System version")
    services: Dict[str, str] = Field(..., description="Individual services statuses")


class SearchRequest(BaseModel):
    """Schema for search requests"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum similarity threshold")
    language: Optional[str] = Field(None, description="Filter by language")
    source_type: Optional[str] = Field(None, description="Filter by source type")


class SearchResult(BaseModel):
    """Schema for individual search results"""
    topic_id: str = Field(..., description="Topic identifier")
    similarity: float = Field(..., description="Similarity score")
    meta_data: Dict[str, Any] = Field(..., description="Topic meta_data")
    document: str = Field(..., description="Representative document")


class SearchResponse(BaseModel):
    """Schema for search response"""
    query: str = Field(..., description="Original search query")
    threshold: float = Field(..., description="Applied similarity threshold")
    total_results: int = Field(..., description="Number of results found")
    results: List[SearchResult] = Field(..., description="Search results")


class BatchProcessRequest(BaseModel):
    """Schema for batch processing requests"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to process")
    source_type: str = Field(default="unknown", description="Source type for all texts")
    user_id: Optional[str] = Field(None, description="User identifier")

    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('At least one text is required')

        valid_texts = []
        for text in v:
            if text and text.strip() and len(text.strip()) >= 3:
                valid_texts.append(text.strip())

        if not valid_texts:
            raise ValueError('No valid texts found (minimum 3 characters each)')

        return valid_texts


class BatchProcessResponse(BaseModel):
    """Schema for batch processing response"""
    total_processed: int = Field(..., description="Total number of texts processed")
    successful: int = Field(..., description="Number of successfully processed texts")
    failed: int = Field(..., description="Number of failed texts")
    processing_time_ms: int = Field(..., description="Total processing time")
    results: List[TopicClusteringResponse] = Field(..., description="Individual processing results")
    errors: List[str] = Field(default_factory=list, description="Error messages for failed texts")


class TopicUpdateRequest(BaseModel):
    """Schema for topic update requests"""
    title: Optional[str] = Field(None, max_length=255, description="New topic title")
    description: Optional[str] = Field(None, max_length=1000, description="New topic description")

    @validator('title')
    def validate_title(cls, v):
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError('Title cannot be empty')
            if len(v.split()) > 10:
                raise ValueError('Title cannot exceed 10 words')
        return v


class TopicMergeRequest(BaseModel):
    """Schema for topic merge requests"""
    source_topic_id: str = Field(..., description="ID of topic to merge from")
    target_topic_id: str = Field(..., description="ID of topic to merge into")
    new_title: Optional[str] = Field(None, description="New title for merged topic")

    @validator('source_topic_id', 'target_topic_id')
    def validate_topic_ids(cls, v):
        if not v or not v.strip():
            raise ValueError('Topic ID cannot be empty')
        return v.strip()


class AnalyticsRequest(BaseModel):
    """Schema for analytics requests"""
    start_date: Optional[datetime] = Field(None, description="Start date for analytics")
    end_date: Optional[datetime] = Field(None, description="End date for analytics")
    language: Optional[str] = Field(None, description="Filter by language")
    source_type: Optional[str] = Field(None, description="Filter by source type")

    @validator('end_date')
    def validate_date_range(cls, v, values):
        if v and 'start_date' in values and values['start_date']:
            if v < values['start_date']:
                raise ValueError('End date must be after start date')
        return v


class AnalyticsResponse(BaseModel):
    """Schema for analytics response"""
    period: Dict[str, datetime] = Field(..., description="Analysis period")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")
    trends: Dict[str, Any] = Field(..., description="Trend analysis")
    top_topics: List[Dict[str, Any]] = Field(..., description="Top topics by activity")
    language_breakdown: Dict[str, int] = Field(..., description="Language distribution")
    source_breakdown: Dict[str, int] = Field(..., description="Source type distribution")
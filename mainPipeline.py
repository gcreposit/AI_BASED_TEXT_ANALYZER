#!/usr/bin/env python3
"""
Main Pipeline for Processing Twitter Scrapper Data
This module handles the automated processing of posts from the dump database,
analyzes them using the text processing API, and stores results.
"""

import os
import sys
import time
import json
import logging
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

import pymysql
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, BigInteger, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Database configuration for DUMP DATABASE
DUMP_DB_CONFIG = {
    'host': os.getenv('DUMP_DB_HOST', '94.136.189.147'),
    'database': os.getenv('DUMP_DB_NAME', 'twitter_scrapper'),
    'user': os.getenv('DUMP_DB_USER', 'gccloud'),
    'password': os.getenv('DUMP_DB_PASSWORD', 'Gccloud@1489$'),
    'port': int(os.getenv('DUMP_DB_PORT', '3306'))
}

# API Configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')
PROCESS_TEXT_ENDPOINT = f"{API_BASE_URL}/api/process-text"
PROCESS_BATCH_ENDPOINT = f"{API_BASE_URL}/api/process-batch"

# Create database engine for dump database
encoded_password = quote_plus(DUMP_DB_CONFIG['password'])
DUMP_DATABASE_URL = f"mysql+pymysql://{DUMP_DB_CONFIG['user']}:{encoded_password}@{DUMP_DB_CONFIG['host']}:{DUMP_DB_CONFIG['port']}/{DUMP_DB_CONFIG['database']}"

engine = create_engine(DUMP_DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PostBank(Base):
    """Model for post_bank table"""
    __tablename__ = 'post_bank'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    post_title = Column(Text, nullable=False)
    post_snippet = Column(Text, nullable=False)
    post_url = Column(Text, nullable=False)
    core_source = Column(String(255))
    source = Column(String(255), default='YouTube')
    post_timestamp = Column(DateTime, default=datetime.utcnow)
    author_name = Column(String(255), nullable=False)
    author_username = Column(String(255), nullable=False)
    post_language = Column(String(50))
    post_location = Column(String(255))
    post_type = Column(String(100), default='video')
    retweets = Column(Integer, default=0)
    bookmarks = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    likes = Column(Integer, default=0)
    views = Column(BigInteger, default=0)
    attachments = Column(Text)
    mention_ids = Column(Text)
    mention_hashtags = Column(Text)
    keyword = Column(String(255))
    unique_hash = Column(String(32))
    video_id = Column(String(50))
    duration = Column(String(20))
    category_id = Column(String(10))
    channel_id = Column(String(50))
    
    # Analysis status column
    analysisStatus = Column(String(20), default='NOT_ANALYZED')


class Topic(Base):
    """Model for topic table to store unique topic combinations"""
    __tablename__ = 'topic'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Topic identification fields (stored as JSON arrays in longTEXT format)
    primary_districts = Column(Text)  # longTEXT - stored as [,,,,] format
    primary_thana = Column(Text)  # longTEXT - stored as [,,,,] format  
    primary_location = Column(Text)  # longTEXT - stored as [,,,,] format
    broad_category = Column(Text)  # longTEXT - stored as [,,,,] format
    sub_category = Column(Text)  # longTEXT - stored as [,,,,] format
    keywords_cloud = Column(Text)  # longTEXT - stored as [,,,,] format
    category_reasoning = Column(Text)  # longTEXT - stored as [,,,,] format
    hashtags = Column(Text)  # longTEXT - stored as [,,,,] format
    mention_id_extraction = Column(Text)  # longTEXT - stored as [,,,,] format
    
    # Count fields
    read_count = Column(Integer, default=0)  # Default to 0
    unread_count = Column(Integer, default=0)  # Increments when topic appears in analyzed_data
    total_no_of_post = Column(Integer, default=0)  # Increments when topic appears in analyzed_data
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AnalyzedData(Base):
    """Model for analyzed_data table to store API response data"""
    __tablename__ = 'analyzed_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    dump_table_id = Column(Integer, nullable=False)  # Reference to post_bank.id
    topic_id = Column(Integer, nullable=True)  # Foreign key to topic table (1:M relationship)
    
    # Text processing fields
    input_text = Column(Text)
    processed_text = Column(Text)
    enhanced_text = Column(Text)
    detected_language = Column(String(10))
    language_confidence = Column(Float)
    action = Column(String(50))
    topic_title = Column(Text)
    similarity_score = Column(Float)
    confidence = Column(String(20))
    source_type = Column(String(50))
    embedding_model = Column(String(100))
    processing_time_ms = Column(Integer)
    boost_reasons = Column(Text)  # JSON array as text
    timestamp = Column(Float)
    
    # PostBank fields with post_bank_ prefix - Core content fields
    post_bank_post_title = Column(Text)  # Post title from PostBank
    post_bank_post_snippet = Column(Text)  # Post snippet/content from PostBank
    post_bank_post_url = Column(Text)  # Post URL from PostBank
    post_bank_core_source = Column(String(255))  # Core source from PostBank
    post_bank_source = Column(String(255))  # Source from PostBank
    post_bank_post_timestamp = Column(DateTime)  # Original timestamp from PostBank
    
    # PostBank fields - Author information
    post_bank_author_name = Column(String(255))  # Author name from PostBank
    post_bank_author_username = Column(String(255))  # Author username from PostBank
    post_bank_author_id = Column(String(255))  # Author ID from PostBank

    # PostBank fields - Language and location
    post_bank_post_language = Column(String(50))  # Post language from PostBank
    post_bank_post_location = Column(String(255))  # Post location from PostBank
    post_bank_post_type = Column(String(100))  # Post type from PostBank

    # PostBank fields - Social media metrics
    post_bank_retweets = Column(Integer, default=0)  # Retweets from PostBank
    post_bank_bookmarks = Column(Integer, default=0)  # Bookmarks from PostBank
    post_bank_comments = Column(Integer, default=0)  # Comments from PostBank
    post_bank_likes = Column(Integer, default=0)  # Likes from PostBank
    post_bank_views = Column(BigInteger, default=0)  # Views from PostBank
    
    # PostBank fields - Metadata
    post_bank_attachments = Column(Text)  # Attachments from PostBank
    post_bank_mention_ids = Column(Text)  # Mention IDs from PostBank
    post_bank_mention_hashtags = Column(Text)  # Mention hashtags from PostBank
    post_bank_keyword = Column(String(255))  # Keyword from PostBank
    post_bank_unique_hash = Column(String(32))  # Unique hash from PostBank
    post_bank_video_id = Column(String(50))  # Video ID from PostBank
    post_bank_duration = Column(String(20))  # Duration from PostBank
    post_bank_category_id = Column(String(10))  # Category ID from PostBank
    post_bank_channel_id = Column(String(50))  # Channel ID from PostBank
    post_bank_post_id = Column(String(100))  # Post ID from PostBank
    
    # Additional fields from logs.txt model (WhatsApp specific)
    post_date = Column(String(20))  # Date in dd/mm/yyyy format
    post_time = Column(String(20))  # Time in 24-hour format
    mobile_number = Column(String(20))  # Mobile number from WhatsApp
    group_id = Column(String(100))  # WhatsApp group ID
    reply_to_message_id = Column(String(100))  # ID of message being replied to
    reply_text = Column(Text)  # Text of the message being replied to
    photo_attachment = Column(Boolean, default=False)  # Indicates photo attachment
    video_attachment = Column(Boolean, default=False)  # Indicates video attachment
    common_attachment_id = Column(Integer)  # Reference to common_attachments table
    
    # Extracted entities
    person_names = Column(Text)  # JSON array as text
    organisation_names = Column(Text)  # JSON array as text
    location_names = Column(Text)  # JSON array as text
    district_names = Column(Text)  # JSON array as text
    thana_names = Column(Text)  # JSON array as text
    incidents = Column(Text)  # JSON array as text
    caste_names = Column(Text)  # JSON array as text
    religion_names = Column(Text)  # JSON array as text
    hashtags = Column(Text)  # JSON array as text
    mention_ids_extracted = Column(Text)  # JSON array as text
    events = Column(Text)  # JSON array as text
    sentiment_label = Column(String(20))
    sentiment_confidence = Column(Float)
    contextual_understanding = Column(Text)
    
    # Location fields from incident_location_analysis
    primary_district = Column(Text)  # JSON array as text - primary districts
    primary_thana = Column(Text)  # JSON array as text - primary thanas
    primary_location = Column(Text)  # JSON array as text - primary locations
    
    # Category classification fields (ordered for display: category, subcategory, keywords, reasoning)
    broad_category = Column(Text)  # JSON array as text - broad categories
    sub_category = Column(Text)  # JSON array as text - sub categories
    keywords_cloud = Column(Text)  # JSON array as text - keywords
    category_reasoning = Column(Text)  # JSON array as text - reasoning for categories
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PipelineProcessor:
    """Main pipeline processor class"""
    
    def __init__(self):
        self.session: Optional[Session] = None
        self.processing_queue = deque()
        self.batch_size = 10  # Process in batches for efficiency
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
    def __enter__(self):
        """Context manager entry"""
        self.session = SessionLocal()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.session:
            self.session.close()
            
    def create_tables(self) -> bool:
        """Create necessary tables if they don't exist"""
        try:
            # Create analyzed_data table
            Base.metadata.create_all(bind=engine)
            logger.info("Tables created successfully")
            
            # Add analysisStatus column to post_bank if it doesn't exist
            self._add_analysis_status_column()
            return True
            
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            return False
            
    def _add_analysis_status_column(self):
        """Add analysisStatus column to post_bank table"""
        try:
            # Check if column exists
            result = self.session.execute(
                text("SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS "
                     "WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = 'post_bank' "
                     "AND COLUMN_NAME = 'analysisStatus'"),
                {'schema': DUMP_DB_CONFIG['database']}
            )
            
            if result.fetchone()[0] == 0:
                # Column doesn't exist, add it
                self.session.execute(
                    text("ALTER TABLE post_bank ADD COLUMN analysisStatus VARCHAR(20) DEFAULT 'NOT_ANALYZED'")
                )
                self.session.commit()
                logger.info("Added analysisStatus column to post_bank table")
            else:
                logger.info("analysisStatus column already exists in post_bank table")
                
        except Exception as e:
            logger.error(f"Error adding analysisStatus column: {str(e)}")

    def get_unanalyzed_posts(self, limit: int = None) -> List[PostBank]:
        """Get posts that haven't been analyzed yet"""
        try:
            query = self.session.query(PostBank).filter(
                PostBank.analysisStatus == 'NOT_ANALYZED'
            )
            
            if limit:
                query = query.limit(limit)
                
            posts = query.all()
            logger.info(f"Found {len(posts)} unanalyzed posts")
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching unanalyzed posts: {str(e)}")
            return []
            
    def call_process_batch_api(self, texts: List[str], source_type: str = "social_media") -> Optional[Dict[str, Any]]:
        """
        Call the batch processing API endpoint
        
        Args:
            texts: List of text strings to process
            source_type: Type of source (default: social_media)
            
        Returns:
            API response dictionary or None if failed
        """
        try:
            payload = {
                "texts": texts,
                "source_type": source_type
            }
            
            logger.info(f"Calling batch API with {len(texts)} texts")
            response = requests.post(
                PROCESS_BATCH_ENDPOINT,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=300  # 5 minutes timeout for batch processing
            )
            
            if response.status_code == 200:
                api_response = response.json()
                logger.info(f"Batch API call successful. Processed: {api_response.get('total_processed', 0)}")
                return api_response
            else:
                logger.error(f"API call failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error during API call: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during API call: {str(e)}")
            return None

    def call_process_text_api(self, text: str, source_type: str = "social_media") -> Optional[Dict[str, Any]]:
        """Call the /api/process-text endpoint"""
        try:
            payload = {
                "text": text,
                "source_type": source_type,
                "user_id": "pipeline_processor"
            }
            
            response = requests.post(
                PROCESS_TEXT_ENDPOINT,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API call failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in API call: {str(e)}")
            return None
            
    def get_or_create_topic(self, topic_data: Dict[str, Any]) -> Optional[int]:
        """Get existing topic or create new one based on unique combination of fields"""
        try:
            # Extract topic fields from API response
            entities = topic_data.get('extracted_entities', {})
            incident_location = entities.get('incident_location_analysis', {})
            category_classifications = entities.get('category_classifications', [])
            primary_classification = entities.get('primary_classification', {})
            
            # Extract primary location data
            primary_districts = []
            primary_thanas = []
            primary_locations = []
            
            if incident_location:
                primary_districts = incident_location.get('primary_districts', [])
                primary_thanas = incident_location.get('primary_thanas', [])
                primary_locations = incident_location.get('primary_locations', [])
            
            # Extract category data
            broad_categories = []
            sub_categories = []
            keywords_clouds = []
            category_reasonings = []
            
            # Handle category_classifications as list
            if isinstance(category_classifications, list) and category_classifications:
                for classification in category_classifications:
                    if isinstance(classification, dict):
                        broad_cat = classification.get('broad_category', '')
                        sub_cat = classification.get('sub_category', '')
                        reasoning = classification.get('reasoning', '')
                        
                        # Handle nested matched_keywords structure
                        matched_keywords_data = classification.get('matched_keywords', [])
                        
                        if broad_cat:
                            broad_categories.append(broad_cat)
                        if sub_cat:
                            sub_categories.append(sub_cat)
                        if reasoning:
                            category_reasonings.append(reasoning)
                        
                        # Handle nested matched_keywords structure
                        if isinstance(matched_keywords_data, dict) and 'matched_keywords' in matched_keywords_data:
                            nested_keywords = matched_keywords_data.get('matched_keywords', [])
                            if nested_keywords:
                                keywords_clouds.extend(nested_keywords)
                        elif isinstance(matched_keywords_data, list) and matched_keywords_data:
                            keywords_clouds.extend(matched_keywords_data)
            
            # Also include primary classification data if available
            if primary_classification and not broad_categories:
                primary_broad = primary_classification.get('broad_category', '')
                primary_sub = primary_classification.get('sub_category', '')
                primary_reasoning = primary_classification.get('reasoning', '')
                
                if primary_broad:
                    broad_categories.append(primary_broad)
                if primary_sub:
                    sub_categories.append(primary_sub)
                if primary_reasoning:
                    category_reasonings.append(primary_reasoning)
            
            # Extract hashtags and mention_ids
            hashtags = entities.get('hashtags', [])
            mention_ids = entities.get('mention_ids', [])
            
            # Convert arrays to [,,,,] format for storage
            primary_districts_str = json.dumps(primary_districts, ensure_ascii=False) if primary_districts else "[]"
            primary_thanas_str = json.dumps(primary_thanas, ensure_ascii=False) if primary_thanas else "[]"
            primary_locations_str = json.dumps(primary_locations, ensure_ascii=False) if primary_locations else "[]"
            broad_categories_str = json.dumps(broad_categories, ensure_ascii=False) if broad_categories else "[]"
            sub_categories_str = json.dumps(sub_categories, ensure_ascii=False) if sub_categories else "[]"
            keywords_clouds_str = json.dumps(keywords_clouds, ensure_ascii=False) if keywords_clouds else "[]"
            category_reasonings_str = json.dumps(category_reasonings, ensure_ascii=False) if category_reasonings else "[]"
            hashtags_str = json.dumps(hashtags, ensure_ascii=False) if hashtags else "[]"
            mention_ids_str = json.dumps(mention_ids, ensure_ascii=False) if mention_ids else "[]"
            
            # Check if topic already exists with same unique combination
            existing_topic = self.session.query(Topic).filter(
                Topic.primary_districts == primary_districts_str,
                Topic.broad_category == broad_categories_str,
                Topic.sub_category == sub_categories_str,
                Topic.keywords_cloud == keywords_clouds_str,
                Topic.category_reasoning == category_reasonings_str,
                Topic.hashtags == hashtags_str,
                Topic.mention_id_extraction == mention_ids_str
            ).first()
            
            if existing_topic:
                # Update counts for existing topic
                existing_topic.unread_count += 1
                existing_topic.total_no_of_post += 1
                self.session.commit()
                logger.info(f"Updated existing topic {existing_topic.id} - unread_count: {existing_topic.unread_count}, total_posts: {existing_topic.total_no_of_post}")
                return existing_topic.id
            else:
                # Create new topic
                new_topic = Topic(
                    primary_districts=primary_districts_str,
                    primary_thana=primary_thanas_str,
                    primary_location=primary_locations_str,
                    broad_category=broad_categories_str,
                    sub_category=sub_categories_str,
                    keywords_cloud=keywords_clouds_str,
                    category_reasoning=category_reasonings_str,
                    hashtags=hashtags_str,
                    mention_id_extraction=mention_ids_str,
                    read_count=0,
                    unread_count=1,  # First occurrence
                    total_no_of_post=1  # First occurrence
                )
                
                self.session.add(new_topic)
                self.session.flush()  # Get the ID without committing
                logger.info(f"Created new topic {new_topic.id} with initial counts")
                return new_topic.id
                
        except Exception as e:
            logger.error(f"Error in get_or_create_topic: {str(e)}")
            return None

    def get_common_attachment_id(self, post_bank_id: int) -> Optional[int]:
        """Get common_attachment_id from common_attachments table based on post_bank_id"""
        try:
            result = self.session.execute(text("""
                SELECT id FROM common_attachments WHERE post_bank_id = :post_bank_id LIMIT 1
            """), {"post_bank_id": post_bank_id})
            
            row = result.fetchone()
            return row[0] if row else None
            
        except Exception as e:
            logger.error(f"Error getting common_attachment_id for post_bank_id {post_bank_id}: {str(e)}")
            return None

    def save_batch_analyzed_data(self, posts: List[PostBank], api_response: Dict[str, Any]) -> bool:
        """
        Save batch analyzed data to analyzed_data table and update analysisStatus
        
        Args:
            posts: List of original PostBank objects
            api_response: API response containing analyzed results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            results = api_response.get('results', [])
            
            if len(results) != len(posts):
                logger.error(f"Mismatch between posts ({len(posts)}) and results ({len(results)})")
                return False
            
            successfully_processed_post_ids = []
            
            for i, (post, result) in enumerate(zip(posts, results)):
                try:
                    # Extract sentiment data
                    entities = result.get('extracted_entities', {})
                    sentiment_data = entities.get('sentiment', {})
                    sentiment_label = sentiment_data.get('label', 'neutral')
                    sentiment_confidence = sentiment_data.get('confidence', 0.0)
                    
                    # Extract category classifications and location analysis
                    category_classifications = entities.get('category_classifications', [])
                    primary_classification = entities.get('primary_classification', {})
                    incident_location_analysis = entities.get('incident_location_analysis', {})
                    
                    # Extract location data from incident_location_analysis
                    primary_districts = []
                    primary_thanas = []
                    primary_locations = []
                    
                    if incident_location_analysis:
                        # Handle nested primary_location structure
                        primary_location_data = incident_location_analysis.get('primary_location', {})
                        
                        if isinstance(primary_location_data, dict):
                            # Extract from nested structure: {"district": "बुलंदशहर", "thana": "", "specific_location": "", "matched_keywords": ["keyword1", "keyword2"]}
                            district = primary_location_data.get('district', '')
                            thana = primary_location_data.get('thana', '')
                            specific_location = primary_location_data.get('specific_location', '')
                            location_matched_keywords = primary_location_data.get('matched_keywords', [])
                            
                            # Store district in primary_district
                            if district:
                                primary_districts.append(district)
                            # Store thana in primary_thana
                            if thana:
                                primary_thanas.append(thana)
                            # Store only specific_location in primary_location
                            if specific_location:
                                primary_locations.append(specific_location)
                            
                            # Store matched_keywords from primary_location for later use in keywords_cloud
                            primary_location_keywords = location_matched_keywords if isinstance(location_matched_keywords, list) else []
                        elif isinstance(primary_location_data, list):
                            # Handle if it's still an array format (backward compatibility)
                            primary_locations = primary_location_data
                        
                        # Also check for direct arrays (backward compatibility)
                        districts_data = incident_location_analysis.get('primary_district', [])
                        thanas_data = incident_location_analysis.get('primary_thana', [])
                        
                        if isinstance(districts_data, list) and districts_data:
                            primary_districts.extend(districts_data)
                        if isinstance(thanas_data, list) and thanas_data:
                            primary_thanas.extend(thanas_data)
                    
                    # Extract category data directly from entities
                    broad_categories = []
                    sub_categories = []
                    keywords_clouds = []
                    category_reasonings = []
                    
                    # Check if entities has direct category data
                    if 'broad_category' in entities:
                        broad_cat_data = entities.get('broad_category', [])
                        broad_categories = broad_cat_data if isinstance(broad_cat_data, list) else []
                    
                    if 'sub_category' in entities:
                        sub_cat_data = entities.get('sub_category', [])
                        sub_categories = sub_cat_data if isinstance(sub_cat_data, list) else []
                    
                    if 'keywords_cloud' in entities:
                        keywords_data = entities.get('keywords_cloud', [])
                        if isinstance(keywords_data, dict) and 'matched_keywords' in keywords_data:
                            # Handle nested structure: {"matched_keywords": ["keyword1", "keyword2"]}
                            keywords_clouds = keywords_data.get('matched_keywords', [])
                        elif isinstance(keywords_data, list):
                            # Handle direct array format
                            keywords_clouds = keywords_data
                        else:
                            keywords_clouds = []
                    
                    if 'category_reasoning' in entities:
                        reasoning_data = entities.get('category_reasoning', [])
                        category_reasonings = reasoning_data if isinstance(reasoning_data, list) else []
                    
                    # Fallback: Process category_classifications if direct data not available
                    if not broad_categories and category_classifications:
                        for classification in category_classifications:
                            if isinstance(classification, dict):
                                broad_cat = classification.get('broad_category', '')
                                sub_cat = classification.get('sub_category', '')
                                # Handle nested matched_keywords structure
                                matched_keywords_data = classification.get('matched_keywords', [])
                                reasoning = classification.get('reasoning', '')
                                
                                if broad_cat:
                                    broad_categories.append(broad_cat)
                                if sub_cat:
                                    sub_categories.append(sub_cat)
                                
                                # Handle nested matched_keywords structure
                                if isinstance(matched_keywords_data, dict) and 'matched_keywords' in matched_keywords_data:
                                    # Handle nested structure: {"matched_keywords": ["keyword1", "keyword2"]}
                                    nested_keywords = matched_keywords_data.get('matched_keywords', [])
                                    if nested_keywords:
                                        keywords_clouds.extend(nested_keywords)
                                elif isinstance(matched_keywords_data, list) and matched_keywords_data:
                                    # Handle direct array format
                                    keywords_clouds.extend(matched_keywords_data)
                                
                                if reasoning:
                                    category_reasonings.append(reasoning)
                    
                    # Also include primary classification data if available (but NOT keywords)
                    if primary_classification and not broad_categories:
                        primary_broad = primary_classification.get('broad_category', '')
                        primary_sub = primary_classification.get('sub_category', '')
                        primary_reasoning = primary_classification.get('reasoning', '')
                        
                        if primary_broad:
                            broad_categories.append(primary_broad)
                        if primary_sub:
                            sub_categories.append(primary_sub)
                        if primary_reasoning:
                            category_reasonings.append(primary_reasoning)
                    
                    # Get or create topic and get topic_id
                    topic_id = self.get_or_create_topic(result)
                    
                    # Get common_attachment_id from common_attachments table
                    common_attachment_id = self.get_common_attachment_id(post.id)
                    
                    # Create AnalyzedData record
                    analyzed_data = AnalyzedData(
                        dump_table_id=post.id,
                        input_text=result.get('input_text', ''),
                        processed_text=result.get('processed_text', ''),
                        enhanced_text=result.get('enhanced_text', ''),
                        detected_language=result.get('detected_language', ''),
                        language_confidence=result.get('language_confidence', 0.0),
                        action=result.get('action', ''),
                        topic_title=result.get('topic_title', ''),
                        topic_id=result.get('topic_id', ''),
                        similarity_score=result.get('similarity_score', 0.0),
                        confidence=result.get('confidence', ''),
                        source_type=result.get('source_type', 'social_media'),
                        embedding_model=result.get('embedding_model', ''),
                        processing_time_ms=result.get('processing_time_ms', 0),
                        boost_reasons=json.dumps(result.get('boost_reasons', []), ensure_ascii=False),
                        timestamp=result.get('timestamp', 0.0),
                        
                        # Extracted entities
                        person_names=json.dumps(entities.get('person_names', []), ensure_ascii=False),
                        organisation_names=json.dumps(entities.get('organisation_names', []), ensure_ascii=False),
                        location_names=json.dumps(entities.get('location_names', []), ensure_ascii=False),
                        district_names=json.dumps(entities.get('district_names', []), ensure_ascii=False),
                        thana_names=json.dumps(entities.get('thana_names', []), ensure_ascii=False),
                        incidents=json.dumps(entities.get('incidents', []), ensure_ascii=False),
                        caste_names=json.dumps(entities.get('caste_names', []), ensure_ascii=False),
                        religion_names=json.dumps(entities.get('religion_names', []), ensure_ascii=False),
                        hashtags=json.dumps(entities.get('hashtags', []), ensure_ascii=False),
                        mention_ids_extracted=json.dumps(entities.get('mention_ids', []), ensure_ascii=False),
                        events=json.dumps(entities.get('events', []), ensure_ascii=False),
                        sentiment_label=sentiment_label,
                        sentiment_confidence=sentiment_confidence,
                        contextual_understanding=entities.get('contextual_understanding', ''),
                        
                        # Location fields from incident_location_analysis
                        primary_district=json.dumps(primary_districts, ensure_ascii=False),
                        primary_thana=json.dumps(primary_thanas, ensure_ascii=False),
                        primary_location=json.dumps(primary_locations, ensure_ascii=False),
                        
                        # Category classification fields (ordered for display: category, subcategory, keywords, reasoning)
                        broad_category=json.dumps(broad_categories, ensure_ascii=False),
                        sub_category=json.dumps(sub_categories, ensure_ascii=False),
                        keywords_cloud=json.dumps(keywords_clouds, ensure_ascii=False),
                        category_reasoning=json.dumps(category_reasonings, ensure_ascii=False)
                    )
                    
                    self.session.add(analyzed_data)
                    successfully_processed_post_ids.append(post.id)
                    
                except Exception as e:
                    logger.error(f"Error processing result {i} for post {post.id}: {str(e)}")
                    continue
            
            # Update analysisStatus to 'ANALYZED' for successfully processed posts
            if successfully_processed_post_ids:
                for post_id in successfully_processed_post_ids:
                    self.update_post_status(post_id, 'ANALYZED')
                logger.info(f"Updated analysisStatus to 'ANALYZED' for {len(successfully_processed_post_ids)} posts")
            
            self.session.commit()
            logger.info(f"Successfully saved {len(results)} analyzed records and updated status")
            return True
            
        except Exception as e:
            logger.error(f"Error saving batch analyzed data: {str(e)}")
            self.session.rollback()
            return False

    def save_analyzed_data(self, post_id: int, api_response: Dict[str, Any]) -> bool:
        """Save API response to analyzed_data table"""
        try:
            # Extract entities data
            entities = api_response.get('extracted_entities', {})
            sentiment = entities.get('sentiment', {})
            
            analyzed_data = AnalyzedData(
                dump_table_id=post_id,
                input_text=api_response.get('input_text'),
                processed_text=api_response.get('processed_text'),
                enhanced_text=api_response.get('enhanced_text'),
                detected_language=api_response.get('detected_language'),
                language_confidence=api_response.get('language_confidence'),
                action=api_response.get('action'),
                topic_title=api_response.get('topic_title'),
                topic_id=api_response.get('topic_id'),
                similarity_score=api_response.get('similarity_score'),
                confidence=api_response.get('confidence'),
                source_type=api_response.get('source_type'),
                embedding_model=api_response.get('embedding_model'),
                processing_time_ms=api_response.get('processing_time_ms'),
                boost_reasons=json.dumps(api_response.get('boost_reasons', []), ensure_ascii=False),
                timestamp=api_response.get('timestamp'),
                
                # Extracted entities
                person_names=json.dumps(entities.get('person_names', []), ensure_ascii=False),
                organisation_names=json.dumps(entities.get('organisation_names', []), ensure_ascii=False),
                location_names=json.dumps(entities.get('location_names', []), ensure_ascii=False),
                district_names=json.dumps(entities.get('district_names', []), ensure_ascii=False),
                thana_names=json.dumps(entities.get('thana_names', []), ensure_ascii=False),
                incidents=json.dumps(entities.get('incidents', []), ensure_ascii=False),
                caste_names=json.dumps(entities.get('caste_names', []), ensure_ascii=False),
                religion_names=json.dumps(entities.get('religion_names', []), ensure_ascii=False),
                hashtags=json.dumps(entities.get('hashtags', []), ensure_ascii=False),
                mention_ids_extracted=json.dumps(entities.get('mention_ids', []), ensure_ascii=False),
                events=json.dumps(entities.get('events', []), ensure_ascii=False),
                sentiment_label=sentiment.get('label'),
                sentiment_confidence=sentiment.get('confidence'),
                contextual_understanding=entities.get('contextual_understanding')
            )
            
            self.session.add(analyzed_data)
            self.session.commit()
            logger.info(f"Saved analyzed data for post_id: {post_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving analyzed data for post_id {post_id}: {str(e)}")
            self.session.rollback()
            return False
            
    def update_post_status(self, post_id: int, status: str = "ANALYZED") -> bool:
        """Update the analysisStatus of a post"""
        try:
            post = self.session.query(PostBank).filter(PostBank.id == post_id).first()
            if post:
                post.analysisStatus = status
                self.session.commit()
                logger.info(f"Updated post {post_id} status to {status}")
                return True
            else:
                logger.warning(f"Post with id {post_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error updating post status for id {post_id}: {str(e)}")
            self.session.rollback()
            return False
            
    def process_single_post(self, post: PostBank) -> bool:
        """Process a single post through the pipeline"""
        try:
            logger.info(f"Processing post ID: {post.id}")
            
            # Call API with post_title and post_snippet combined
            combined_text = f"{post.post_title} - {post.post_snippet}"
            api_response = self.call_process_text_api(
                text=combined_text,
                source_type=post.source.lower() if post.source else "social_media"
            )
            
            if not api_response:
                logger.error(f"Failed to get API response for post {post.id}")
                return False
                
            # Save analyzed data
            if not self.save_analyzed_data(post.id, api_response):
                logger.error(f"Failed to save analyzed data for post {post.id}")
                return False
                
            # Update post status
            if not self.update_post_status(post.id, "ANALYZED"):
                logger.error(f"Failed to update status for post {post.id}")
                return False
                
            logger.info(f"Successfully processed post ID: {post.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing post {post.id}: {str(e)}")
            return False
            
    def process_batch(self, posts: List[PostBank]) -> Tuple[int, int]:
        """Process a batch of posts using batch API"""
        successful = 0
        failed = 0
        
        try:
            # Extract texts from posts
            texts = []
            for post in posts:
                # Combine title and snippet for processing
                text_content = f"{post.post_title} -{post.post_snippet}"
                texts.append(text_content)
            
            # Call batch API
            api_response = self.call_process_batch_api(texts, source_type="social_media")
            
            if api_response and api_response.get('successful', 0) > 0:
                # Save batch results
                if self.save_batch_analyzed_data(posts, api_response):
                    successful = api_response.get('successful', 0)
                    failed = api_response.get('failed', 0)
                    logger.info(f"Batch processing completed: {successful} successful, {failed} failed")
                else:
                    logger.error("Failed to save batch results")
                    failed = len(posts)
            else:
                # THIS IS FOR FAILING CASE
                logger.error("Batch API call failed, falling back to individual processing")
                # Fallback to individual processing
                for post in posts:
                    try:
                        if self.process_single_post(post):
                            successful += 1
                        else:
                            failed += 1
                            
                        # Small delay between posts to avoid overwhelming the API
                        time.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(f"Unexpected error processing post {post.id}: {str(e)}")
                        failed += 1
                        
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            # Fallback to individual processing
            for post in posts:
                try:
                    if self.process_single_post(post):
                        successful += 1
                    else:
                        failed += 1
                        
                    # Small delay between posts to avoid overwhelming the API
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Unexpected error processing post {post.id}: {str(e)}")
                    failed += 1
                
        return successful, failed
        
    def run_pipeline(self, batch_size: int = None) -> Dict[str, Any]:
        """Run the main processing pipeline"""
        if batch_size:
            self.batch_size = batch_size
            
        logger.info("Starting pipeline processing...")
        start_time = time.time()
        
        total_processed = 0
        total_successful = 0
        total_failed = 0
        
        try:
            while True:
                # Get unanalyzed posts
                posts = self.get_unanalyzed_posts(limit=self.batch_size)
                
                if not posts:
                    logger.info("No more unanalyzed posts found. Pipeline completed.")
                    break
                    
                logger.info(f"Processing batch of {len(posts)} posts")
                
                # Process batch
                successful, failed = self.process_batch(posts)
                
                total_processed += len(posts)
                total_successful += successful
                total_failed += failed
                
                logger.info(f"Batch completed: {successful} successful, {failed} failed")
                
                # Small delay between batches
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            
        end_time = time.time()
        duration = end_time - start_time
        
        results = {
            "total_processed": total_processed,
            "successful": total_successful,
            "failed": total_failed,
            "duration_seconds": duration,
            "posts_per_second": total_processed / duration if duration > 0 else 0
        }
        
        logger.info(f"Pipeline completed: {results}")
        return results
        
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        try:
            total_posts = self.session.query(PostBank).count()
            analyzed_posts = self.session.query(PostBank).filter(
                PostBank.analysisStatus == 'ANALYZED'
            ).count()
            unanalyzed_posts = self.session.query(PostBank).filter(
                PostBank.analysisStatus == 'NOT_ANALYZED'
            ).count()
            
            return {
                "total_posts": total_posts,
                "analyzed_posts": analyzed_posts,
                "unanalyzed_posts": unanalyzed_posts,
                "completion_percentage": (analyzed_posts / total_posts * 100) if total_posts > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting pipeline stats: {str(e)}")
            return {}


def main():
    """Main function to run the pipeline"""
    logger.info("Starting Main Pipeline Processor")
    
    try:
        with PipelineProcessor() as processor:
            # Create tables
            if not processor.create_tables():
                logger.error("Failed to create tables. Exiting.")
                return
                
            # Get initial stats
            stats = processor.get_pipeline_stats()
            logger.info(f"Pipeline stats: {stats}")
            
            if stats.get('unanalyzed_posts', 0) == 0:
                logger.info("Currently no rows present with status NOT_ANALYZED")
                return
                
            # Run pipeline
            results = processor.run_pipeline(batch_size=10)
            
            # Final stats
            final_stats = processor.get_pipeline_stats()
            logger.info(f"Final pipeline stats: {final_stats}")
            
    except Exception as e:
        logger.error(f"Main pipeline error: {str(e)}")
        

if __name__ == "__main__":
    main()
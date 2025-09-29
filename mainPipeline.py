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
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, BigInteger
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


class AnalyzedData(Base):
    """Model for analyzed_data table to store API response data"""
    __tablename__ = 'analyzed_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    dump_table_id = Column(Integer, nullable=False)  # Reference to post_bank.id
    
    # Text processing fields
    input_text = Column(Text)
    processed_text = Column(Text)
    enhanced_text = Column(Text)
    detected_language = Column(String(10))
    language_confidence = Column(Float)
    action = Column(String(50))
    topic_title = Column(Text)
    topic_id = Column(String(100))
    similarity_score = Column(Float)
    confidence = Column(String(20))
    source_type = Column(String(50))
    embedding_model = Column(String(100))
    processing_time_ms = Column(Integer)
    boost_reasons = Column(Text)  # JSON array as text
    timestamp = Column(Float)
    
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
            from sqlalchemy import text
            with engine.connect() as conn:
                # Check if column exists
                result = conn.execute(
                    text("SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS "
                         "WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = 'post_bank' "
                         "AND COLUMN_NAME = 'analysisStatus'"),
                    {'schema': DUMP_DB_CONFIG['database']}
                )
                
                if result.fetchone()[0] == 0:
                    # Column doesn't exist, add it
                    conn.execute(
                        text("ALTER TABLE post_bank ADD COLUMN analysisStatus VARCHAR(20) DEFAULT 'NOT_ANALYZED'")
                    )
                    conn.commit()
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
                        # Handle both direct arrays and nested structure
                        districts_data = incident_location_analysis.get('primary_district', [])
                        thanas_data = incident_location_analysis.get('primary_thana', [])
                        locations_data = incident_location_analysis.get('primary_location', [])
                        
                        # Ensure we have proper arrays
                        primary_districts = districts_data if isinstance(districts_data, list) else []
                        primary_thanas = thanas_data if isinstance(thanas_data, list) else []
                        primary_locations = locations_data if isinstance(locations_data, list) else []
                    
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
                        keywords_clouds = keywords_data if isinstance(keywords_data, list) else []
                    
                    if 'category_reasoning' in entities:
                        reasoning_data = entities.get('category_reasoning', [])
                        category_reasonings = reasoning_data if isinstance(reasoning_data, list) else []
                    
                    # Fallback: Process category_classifications if direct data not available
                    if not broad_categories and category_classifications:
                        for classification in category_classifications:
                            if isinstance(classification, dict):
                                broad_cat = classification.get('broad_category', '')
                                sub_cat = classification.get('sub_category', '')
                                keywords = classification.get('keywords', [])
                                reasoning = classification.get('reasoning', '')
                                
                                if broad_cat:
                                    broad_categories.append(broad_cat)
                                if sub_cat:
                                    sub_categories.append(sub_cat)
                                if keywords:
                                    keywords_clouds.extend(keywords)
                                if reasoning:
                                    category_reasonings.append(reasoning)
                    
                    # Also include primary classification data if available
                    if primary_classification and not broad_categories:
                        primary_broad = primary_classification.get('broad_category', '')
                        primary_sub = primary_classification.get('sub_category', '')
                        primary_keywords = primary_classification.get('keywords', [])
                        primary_reasoning = primary_classification.get('reasoning', '')
                        
                        if primary_broad:
                            broad_categories.append(primary_broad)
                        if primary_sub:
                            sub_categories.append(primary_sub)
                        if primary_keywords:
                            keywords_clouds.extend(primary_keywords)
                        if primary_reasoning:
                            category_reasonings.append(primary_reasoning)
                    
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
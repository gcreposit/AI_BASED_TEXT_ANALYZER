#!/usr/bin/env python3
"""
UPPCL Main Pipeline for Processing Filtered Awario Data
This module handles the automated batch processing of posts from the filtered_awariodata table,
analyzes them using the batch text processing API, and stores results in uppcl_analyzed_data.
"""

import os
import sys
import time
import json
import logging
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from sqlalchemy import create_engine, Column, String, Text, DateTime, Float, Integer, JSON, Boolean, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase, relationship
from sqlalchemy.sql import func
from urllib.parse import quote_plus
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

# Database Models
class Base(DeclarativeBase):
    pass


class FilteredAwarioData(Base):
    """
    FilteredAwarioData model representing imported social media data from Awario
    """
    __tablename__ = "filtered_awariodata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(100), index=True)
    mentionurl = Column(Text)
    mentiondate = Column(DateTime, index=True)
    authorname = Column(String(255))
    authorusername = Column(String(255))
    title = Column(Text)
    postsnippet = Column(Text, nullable=False)
    reach = Column(Integer)
    sentiment = Column(String(20))
    starred = Column(Boolean, default=False)
    done = Column(Boolean, default=False)
    extra1 = Column(Text)
    extra2 = Column(Text)
    extra3 = Column(Text)
    analysisStatus = Column(String(20), default='NOT_ANALYZED', index=True)
    ts = Column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<FilteredAwarioData(id={self.id}, source='{self.source}', title='{self.title[:50]}...')>"

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'source': self.source,
            'mentionurl': self.mentionurl,
            'mentiondate': self.mentiondate.isoformat() if self.mentiondate else None,
            'authorname': self.authorname,
            'authorusername': self.authorusername,
            'title': self.title,
            'postsnippet': self.postsnippet,
            'reach': self.reach,
            'sentiment': self.sentiment,
            'starred': self.starred,
            'done': self.done,
            'extra1': self.extra1,
            'extra2': self.extra2,
            'extra3': self.extra3,
            'analysisStatus': self.analysisStatus,
            'ts': self.ts.isoformat() if self.ts else None
        }


class UppcLAnalyzedData(Base):
    """
    UppcLAnalyzedData model for storing analyzed results from filtered_awariodata
    """
    __tablename__ = "uppcl_analyzed_data"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    uppcl_table_id = Column(Integer, ForeignKey("filtered_awariodata.id"), nullable=False, index=True)
    
    # Text processing results
    input_text = Column(Text, nullable=False)
    processed_text = Column(Text)
    enhanced_text = Column(Text)
    
    # Language detection
    detected_language = Column(String(20), index=True)
    language_confidence = Column(Float)
    
    # Topic clustering results
    action = Column(String(20))  # grouped, new_topic
    topic_title = Column(String(255))
    topic_id = Column(String(36))
    similarity_score = Column(Float)
    confidence = Column(String(10))  # high, medium, low
    
    # Processing metadata
    source_type = Column(String(50), index=True)
    embedding_model = Column(String(100))
    processing_time_ms = Column(Integer)
    
    # NER extracted entities (JSON fields)
    person_names = Column(JSON)
    organisation_names = Column(JSON)
    location_names = Column(JSON)
    district_names = Column(JSON)
    thana_names = Column(JSON)
    incidents = Column(JSON)
    caste_names = Column(JSON)
    religion_names = Column(JSON)
    hashtags = Column(JSON)
    mention_ids = Column(JSON)
    events = Column(JSON)
    
    # Sentiment analysis
    sentiment_label = Column(String(20))
    sentiment_confidence = Column(Float)
    
    # Additional analysis
    contextual_understanding = Column(Text)
    boost_reasons = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False, index=True)

    # Relationship
    filtered_awario_data = relationship("FilteredAwarioData")

    def __repr__(self):
        return f"<UppcLAnalyzedData(id={self.id}, uppcl_table_id={self.uppcl_table_id}, topic_title='{self.topic_title}')>"

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'uppcl_table_id': self.uppcl_table_id,
            'input_text': self.input_text,
            'processed_text': self.processed_text,
            'enhanced_text': self.enhanced_text,
            'detected_language': self.detected_language,
            'language_confidence': self.language_confidence,
            'action': self.action,
            'topic_title': self.topic_title,
            'topic_id': self.topic_id,
            'similarity_score': self.similarity_score,
            'confidence': self.confidence,
            'source_type': self.source_type,
            'embedding_model': self.embedding_model,
            'processing_time_ms': self.processing_time_ms,
            'person_names': self.person_names,
            'organisation_names': self.organisation_names,
            'location_names': self.location_names,
            'district_names': self.district_names,
            'thana_names': self.thana_names,
            'incidents': self.incidents,
            'caste_names': self.caste_names,
            'religion_names': self.religion_names,
            'hashtags': self.hashtags,
            'mention_ids': self.mention_ids,
            'events': self.events,
            'sentiment_label': self.sentiment_label,
            'sentiment_confidence': self.sentiment_confidence,
            'contextual_understanding': self.contextual_understanding,
            'boost_reasons': self.boost_reasons,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/uppcl_pipeline.log'),
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
PROCESS_BATCH_ENDPOINT = f"{API_BASE_URL}/api/process-batch"

# Create database engine for dump database
encoded_password = quote_plus(DUMP_DB_CONFIG['password'])
DUMP_DATABASE_URL = f"mysql+pymysql://{DUMP_DB_CONFIG['user']}:{encoded_password}@{DUMP_DB_CONFIG['host']}:{DUMP_DB_CONFIG['port']}/{DUMP_DB_CONFIG['database']}"

engine = create_engine(DUMP_DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class UppcLPipelineProcessor:
    """
    UPPCL Pipeline Processor for batch processing filtered Awario data
    """
    
    def __init__(self):
        self.session = None
        logger.info("UPPCL Pipeline Processor initialized")
    
    def __enter__(self):
        """Context manager entry"""
        self.session = SessionLocal()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.session:
            self.session.close()
    
    def create_tables(self) -> bool:
        """Create tables if they don't exist and add missing columns"""
        try:
            # Create tables if they don't exist
            Base.metadata.create_all(bind=engine)
            
            # Check and add analysisStatus column if missing
            self._add_analysis_status_column_if_missing()
            
            logger.info("Tables created/updated successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            return False
    
    def _add_analysis_status_column_if_missing(self):
        """Add analysisStatus column to filtered_awariodata if it doesn't exist
        Also ensure proper indexing for foreign key constraints"""
        try:
            # First, ensure the primary key has proper indexing
            logger.info("Checking and fixing primary key index...")
            
            # Check if primary key index exists
            pk_result = self.session.execute("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.STATISTICS 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND TABLE_NAME = 'filtered_awariodata' 
                AND INDEX_NAME = 'PRIMARY'
            """)
            
            pk_exists = pk_result.scalar() > 0
            
            if not pk_exists:
                logger.info("Adding primary key constraint to id column...")
                # Add primary key constraint if missing
                self.session.execute("""
                    ALTER TABLE filtered_awariodata 
                    ADD PRIMARY KEY (id)
                """)
                logger.info("Primary key constraint added successfully")
            
            # Now check if analysisStatus column exists
            result = self.session.execute("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND TABLE_NAME = 'filtered_awariodata' 
                AND COLUMN_NAME = 'analysisStatus'
            """)
            
            column_exists = result.scalar() > 0
            
            if not column_exists:
                logger.info("Adding missing 'analysisStatus' column to filtered_awariodata table...")
                
                # Add the column
                self.session.execute("""
                    ALTER TABLE filtered_awariodata 
                    ADD COLUMN analysisStatus VARCHAR(20) DEFAULT 'NOT_ANALYZED'
                """)
                
                # Create index
                self.session.execute("""
                    CREATE INDEX idx_filtered_awariodata_analysisStatus 
                    ON filtered_awariodata(analysisStatus)
                """)
                
                # Update existing records
                self.session.execute("""
                    UPDATE filtered_awariodata 
                    SET analysisStatus = 'NOT_ANALYZED' 
                    WHERE analysisStatus IS NULL
                """)
                
                logger.info("✅ analysisStatus column added successfully")
            else:
                logger.info("✅ analysisStatus column already exists")
            
            self.session.commit()
            logger.info("Database schema updates completed successfully")
                
        except Exception as e:
            logger.error(f"Error updating database schema: {str(e)}")
            self.session.rollback()
            raise
    
    def get_unprocessed_posts(self, limit: int = None) -> List[FilteredAwarioData]:
        """
        Get unprocessed posts from filtered_awariodata table based on analysisStatus
        
        Args:
            limit: Maximum number of posts to retrieve
            
        Returns:
            List of FilteredAwarioData objects with analysisStatus = 'NOT_ANALYZED'
        """
        try:
            # Get posts that have analysisStatus = 'NOT_ANALYZED'
            query = self.session.query(FilteredAwarioData).filter(
                FilteredAwarioData.analysisStatus == 'NOT_ANALYZED'
            ).order_by(FilteredAwarioData.id)
            
            if limit:
                query = query.limit(limit)
            
            posts = query.all()
            logger.info(f"Retrieved {len(posts)} unprocessed posts (analysisStatus = 'NOT_ANALYZED')")
            return posts
            
        except Exception as e:
            logger.error(f"Error retrieving unprocessed posts: {str(e)}")
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
    
    def save_batch_analyzed_data(self, posts: List[FilteredAwarioData], api_response: Dict[str, Any]) -> bool:
        """
        Save batch analyzed data to uppcl_analyzed_data table and update analysisStatus
        
        Args:
            posts: List of original FilteredAwarioData objects
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
                    sentiment_data = result.get('extracted_entities', {}).get('sentiment', {})
                    sentiment_label = sentiment_data.get('label', 'neutral')
                    sentiment_confidence = sentiment_data.get('confidence', 0.0)
                    
                    # Extract entities
                    entities = result.get('extracted_entities', {})
                    
                    # Create UppcLAnalyzedData record
                    analyzed_data = UppcLAnalyzedData(
                        uppcl_table_id=post.id,
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
                        person_names=entities.get('person_names', []),
                        organisation_names=entities.get('organisation_names', []),
                        location_names=entities.get('location_names', []),
                        district_names=entities.get('district_names', []),
                        thana_names=entities.get('thana_names', []),
                        incidents=entities.get('incidents', []),
                        caste_names=entities.get('caste_names', []),
                        religion_names=entities.get('religion_names', []),
                        hashtags=entities.get('hashtags', []),
                        mention_ids=entities.get('mention_ids', []),
                        events=entities.get('events', []),
                        sentiment_label=sentiment_label,
                        sentiment_confidence=sentiment_confidence,
                        contextual_understanding=result.get('contextual_understanding', ''),
                        boost_reasons=result.get('boost_reasons', [])
                    )
                    
                    self.session.add(analyzed_data)
                    successfully_processed_post_ids.append(post.id)
                    
                except Exception as e:
                    logger.error(f"Error processing result {i} for post {post.id}: {str(e)}")
                    continue
            
            # Update analysisStatus to 'ANALYZED' for successfully processed posts
            if successfully_processed_post_ids:
                self.session.query(FilteredAwarioData).filter(
                    FilteredAwarioData.id.in_(successfully_processed_post_ids)
                ).update(
                    {FilteredAwarioData.analysisStatus: 'ANALYZED'},
                    synchronize_session=False
                )
                logger.info(f"Updated analysisStatus to 'ANALYZED' for {len(successfully_processed_post_ids)} posts")
            
            self.session.commit()
            logger.info(f"Successfully saved {len(results)} analyzed records and updated status")
            return True
            
        except Exception as e:
            logger.error(f"Error saving batch analyzed data: {str(e)}")
            self.session.rollback()
            return False
    
    def process_batch(self, posts: List[FilteredAwarioData]) -> Tuple[int, int]:
        """
        Process a batch of posts
        
        Args:
            posts: List of FilteredAwarioData objects to process
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        if not posts:
            return 0, 0
        
        try:
            # Prepare texts for batch processing (combine title and postsnippet)
            texts = []
            for post in posts:
                combined_text = f"{post.title} - {post.postsnippet}" if post.title else post.postsnippet
                texts.append(combined_text)
            
            # Call batch API
            api_response = self.call_process_batch_api(texts)
            
            if not api_response:
                logger.error("Batch API call failed")
                return 0, len(posts)
            
            # Save results
            if self.save_batch_analyzed_data(posts, api_response):
                successful = api_response.get('successful', 0)
                failed = api_response.get('failed', 0)
                logger.info(f"Batch processing completed: {successful} successful, {failed} failed")
                return successful, failed
            else:
                logger.error("Failed to save batch analyzed data")
                return 0, len(posts)
                
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return 0, len(posts)
    
    def run_pipeline(self, batch_size: int = 10) -> Dict[str, Any]:
        """
        Run the complete UPPCL pipeline
        
        Args:
            batch_size: Number of posts to process in each batch
            
        Returns:
            Dictionary containing pipeline statistics
        """
        start_time = time.time()
        total_processed = 0
        total_successful = 0
        total_failed = 0
        batches_processed = 0
        
        logger.info(f"Starting UPPCL pipeline with batch size: {batch_size}")
        
        try:
            # Create tables if they don't exist
            if not self.create_tables():
                logger.error("Failed to create tables")
                return {"error": "Failed to create tables"}
            
            while True:
                # Get next batch of unprocessed posts (analysisStatus = 'NOT_ANALYZED')
                posts = self.get_unprocessed_posts(limit=batch_size)
                
                if not posts:
                    logger.info("No more unprocessed posts found")
                    break
                
                logger.info(f"Processing batch {batches_processed + 1} with {len(posts)} posts")
                
                # Process the batch
                successful, failed = self.process_batch(posts)
                
                # Update counters
                total_processed += len(posts)
                total_successful += successful
                total_failed += failed
                batches_processed += 1
                
                logger.info(f"Batch {batches_processed} completed: {successful} successful, {failed} failed")
                
                # Small delay between batches to avoid overwhelming the API
                time.sleep(1)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Prepare final statistics
            stats = {
                "total_processed": total_processed,
                "total_successful": total_successful,
                "total_failed": total_failed,
                "batches_processed": batches_processed,
                "processing_time_seconds": round(processing_time, 2),
                "average_time_per_post": round(processing_time / max(total_processed, 1), 2),
                "success_rate": round((total_successful / max(total_processed, 1)) * 100, 2)
            }
            
            logger.info(f"UPPCL Pipeline completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error in pipeline execution: {str(e)}")
            return {"error": str(e)}
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get current pipeline statistics based on analysisStatus
        
        Returns:
            Dictionary containing current statistics
        """
        try:
            total_posts = self.session.query(FilteredAwarioData).count()
            analyzed_posts = self.session.query(FilteredAwarioData).filter(
                FilteredAwarioData.analysisStatus == 'ANALYZED'
            ).count()
            not_analyzed_posts = self.session.query(FilteredAwarioData).filter(
                FilteredAwarioData.analysisStatus == 'NOT_ANALYZED'
            ).count()
            
            return {
                "total_posts": total_posts,
                "analyzed_posts": analyzed_posts,
                "not_analyzed_posts": not_analyzed_posts,
                "processing_percentage": round((analyzed_posts / max(total_posts, 1)) * 100, 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting pipeline stats: {str(e)}")
            return {"error": str(e)}


def main():
    """Main function to run the UPPCL pipeline"""
    try:
        # Parse command line arguments for batch size
        batch_size = 10  # Default batch size
        if len(sys.argv) > 1:
            try:
                batch_size = int(sys.argv[1])
                if batch_size <= 0:
                    raise ValueError("Batch size must be positive")
            except ValueError as e:
                logger.error(f"Invalid batch size: {e}")
                sys.exit(1)
        
        logger.info(f"Starting UPPCL Main Pipeline with batch size: {batch_size}")
        
        # Run the pipeline
        with UppcLPipelineProcessor() as processor:
            # Ensure tables and columns exist before getting stats
            processor.create_tables()
            
            # Get initial stats
            initial_stats = processor.get_pipeline_stats()
            logger.info(f"Initial stats: {initial_stats}")
            
            # Run the pipeline
            results = processor.run_pipeline(batch_size=batch_size)
            
            # Get final stats
            final_stats = processor.get_pipeline_stats()
            logger.info(f"Final stats: {final_stats}")
            
            # Print summary
            print("\n" + "="*50)
            print("UPPCL PIPELINE EXECUTION SUMMARY")
            print("="*50)
            print(f"Results: {results}")
            print(f"Final Statistics: {final_stats}")
            print("="*50)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Database Migration Script for Topic Table and Analyzed Data Updates
This script handles database schema migrations for the new Topic table and analyzed_data updates.
"""

import os
import sys
import logging
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/migration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Database configuration
DUMP_DB_CONFIG = {
    'host': os.getenv('DUMP_DB_HOST', '94.136.189.147'),
    'database': os.getenv('DUMP_DB_NAME', 'twitter_scrapper'),
    'user': os.getenv('DUMP_DB_USER', 'gccloud'),
    'password': os.getenv('DUMP_DB_PASSWORD', 'Gccloud@1489$'),
    'port': int(os.getenv('DUMP_DB_PORT', '3306'))
}

# Create database engine
encoded_password = quote_plus(DUMP_DB_CONFIG['password'])
DUMP_DATABASE_URL = f"mysql+pymysql://{DUMP_DB_CONFIG['user']}:{encoded_password}@{DUMP_DB_CONFIG['host']}:{DUMP_DB_CONFIG['port']}/{DUMP_DB_CONFIG['database']}"

engine = create_engine(DUMP_DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class TopicAnalyzedDataMigrator:
    """Database migration handler for Topic table and analyzed_data updates"""
    
    def __init__(self):
        self.session = SessionLocal()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.session.rollback()
        else:
            self.session.commit()
        self.session.close()
    
    def check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database"""
        try:
            result = self.session.execute(text("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND TABLE_NAME = :table_name
            """), {"table_name": table_name})
            
            return result.scalar() > 0
        except Exception as e:
            logger.error(f"Error checking table existence: {str(e)}")
            return False
    
    def check_column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table"""
        try:
            result = self.session.execute(text("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND TABLE_NAME = :table_name 
                AND COLUMN_NAME = :column_name
            """), {"table_name": table_name, "column_name": column_name})
            
            return result.scalar() > 0
        except Exception as e:
            logger.error(f"Error checking column existence: {str(e)}")
            return False
    
    def create_topic_table(self) -> bool:
        """Create the topic table if it doesn't exist"""
        try:
            logger.info("Creating topic table...")
            
            if self.check_table_exists('topic'):
                logger.info("✅ Topic table already exists")
                return True
            
            # Create topic table
            self.session.execute(text("""
                CREATE TABLE topic (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    primary_districts LONGTEXT COMMENT 'JSON array stored as [,,,,] format',
                    primary_thana LONGTEXT COMMENT 'JSON array stored as [,,,,] format',
                    primary_location LONGTEXT COMMENT 'JSON array stored as [,,,,] format',
                    broad_category LONGTEXT COMMENT 'JSON array stored as [,,,,] format',
                    sub_category LONGTEXT COMMENT 'JSON array stored as [,,,,] format',
                    keywords_cloud LONGTEXT COMMENT 'JSON array stored as [,,,,] format',
                    category_reasoning LONGTEXT COMMENT 'JSON array stored as [,,,,] format',
                    hashtags LONGTEXT COMMENT 'JSON array stored as [,,,,] format',
                    mention_id_extraction LONGTEXT COMMENT 'JSON array stored as [,,,,] format',
                    read_count INT DEFAULT 0 COMMENT 'Default to 0',
                    unread_count INT DEFAULT 0 COMMENT 'Increments when topic appears in analyzed_data',
                    total_no_of_post INT DEFAULT 0 COMMENT 'Increments when topic appears in analyzed_data',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """))
            
            logger.info("✅ Topic table created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating topic table: {str(e)}")
            return False
    
    def migrate_analyzed_data_table(self) -> bool:
        """Add missing columns to analyzed_data table"""
        try:
            logger.info("Starting migration for analyzed_data table...")
            
            # Check if table exists
            if not self.check_table_exists('analyzed_data'):
                logger.error("analyzed_data table does not exist!")
                return False
            
            # Define new columns to add
            new_columns = {
                'topic_id': 'INT NULL COMMENT "Foreign key to topic table (1:M relationship)"',
                
                # PostBank fields with post_bank_ prefix - Core content fields
                'post_bank_post_title': 'TEXT NULL COMMENT "Post title from PostBank"',
                'post_bank_post_snippet': 'TEXT NULL COMMENT "Post snippet/content from PostBank"',
                'post_bank_post_url': 'TEXT NULL COMMENT "Post URL from PostBank"',
                'post_bank_core_source': 'VARCHAR(255) NULL COMMENT "Core source from PostBank"',
                'post_bank_source': 'VARCHAR(255) NULL COMMENT "Source from PostBank"',
                'post_bank_post_timestamp': 'DATETIME NULL COMMENT "Original timestamp from PostBank"',
                
                # PostBank fields - Author information
                'post_bank_author_name': 'VARCHAR(255) NULL COMMENT "Author name from PostBank"',
                'post_bank_author_username': 'VARCHAR(255) NULL COMMENT "Author username from PostBank"',
                
                # PostBank fields - Language and location
                'post_bank_post_language': 'VARCHAR(50) NULL COMMENT "Post language from PostBank"',
                'post_bank_post_location': 'VARCHAR(255) NULL COMMENT "Post location from PostBank"',
                'post_bank_post_type': 'VARCHAR(100) NULL COMMENT "Post type from PostBank"',
                
                # PostBank fields - Social media metrics
                'post_bank_retweets': 'INT DEFAULT 0 COMMENT "Retweets from PostBank"',
                'post_bank_bookmarks': 'INT DEFAULT 0 COMMENT "Bookmarks from PostBank"',
                'post_bank_comments': 'INT DEFAULT 0 COMMENT "Comments from PostBank"',
                'post_bank_likes': 'INT DEFAULT 0 COMMENT "Likes from PostBank"',
                'post_bank_views': 'BIGINT DEFAULT 0 COMMENT "Views from PostBank"',
                
                # PostBank fields - Metadata
                'post_bank_attachments': 'TEXT NULL COMMENT "Attachments from PostBank"',
                'post_bank_mention_ids': 'TEXT NULL COMMENT "Mention IDs from PostBank"',
                'post_bank_mention_hashtags': 'TEXT NULL COMMENT "Mention hashtags from PostBank"',
                'post_bank_keyword': 'VARCHAR(255) NULL COMMENT "Keyword from PostBank"',
                'post_bank_unique_hash': 'VARCHAR(32) NULL COMMENT "Unique hash from PostBank"',
                'post_bank_video_id': 'VARCHAR(50) NULL COMMENT "Video ID from PostBank"',
                'post_bank_duration': 'VARCHAR(20) NULL COMMENT "Duration from PostBank"',
                'post_bank_category_id': 'VARCHAR(10) NULL COMMENT "Category ID from PostBank"',
                'post_bank_channel_id': 'VARCHAR(50) NULL COMMENT "Channel ID from PostBank"',
                'post_bank_post_id': 'VARCHAR(100) NULL COMMENT "Post ID from PostBank"',
                
                # WhatsApp specific fields
                'post_date': 'VARCHAR(20) NULL COMMENT "Date in dd/mm/yyyy format"',
                'post_time': 'VARCHAR(20) NULL COMMENT "Time in 24-hour format"',
                'mobile_number': 'VARCHAR(20) NULL COMMENT "Mobile number from WhatsApp"',
                'group_id': 'VARCHAR(100) NULL COMMENT "WhatsApp group ID"',
                'reply_to_message_id': 'VARCHAR(100) NULL COMMENT "ID of message being replied to"',
                'reply_text': 'TEXT NULL COMMENT "Text of the message being replied to"',
                'photo_attachment': 'BOOLEAN DEFAULT FALSE COMMENT "Indicates photo attachment"',
                'video_attachment': 'BOOLEAN DEFAULT FALSE COMMENT "Indicates video attachment"',
                'common_attachment_id': 'INT NULL COMMENT "Reference to common_attachments table"',
                
                # Read status field
                'read_status': 'VARCHAR(20) DEFAULT "UNREAD" COMMENT "Read status with default as UNREAD"'
            }
            
            # Add missing columns
            columns_added = 0
            for column_name, column_definition in new_columns.items():
                if not self.check_column_exists('analyzed_data', column_name):
                    logger.info(f"Adding missing column: {column_name}")
                    self.session.execute(text(f"""
                        ALTER TABLE analyzed_data 
                        ADD COLUMN {column_name} {column_definition}
                    """))
                    columns_added += 1
                else:
                    logger.info(f"✅ Column {column_name} already exists")
            
            logger.info(f"✅ analyzed_data table migration completed. {columns_added} columns added.")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating analyzed_data table: {str(e)}")
            return False
    
    def run_migration(self) -> bool:
        """Run the complete migration process"""
        try:
            logger.info("=== Starting Topic and Analyzed Data Migration ===")
            
            # Create topic table
            if not self.create_topic_table():
                logger.error("Failed to create topic table")
                return False
            
            # Migrate analyzed_data table
            if not self.migrate_analyzed_data_table():
                logger.error("Failed to migrate analyzed_data table")
                return False
            
            logger.info("=== Migration completed successfully ===")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            return False


def main():
    """Main migration function"""
    try:
        with TopicAnalyzedDataMigrator() as migrator:
            success = migrator.run_migration()
            if success:
                logger.info("✅ All migrations completed successfully!")
                return 0
            else:
                logger.error("❌ Migration failed!")
                return 1
    except Exception as e:
        logger.error(f"Migration script failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
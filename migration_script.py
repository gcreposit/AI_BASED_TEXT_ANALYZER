#!/usr/bin/env python3
"""
Database Migration Script for UPPCL Pipeline
This script handles database schema migrations and fixes for the UPPCL pipeline.
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


class DatabaseMigrator:
    """Database migration handler for UPPCL pipeline"""
    
    def __init__(self):
        self.session = None
        logger.info("Database Migrator initialized")
    
    def __enter__(self):
        """Context manager entry"""
        self.session = SessionLocal()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.session:
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
    
    def check_primary_key_exists(self, table_name: str) -> bool:
        """Check if primary key exists on a table"""
        try:
            result = self.session.execute(text("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.STATISTICS 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND TABLE_NAME = :table_name 
                AND INDEX_NAME = 'PRIMARY'
            """), {"table_name": table_name})
            
            return result.scalar() > 0
        except Exception as e:
            logger.error(f"Error checking primary key existence: {str(e)}")
            return False
    
    def check_index_exists(self, table_name: str, index_name: str) -> bool:
        """Check if an index exists on a table"""
        try:
            result = self.session.execute(text("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.STATISTICS 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND TABLE_NAME = :table_name 
                AND INDEX_NAME = :index_name
            """), {"table_name": table_name, "index_name": index_name})
            
            return result.scalar() > 0
        except Exception as e:
            logger.error(f"Error checking index existence: {str(e)}")
            return False
    
    def migrate_filtered_awariodata_table(self) -> bool:
        """Migrate filtered_awariodata table with proper schema"""
        try:
            logger.info("Starting migration for filtered_awariodata table...")
            
            # Check if table exists
            if not self.check_table_exists('filtered_awariodata'):
                logger.error("filtered_awariodata table does not exist!")
                return False
            
            # Check and add primary key if missing
            if not self.check_primary_key_exists('filtered_awariodata'):
                logger.info("Adding primary key constraint to id column...")
                self.session.execute(text("""
                    ALTER TABLE filtered_awariodata 
                    ADD PRIMARY KEY (id)
                """))
                logger.info("✅ Primary key constraint added successfully")
            else:
                logger.info("✅ Primary key already exists")
            
            # Check and add analysisStatus column if missing
            if not self.check_column_exists('filtered_awariodata', 'analysisStatus'):
                logger.info("Adding missing 'analysisStatus' column...")
                
                # Add the column
                self.session.execute(text("""
                    ALTER TABLE filtered_awariodata 
                    ADD COLUMN analysisStatus VARCHAR(20) DEFAULT 'NOT_ANALYZED'
                """))
                
                # Create index
                self.session.execute(text("""
                    CREATE INDEX idx_filtered_awariodata_analysisStatus 
                    ON filtered_awariodata(analysisStatus)
                """))
                
                # Update existing records
                self.session.execute(text("""
                    UPDATE filtered_awariodata 
                    SET analysisStatus = 'NOT_ANALYZED' 
                    WHERE analysisStatus IS NULL
                """))
                
                logger.info("✅ analysisStatus column added successfully")
            else:
                logger.info("✅ analysisStatus column already exists")
            
            # Commit all changes
            self.session.commit()
            logger.info("✅ filtered_awariodata table migration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating filtered_awariodata table: {str(e)}")
            self.session.rollback()
            return False
    
    def migrate_uppcl_analyzed_data_table(self) -> bool:
        """Ensure uppcl_analyzed_data table has proper foreign key constraints"""
        try:
            logger.info("Checking uppcl_analyzed_data table constraints...")
            
            # Check if table exists
            if not self.check_table_exists('uppcl_analyzed_data'):
                logger.info("uppcl_analyzed_data table does not exist, will be created by SQLAlchemy")
                return True
            
            # Check foreign key constraint exists
            result = self.session.execute(text("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND TABLE_NAME = 'uppcl_analyzed_data' 
                AND REFERENCED_TABLE_NAME = 'filtered_awariodata'
                AND REFERENCED_COLUMN_NAME = 'id'
            """))
            
            fk_exists = result.scalar() > 0
            
            if not fk_exists:
                logger.info("Adding foreign key constraint...")
                self.session.execute(text("""
                    ALTER TABLE uppcl_analyzed_data 
                    ADD CONSTRAINT fk_uppcl_filtered_awario 
                    FOREIGN KEY (uppcl_table_id) REFERENCES filtered_awariodata(id)
                """))
                logger.info("✅ Foreign key constraint added successfully")
            else:
                logger.info("✅ Foreign key constraint already exists")
            
            self.session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error migrating uppcl_analyzed_data table: {str(e)}")
            self.session.rollback()
            return False
    
    def run_all_migrations(self) -> bool:
        """Run all database migrations"""
        try:
            logger.info("=" * 50)
            logger.info("STARTING DATABASE MIGRATIONS")
            logger.info("=" * 50)
            
            # Migrate filtered_awariodata table
            if not self.migrate_filtered_awariodata_table():
                logger.error("Failed to migrate filtered_awariodata table")
                return False
            
            # Migrate uppcl_analyzed_data table
            if not self.migrate_uppcl_analyzed_data_table():
                logger.error("Failed to migrate uppcl_analyzed_data table")
                return False
            
            logger.info("=" * 50)
            logger.info("ALL MIGRATIONS COMPLETED SUCCESSFULLY")
            logger.info("=" * 50)
            return True
            
        except Exception as e:
            logger.error(f"Error running migrations: {str(e)}")
            return False


def main():
    """Main function to run migrations"""
    try:
        with DatabaseMigrator() as migrator:
            success = migrator.run_all_migrations()
            
            if success:
                logger.info("🎉 Database migration completed successfully!")
                sys.exit(0)
            else:
                logger.error("❌ Database migration failed!")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"Fatal error during migration: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
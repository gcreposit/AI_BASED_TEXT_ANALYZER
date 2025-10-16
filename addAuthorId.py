#!/usr/bin/env python3
"""
Simple script to add post_bank_author_id column to analyzed_data table
"""

import pymysql
from config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def add_post_bank_author_id_column():
    """Add post_bank_author_id column to analyzed_data table"""

    connection = None
    try:
        # Connect to database
        connection = pymysql.connect(
            host=Config.MYSQL_HOST,
            port=Config.MYSQL_PORT,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DATABASE,
            charset='utf8mb4'
        )

        cursor = connection.cursor()

        # Check if column already exists
        check_query = """
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = %s 
        AND TABLE_NAME = 'analyzed_data' 
        AND COLUMN_NAME = 'post_bank_author_id'
        """

        cursor.execute(check_query, (Config.MYSQL_DATABASE,))
        column_exists = cursor.fetchone()[0] > 0

        if column_exists:
            logger.info("Column 'post_bank_author_id' already exists in analyzed_data table")
            return True

        # Add the column
        alter_query = """
        ALTER TABLE analyzed_data 
        ADD COLUMN post_bank_author_id VARCHAR(255) DEFAULT NULL
        """

        logger.info("Adding post_bank_author_id column to analyzed_data table...")
        cursor.execute(alter_query)
        connection.commit()

        logger.info("Successfully added post_bank_author_id column!")
        return True

    except Exception as e:
        logger.error(f"Error adding post_bank_author_id column: {str(e)}")
        if connection:
            connection.rollback()
        return False

    finally:
        if connection:
            connection.close()


if __name__ == "__main__":
    logger.info("Starting post_bank_author_id column addition...")
    success = add_post_bank_author_id_column()

    if success:
        logger.info("Script completed successfully!")
    else:
        logger.error("Script failed!")
        exit(1)
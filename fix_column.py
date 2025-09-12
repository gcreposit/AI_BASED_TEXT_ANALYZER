#!/usr/bin/env python3
"""
Quick fix script to add the analysisStatus column to post_bank table
"""

import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DUMP_DB_CONFIG = {
    'host': os.getenv('DUMP_DB_HOST', '94.136.189.147'),
    'database': os.getenv('DUMP_DB_NAME', 'twitter_scrapper'),
    'user': os.getenv('DUMP_DB_USER', 'gccloud'),
    'password': os.getenv('DUMP_DB_PASSWORD', 'Gccloud@1489$'),
    'port': int(os.getenv('DUMP_DB_PORT', '3306'))
}

def add_analysis_status_column():
    """Add analysisStatus column to post_bank table using direct MySQL connection"""
    connection = None
    try:
        # Connect to MySQL database
        connection = pymysql.connect(
            host=DUMP_DB_CONFIG['host'],
            user=DUMP_DB_CONFIG['user'],
            password=DUMP_DB_CONFIG['password'],
            database=DUMP_DB_CONFIG['database'],
            port=DUMP_DB_CONFIG['port']
        )
        
        with connection.cursor() as cursor:
            # Check if column exists
            cursor.execute(
                "SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS "
                "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'post_bank' "
                "AND COLUMN_NAME = 'analysisStatus'",
                (DUMP_DB_CONFIG['database'],)
            )
            
            result = cursor.fetchone()
            
            if result[0] == 0:
                print("📝 Adding analysisStatus column to post_bank table...")
                
                # Add the column
                cursor.execute(
                    "ALTER TABLE post_bank ADD COLUMN analysisStatus VARCHAR(20) DEFAULT 'NOT_ANALYZED'"
                )
                
                # Commit the changes
                connection.commit()
                print("✅ Successfully added analysisStatus column!")
                
                # Verify the column was added
                cursor.execute(
                    "SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS "
                    "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'post_bank' "
                    "AND COLUMN_NAME = 'analysisStatus'",
                    (DUMP_DB_CONFIG['database'],)
                )
                
                if cursor.fetchone()[0] > 0:
                    print("✅ Column verification successful!")
                else:
                    print("❌ Column verification failed!")
                    
            else:
                print("ℹ️ analysisStatus column already exists in post_bank table")
                
            # Show some sample data
            cursor.execute("SELECT COUNT(*) FROM post_bank")
            total_posts = cursor.fetchone()[0]
            print(f"📊 Total posts in post_bank: {total_posts}")
            
            if total_posts > 0:
                cursor.execute(
                    "SELECT COUNT(*) FROM post_bank WHERE analysisStatus = 'NOT_ANALYZED'"
                )
                unanalyzed = cursor.fetchone()[0]
                print(f"📋 Posts ready for analysis: {unanalyzed}")
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
        
    finally:
        if connection:
            connection.close()
            
    return True

def main():
    print("🔧 Running Column Fix Script")
    print("=" * 40)
    
    success = add_analysis_status_column()
    
    if success:
        print("\n✅ Fix completed successfully!")
        print("\n📝 Next steps:")
        print("   1. Run: python test_pipeline.py")
        print("   2. If tests pass, run: python mainPipeline.py")
    else:
        print("\n❌ Fix failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
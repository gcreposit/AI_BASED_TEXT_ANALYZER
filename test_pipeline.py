#!/usr/bin/env python3
"""
Test script for the Main Pipeline
This script demonstrates how to use the mainPipeline.py functionality
"""

import sys
import os
import logging
from mainPipeline import PipelineProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_pipeline_stats():
    """Test getting pipeline statistics"""
    print("\n=== Testing Pipeline Statistics ===")
    
    try:
        with PipelineProcessor() as processor:
            stats = processor.get_pipeline_stats()
            print(f"Pipeline Statistics: {stats}")
            
            if stats.get('unanalyzed_posts', 0) == 0:
                print("✅ Currently no rows present with status NOT_ANALYZED")
            else:
                print(f"📊 Found {stats.get('unanalyzed_posts', 0)} posts ready for analysis")
                
    except Exception as e:
        print(f"❌ Error getting stats: {str(e)}")


def test_table_creation():
    """Test table creation functionality"""
    print("\n=== Testing Table Creation ===")
    
    try:
        with PipelineProcessor() as processor:
            success = processor.create_tables()
            if success:
                print("✅ Tables created/verified successfully")
            else:
                print("❌ Failed to create tables")
                
    except Exception as e:
        print(f"❌ Error creating tables: {str(e)}")


def test_fetch_posts(limit=5):
    """Test fetching unanalyzed posts"""
    print(f"\n=== Testing Fetch Posts (limit={limit}) ===")
    
    try:
        with PipelineProcessor() as processor:
            posts = processor.get_unanalyzed_posts(limit=limit)
            print(f"📋 Found {len(posts)} unanalyzed posts")
            
            for i, post in enumerate(posts[:3], 1):  # Show first 3
                print(f"\nPost {i}:")
                print(f"  ID: {post.id}")
                print(f"  Title: {post.post_title[:100]}...")
                print(f"  Snippet: {post.post_snippet[:150]}...")
                print(f"  Source: {post.source}")
                print(f"  Status: {post.analysisStatus}")
                
    except Exception as e:
        print(f"❌ Error fetching posts: {str(e)}")


def test_single_post_processing():
    """Test processing a single post (dry run)"""
    print("\n=== Testing Single Post Processing (Dry Run) ===")
    
    try:
        with PipelineProcessor() as processor:
            posts = processor.get_unanalyzed_posts(limit=1)
            
            if not posts:
                print("ℹ️ No unanalyzed posts available for testing")
                return
                
            post = posts[0]
            print(f"📝 Testing with Post ID: {post.id}")
            print(f"   Title: {post.post_title[:100]}...")
            print(f"   Snippet length: {len(post.post_snippet)} characters")
            
            # Test API call (without saving)
            print("\n🔄 Testing API call...")
            api_response = processor.call_process_text_api(
                text=post.post_snippet[:1000],  # Limit text for testing
                source_type=post.source.lower() if post.source else "social_media"
            )
            
            if api_response:
                print("✅ API call successful!")
                print(f"   Detected Language: {api_response.get('detected_language')}")
                print(f"   Topic Title: {api_response.get('topic_title')}")
                print(f"   Confidence: {api_response.get('confidence')}")
                print(f"   Processing Time: {api_response.get('processing_time_ms')}ms")
            else:
                print("❌ API call failed")
                
    except Exception as e:
        print(f"❌ Error in single post processing test: {str(e)}")


def run_pipeline_batch(batch_size=2):
    """Run pipeline for a small batch"""
    print(f"\n=== Running Pipeline Batch (size={batch_size}) ===")
    
    try:
        with PipelineProcessor() as processor:
            # Get initial stats
            initial_stats = processor.get_pipeline_stats()
            print(f"📊 Initial stats: {initial_stats}")
            
            if initial_stats.get('unanalyzed_posts', 0) == 0:
                print("ℹ️ No posts to process")
                return
                
            # Run pipeline with small batch
            results = processor.run_pipeline(batch_size=batch_size)
            print(f"\n✅ Pipeline completed: {results}")
            
            # Get final stats
            final_stats = processor.get_pipeline_stats()
            print(f"📊 Final stats: {final_stats}")
            
    except Exception as e:
        print(f"❌ Error running pipeline batch: {str(e)}")


def main():
    """Main test function"""
    print("🚀 Starting Pipeline Tests")
    print("=" * 50)
    
    # Test 1: Table creation
    test_table_creation()
    
    # Test 2: Pipeline statistics
    test_pipeline_stats()
    
    # Test 3: Fetch posts
    test_fetch_posts(limit=5)
    
    # Test 4: Single post processing (dry run)
    test_single_post_processing()
    
    # Ask user if they want to run actual processing
    print("\n" + "=" * 50)
    response = input("\n🤔 Do you want to run actual pipeline processing (y/N)? ").strip().lower()
    
    if response in ['y', 'yes']:
        run_pipeline_batch(batch_size=2)
    else:
        print("ℹ️ Skipping actual processing. Tests completed.")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()
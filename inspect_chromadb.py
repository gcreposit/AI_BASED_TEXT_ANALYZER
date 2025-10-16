#!/usr/bin/env python3
"""Inspect ChromaDB to see what's actually stored"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.vector_service import VectorService


def inspect_chromadb():
    """Inspect all topics in ChromaDB"""

    print("=" * 80)
    print("🔍 INSPECTING CHROMADB DATABASE")
    print("=" * 80)

    # Initialize vector service
    vector_service = VectorService(
        persist_directory="./chroma_db_2",
        collection_name="topic_vector"
    )

    # Get collection stats
    stats = vector_service.get_collection_stats()
    print(f"\n📊 COLLECTION STATS:")
    print(f"   Total topics: {stats['total_topics']}")
    print(f"   Collection name: {stats['collection_name']}")

    # Get all topics
    try:
        all_data = vector_service.collection.get(
            include=['metadatas', 'documents', 'embeddings']
        )

        print(f"\n📝 FOUND {len(all_data['ids'])} TOPICS\n")

        for i, topic_id in enumerate(all_data['ids']):
            metadata = all_data['metadatas'][i]
            document = all_data['documents'][i] if all_data['documents'] else ""

            print("=" * 80)
            print(f"TOPIC #{i + 1}")
            print("=" * 80)
            print(f"🆔 ID: {topic_id}")
            print(f"📌 Title: {metadata.get('topic_title', 'N/A')}")
            print(f"🌍 Language: {metadata.get('primary_language', 'N/A')}")
            print(f"📍 Status: {metadata.get('topic_status', 'N/A')}")
            print(f"📊 Content Count: {metadata.get('content_count', 0)}")

            # ✅ CRITICAL: Check district fields
            print(f"\n📍 LOCATION DATA:")
            print(f"   primary_district: '{metadata.get('primary_district', '')}'")
            print(f"   district_1: '{metadata.get('district_1', '')}'")
            print(f"   district_2: '{metadata.get('district_2', '')}'")
            print(f"   district_3: '{metadata.get('district_3', '')}'")
            print(f"   district_names_json: {metadata.get('district_names_json', 'N/A')}")

            # ✅ CRITICAL: Check incident data
            print(f"\n🚨 INCIDENT DATA:")
            incidents = metadata.get('incidents', 'N/A')
            print(f"   incidents (raw): {incidents}")
            print(f"   incidents (type): {type(incidents)}")

            # Try to parse if it's JSON string
            if isinstance(incidents, str) and incidents != 'N/A':
                try:
                    parsed_incidents = json.loads(incidents)
                    print(f"   incidents (parsed): {parsed_incidents}")
                except:
                    print(f"   incidents (failed to parse)")

            # Check primary classification
            primary_class = metadata.get('primary_classification', 'N/A')
            print(f"\n📋 PRIMARY CLASSIFICATION:")
            print(f"   raw: {primary_class}")
            print(f"   type: {type(primary_class)}")

            if isinstance(primary_class, str) and primary_class != 'N/A':
                try:
                    parsed_class = json.loads(primary_class)
                    print(f"   broad_category: {parsed_class.get('broad_category', 'N/A')}")
                    print(f"   sub_category: {parsed_class.get('sub_category', 'N/A')}")
                    print(f"   confidence: {parsed_class.get('confidence', 0)}")
                except:
                    print(f"   (failed to parse)")

            # Check category classifications
            cat_class = metadata.get('category_classifications', 'N/A')
            print(f"\n🏷️  CATEGORY CLASSIFICATIONS:")
            print(f"   raw: {cat_class}")
            print(f"   type: {type(cat_class)}")

            # Check temporal info
            temporal = metadata.get('temporal_info', 'N/A')
            print(f"\n⏰ TEMPORAL INFO:")
            print(f"   raw: {temporal}")

            # Show document preview
            print(f"\n📄 DOCUMENT (first 200 chars):")
            print(f"   {document[:200]}...")

            print("\n" + "=" * 80 + "\n")

        # Summary
        print("\n" + "=" * 80)
        print("📊 SUMMARY")
        print("=" * 80)

        # Check if any topics have district_1 field
        topics_with_d1 = sum(1 for m in all_data['metadatas'] if m.get('district_1'))
        print(f"✅ Topics with district_1 field: {topics_with_d1}/{len(all_data['ids'])}")

        # Check if any topics have primary_district field
        topics_with_pd = sum(1 for m in all_data['metadatas'] if m.get('primary_district'))
        print(f"✅ Topics with primary_district field: {topics_with_pd}/{len(all_data['ids'])}")

        # Check if any topics have incidents
        topics_with_incidents = sum(1 for m in all_data['metadatas'] if m.get('incidents'))
        print(f"✅ Topics with incidents field: {topics_with_incidents}/{len(all_data['ids'])}")

        # Check if any topics have primary_classification
        topics_with_class = sum(1 for m in all_data['metadatas'] if m.get('primary_classification'))
        print(f"✅ Topics with primary_classification: {topics_with_class}/{len(all_data['ids'])}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    inspect_chromadb()
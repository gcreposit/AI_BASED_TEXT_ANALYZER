"""
Pure keyword-based category classification
Uses external DB/JSON for category keywords
"""

import logging
import json
import os
import re
from typing import Dict, Any, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class KeywordCategoryClassifier:
    """
    Pure keyword-based classification without LLM
    Loads keywords from external source (DB/JSON)
    """

    def __init__(self, keywords_source: str = "config/categories_keywords.json"):
        """
        Initialize classifier with keyword mappings

        Args:
            keywords_source: Path to JSON file with category keywords
        """
        self.keywords_source = keywords_source
        self.category_keywords = {}
        self._load_keywords()

    def _load_keywords(self):
        """Load category keywords from external source"""
        try:
            with open(self.keywords_source, 'r', encoding='utf-8') as f:
                self.category_keywords = json.load(f)

            logger.info(f"✅ Loaded keywords for {len(self.category_keywords)} broad categories")

            # Log summary
            for broad_cat, subcats in self.category_keywords.items():
                total_keywords = sum(len(keywords) for keywords in subcats.values())
                logger.info(f"  - {broad_cat}: {len(subcats)} subcategories, {total_keywords} keywords")

        except FileNotFoundError:
            logger.error(f"❌ Keywords file not found: {self.keywords_source}")
            self.category_keywords = {}
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in keywords file: {e}")
            self.category_keywords = {}
        except Exception as e:
            logger.error(f"❌ Failed to load keywords: {e}")
            self.category_keywords = {}

    def reload_keywords(self):
        """Reload keywords from source (call when DB is updated)"""
        logger.info("🔄 Reloading category keywords...")
        self._load_keywords()

    def classify(self, text: str, ner_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Classify text based on keyword matching

        Args:
            text: Original text to classify
            ner_data: Optional NER data for enhanced classification

        Returns:
            Classification result with all matching categories
        """
        text_lower = text.lower()

        # Extract additional context from NER if available
        incidents_text = ""
        if ner_data:
            incidents = ner_data.get('incidents', [])
            events = ner_data.get('events', [])
            incidents_text = " ".join(incidents + events).lower()

        # Combined text for matching
        combined_text = f"{text_lower} {incidents_text}"

        # Store all matches with scores
        all_matches = []

        # Scan all categories
        for broad_category, subcategories in self.category_keywords.items():
            for sub_category, keywords in subcategories.items():
                matched_keywords = []
                match_score = 0

                # Check each keyword
                for keyword in keywords:
                    keyword_lower = keyword.lower()

                    # Direct match (case-insensitive)
                    if keyword_lower in combined_text:
                        matched_keywords.append(keyword)
                        match_score += 2

                    # Word boundary match (more precise)
                    elif re.search(r'\b' + re.escape(keyword_lower) + r'\b', combined_text):
                        matched_keywords.append(keyword)
                        match_score += 3  # Higher score for exact word match

                # If we found matches, add to results
                if match_score > 0:
                    # Calculate confidence based on matches
                    confidence = min(0.95, match_score / (len(keywords) + 5))

                    all_matches.append({
                        "broad_category": broad_category,
                        "sub_category": sub_category,
                        "confidence": confidence,
                        "matched_keywords": matched_keywords[:5],  # Limit to 5
                        "match_score": match_score,
                        "reasoning": f"Matched {len(matched_keywords)} keywords: {', '.join(matched_keywords[:3])}"
                    })

        # Sort by match score
        all_matches.sort(key=lambda x: x["match_score"], reverse=True)

        # Remove match_score from final output (internal use only)
        for match in all_matches:
            del match["match_score"]

        # Prepare result
        if all_matches:
            return {
                "category_classifications": all_matches[:5],  # Top 5 matches
                "primary_classification": {
                    "broad_category": all_matches[0]["broad_category"],
                    "sub_category": all_matches[0]["sub_category"],
                    "confidence": all_matches[0]["confidence"]
                },
                "classification_method": "keyword_matching"
            }
        else:
            return {
                "category_classifications": [{
                    "broad_category": "UNCATEGORIZED",
                    "sub_category": "GENERAL",
                    "confidence": 0.0,
                    "matched_keywords": [],
                    "reasoning": "No keyword matches found"
                }],
                "primary_classification": {
                    "broad_category": "UNCATEGORIZED",
                    "sub_category": "GENERAL",
                    "confidence": 0.0
                },
                "classification_method": "keyword_matching"
            }

    def get_category_info(self) -> Dict[str, Any]:
        """Get information about loaded categories"""
        total_broad = len(self.category_keywords)
        total_sub = sum(len(subs) for subs in self.category_keywords.values())
        total_keywords = sum(
            len(keywords)
            for subcats in self.category_keywords.values()
            for keywords in subcats.values()
        )

        return {
            "total_broad_categories": total_broad,
            "total_subcategories": total_sub,
            "total_keywords": total_keywords,
            "source": self.keywords_source,
            "categories": list(self.category_keywords.keys())
        }

    # ADD THIS METHOD TO KeywordCategoryClassifier class

    def fetch_and_update_from_db(self, db_config: Dict[str, str] = None) -> bool:
        """
        Fetch categories from external database and update local JSON

        Args:
            db_config: Database configuration dict (host, database, user, password, port)

        Returns:
            Success status
        """
        try:
            import pymysql
            import json
            from datetime import datetime

            # Use provided config or environment variables
            if not db_config:
                db_config = {
                    'host': os.getenv('DUMP_DB_HOST', '127.0.0.1'),
                    'database': os.getenv('DUMP_DB_NAME', 'up_police_matrix'),
                    'user': os.getenv('DUMP_DB_USER', 'root'),
                    'password': os.getenv('DUMP_DB_PASSWORD', ''),
                    'port': int(os.getenv('DUMP_DB_PORT', '3306'))
                }

            logger.info(f"Connecting to category database at {db_config['host']}")

            # Connect to database
            connection = pymysql.connect(
                host=db_config['host'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database'],
                port=db_config['port'],
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )

            with connection:
                with connection.cursor() as cursor:
                    # MODIFY THIS QUERY BASED ON YOUR ACTUAL DB SCHEMA
                    query = """
                    SELECT 
                        broad_category,
                        sub_category,
                        keywords
                    FROM category_keywords
                    WHERE is_active = 1
                    ORDER BY broad_category, sub_category
                    """

                    cursor.execute(query)
                    results = cursor.fetchall()

            if not results:
                logger.warning("No categories fetched from database")
                return False

            # Build category structure
            updated_categories = {}

            for row in results:
                broad_cat = row['broad_category'].strip().upper()
                sub_cat = row['sub_category'].strip().upper()
                keywords_str = row['keywords']

                # Parse keywords (assuming comma-separated or JSON)
                if keywords_str.startswith('['):
                    # JSON array
                    keywords = json.loads(keywords_str)
                else:
                    # Comma-separated
                    keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]

                # Add to structure
                if broad_cat not in updated_categories:
                    updated_categories[broad_cat] = {}

                updated_categories[broad_cat][sub_cat] = keywords

            # Backup old file
            backup_path = f"{self.keywords_source}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if os.path.exists(self.keywords_source):
                import shutil
                shutil.copy(self.keywords_source, backup_path)
                logger.info(f"Created backup: {backup_path}")

            # Write updated categories
            with open(self.keywords_source, 'w', encoding='utf-8') as f:
                json.dump(updated_categories, f, ensure_ascii=False, indent=2)

            # Reload into memory
            self.category_keywords = updated_categories

            logger.info(f"✅ Successfully updated categories from DB")
            logger.info(f"   - {len(updated_categories)} broad categories")
            logger.info(f"   - {sum(len(subs) for subs in updated_categories.values())} subcategories")
            logger.info(
                f"   - {sum(len(kws) for subs in updated_categories.values() for kws in subs.values())} total keywords")

            return True

        except Exception as e:
            logger.error(f"Failed to fetch categories from DB: {e}")
            return False
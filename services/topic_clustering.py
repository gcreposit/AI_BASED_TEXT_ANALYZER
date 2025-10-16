"""
Enhanced topic clustering service with intelligent topic title generation
"""
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
from sqlalchemy.sql import func
import re
from datetime import datetime, timedelta  # ✅ ADD THIS LINE
from services.keyword_classifier import KeywordCategoryClassifier
from services.text_cleaner import AITextCleaner

from district_normalizer import DistrictNormalizer

logger = logging.getLogger(__name__)

def generate_unassigned_topic_id():
    return str(uuid.uuid4())

# Example usage
UNASSIGNED_TOPIC_ID = generate_unassigned_topic_id()
UNASSIGNED_TOPIC_TITLE_HINDI = "असाइन नहीं की गई पोस्ट"

# Temporal window for topic clustering (days)
TEMPORAL_WINDOW_DAYS = 30

# Strict category validation
ALLOWED_CATEGORIES = {
    "CRIME": {
        "MURDER", "AGAINST WOMEN", "COMMUNAL", "CASTEISM",
        "AGAINST MINORS", "LOVE JIHAAD", "AGAINST COW",
        "ROBBERY", "LOOT", "THEFT", "KIDNAPPING", "ASSAULT",
        "AGAINST ANIMAL", "PETA"
    },
    "TRAFFIC RELATED": {
        "TRAFFIC JAM", "DIVERSION", "TRAFFIC RULES VIOLATION"
    },
    "RAILWAY RELATED": {
        "INVOLVING RAILWAYS"
    },
    "POLICE MISCONDUCT": {
        "CORRUPTION", "MISCONDUCT"
    },
    "GRIEVANCE": {
        "EMERGENCY", "FIR RELATED", "COMPLAINTS",
        "OFFICIAL SERVICE RELATED", "FIRE RELATED", "ACCIDENT"
    },
    "CYBER CRIME": {
        "HACKING", "PHISING", "DIGITAL ARREST", "MONEY FRAUD",
        "EXPLICIT CONTENT", "DIGITAL RANSOMI"
    },
    "HATE SPEECH": {
        "AGAINST RELIGION", "AGAINST CASTEISM", "POLITICALLY MOTIVATED"
    },
    "VIRAL & FACT CHECK": {
        "VIRAL", "FAKE NEWS", "RUMOURS"
    },
    "ELECTION": {
        "BOOTH CAPTURING", "MCC VIOLATIONS", "FAKE VOTING",
        "BOOTH FACILITIES RELATED ISSUE", "BOOTH MACHINE RELATED ISSUES",
        "COMPLAIN AGAINST BOOTH OFFICIALS", "HINDRANCE IN ELECTION SERVICES",
        "ELECTORAL DISPUTES"
    },
    "LAW & ORDER": {
        "PROTEST", "MOVEMENTS", "CROWD SUMMON",
        "ANTI NATIONAL ACTIVITIES", "TERRORIST RELATED", "DISASTER RELATED"
    },
    "ANTI NARCOTICS": {
        "ANTI NARCOTICS"
    },
    "FESTIVALS": {
        "HINDU RELATED", "MUSLIM RELATED", "GOVERNMENT INITIATIVES"
    }
}


class TopicClusteringService:
    """
    Enhanced topic clustering service with intelligent topic title generation
    that creates meaningful, descriptive titles based on incident, location, and context
    """

    def __init__(self,
                 embedding_service,
                 ner_extractor,
                 vector_service,
                 db_manager):
        self.embedding_service = embedding_service
        self.ner_extractor = ner_extractor
        self.vector_service = vector_service
        self.db_manager = db_manager

        # ✅ Initialize AI-powered text cleaner
        self.text_cleaner = AITextCleaner(ner_extractor)

        # ✅ NEW: Initialize keyword-based classifier
        self.keyword_classifier = KeywordCategoryClassifier()

        # Enhanced incident categorization patterns
        self.incident_patterns = {
            'legal_action': [
                r'सजा', r'दंडित', r'कारावास', r'जेल', r'अदालत', r'न्यायालय', r'फैसला',
                r'sentence', r'prison', r'court', r'judgment', r'conviction', r'BNS', r'IPC',
                r'धारा', r'मु0अ0सं0', r'FIR', r'पोक्सो', r'POCSO', r'अर्थदंड'
            ],
            'crime': [
                r'चोरी', r'डकैती', r'लूट', r'हत्या', r'दुष्कर्म', r'बलात्कार', r'अपहरण',
                r'theft', r'robbery', r'murder', r'rape', r'kidnapping', r'assault',
                r'मारपीट', r'छेड़छाड़', r'घूसखोरी', r'भ्रष्टाचार'
            ],
            'police_action': [
                r'गिरफ्तार', r'पकड़', r'छापेमारी', r'ऑपरेशन', r'पुलिस कार्रवाई',
                r'arrest', r'caught', r'raid', r'operation', r'police action',
                r'जांच', r'तलाशी', r'घेराबंदी', r'कार्रवाई'
            ],
            'accident': [
                r'दुर्घटना', r'एक्सीडेंट', r'टक्कर', r'आग', r'विस्फोट',
                r'accident', r'collision', r'fire', r'explosion', r'crash',
                r'गिरना', r'डूबना', r'जलना', r'घायल'
            ],
            'protest_political': [
                r'प्रदर्शन', r'धरना', r'रैली', r'आंदोलन', r'विरोध', r'चुनाव',
                r'protest', r'demonstration', r'rally', r'movement', r'election',
                r'नारेबाजी', r'जुलूस', r'सभा', r'मीटिंग'
            ],
            'administrative': [
                r'योजना', r'सरकारी', r'प्रशासनिक', r'विभाग', r'कार्यक्रम', r'नीति',
                r'scheme', r'government', r'administrative', r'department', r'program', r'policy',
                r'मंत्रालय', r'अधिकारी', r'सचिव', r'मुख्यमंत्री'
            ],
            'social_incident': [
                r'समाजिक', r'सामुदायिक', r'धार्मिक', r'जातिवाद', r'भेदभाव',
                r'social', r'community', r'religious', r'discrimination', r'caste',
                r'दंगा', r'संघर्ष', r'तनाव', r'विवाद'
            ]
        }

        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'total_processing_time': 0,
            'topics_created': 0,
            'topics_merged': 0,
            'errors': 0,
            'title_improvements': 0
        }

    @staticmethod
    def filter_low_confidence_data(data: Dict[str, Any], min_confidence: float = 0.6) -> Dict[str, Any]:
        """Remove data with confidence below threshold"""

        # ✅ Filter advanced_sentiment
        if 'advanced_sentiment' in data:
            sentiment = data['advanced_sentiment']

            for stance_type in ['pro_towards', 'against_towards', 'neutral_towards']:
                if stance_type in sentiment:
                    for category in sentiment[stance_type]:
                        if isinstance(sentiment[stance_type][category], list):
                            # Filter list items
                            sentiment[stance_type][category] = [
                                item for item in sentiment[stance_type][category]
                                if item.get('confidence', 0) >= min_confidence
                            ]

        # ✅ Filter category_classifications
        if 'category_classifications' in data:
            data['category_classifications'] = [
                cat for cat in data['category_classifications']
                if cat.get('confidence', 0) >= min_confidence
            ]

        return data

    # MODIFY at the beginning of process_text_complete

    def process_text_complete(self,
                              text: str,
                              source_type: str = "unknown",
                              user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        COMPLETE process with text cleaning and all enhancements
        """
        start_time = time.time()

        try:
            # ========== STEP 0: TEXT CLEANING (NEW) ==========

            cleaning_result = self.text_cleaner.clean_text(
                text,
                source_type=source_type,
                use_ai=True
            )

            cleaned_text = cleaning_result["cleaned_text"]

            if not cleaned_text or len(cleaned_text) < 10:
                raise ValueError("Text too short after cleaning")

            logger.info(
                f"Text cleaning: {cleaning_result['original_length']} → {cleaning_result['cleaned_length']} chars")

            # Use cleaned_text for all further processing
            text = cleaned_text

            # ========== STEP 1: Validation ==========
            if not text or len(text.strip()) < 3:
                raise ValueError("Invalid text")

            logger.info(f"STEP 1 COMPLETED - Processing text: {len(text)} chars")

            # ========== STEP 2: Language & Preprocessing ==========
            detected_language, lang_confidence = self.embedding_service.detect_language(text)
            processed_text = self.embedding_service.preprocess_text(text)

            logger.info(f"STEP 2 COMPLETED - Language: {detected_language} ({lang_confidence:.2f})")

            # ========== STEP 3: NER Extraction (includes temporal & sentiment) ==========
            ner_data = self.ner_extractor.extract(text)

            logger.info(f"STEP 3 COMPLETED - NER complete - Districts: {ner_data.get('district_names', [])}")

            # ========== STEP 4: District Normalization ==========
            ner_data['district_names'] = DistrictNormalizer.normalize_list(
                ner_data.get('district_names', [])
            )

            # Normalize in location analysis too
            if 'incident_location_analysis' in ner_data:
                loc_analysis = ner_data['incident_location_analysis']
                loc_analysis['incident_districts'] = DistrictNormalizer.normalize_list(
                    loc_analysis.get('incident_districts', [])
                )
                loc_analysis['related_districts'] = DistrictNormalizer.normalize_list(
                    loc_analysis.get('related_districts', [])
                )

            logger.info(f"Normalized districts: {ner_data['district_names']}")

            # ========== STEP 5: Temporal Info Check ==========
            if 'temporal_info' not in ner_data or not ner_data['temporal_info'].get('incident_date'):
                ner_data['temporal_info'] = self.ner_extractor._extract_temporal_info(text)

            logger.info(
                f"Temporal: {ner_data['temporal_info'].get('temporal_phrase')} ({ner_data['temporal_info'].get('days_ago')} days ago)")

            # ========== STEP 6: KEYWORD-BASED CATEGORY CLASSIFICATION ==========
            classification_result = self.keyword_classifier.classify(text, ner_data)

            ner_data['category_classifications'] = classification_result['category_classifications']
            ner_data['primary_classification'] = classification_result['primary_classification']

            logger.info(
                f"Category: {ner_data['primary_classification'].get('broad_category')} > "
                f"{ner_data['primary_classification'].get('sub_category')} "
                f"(confidence: {ner_data['primary_classification'].get('confidence'):.2f})"
            )

            # ========== STEP 7: Enhanced Text Creation ==========
            enhanced_text = self.embedding_service.create_enhanced_text(processed_text, ner_data)

            # ========== STEP 8: Generate Embedding ==========
            embeddings = self.embedding_service.generate_embeddings([enhanced_text])
            query_embedding = embeddings[0]

            # ========== STEP 9: THREE-PHASE MATCHING ==========
            similar_topics = self._find_similar_topics_three_phase(
                query_embedding, ner_data, detected_language, source_type
            )

            logger.info(f"Found {len(similar_topics)} similar topics")

            # ========== STEP 10: Topic Assignment Decision ==========
            has_location = bool(ner_data.get('district_names'))
            has_incident = bool(ner_data.get('incidents') or ner_data.get('events'))

            information_score = (
                    (1.0 if has_location else 0.0) +
                    (1.0 if has_incident else 0.0) +
                    (0.5 if len(ner_data.get('contextual_understanding', '')) > 50 else 0.0)
            )

            if information_score < 1.5:
                # Assign to Unassigned Posts
                topic_result = self.assign_to_unassigned_topic(
                    text, ner_data,
                    reason=f"Insufficient information (score: {information_score:.1f})",
                    user_id=user_id
                )

            elif similar_topics:
                # Group with existing topic
                best_match = similar_topics[0]
                topic_id = best_match['topic_id']

                self._update_existing_topic(topic_id, text, ner_data, user_id)

                topic_result = {
                    "action": "grouped",
                    "topic_id": topic_id,
                    "topic_title": best_match['metadata'].get('topic_title'),
                    "similarity_score": best_match['enhanced_similarity'],
                    "confidence": self._calculate_confidence(best_match['enhanced_similarity']),
                    "boost_reasons": best_match.get('boost_reasons', []),
                    "temporal_distance_days": best_match.get('temporal_distance_days'),
                    "incident_match_score": best_match.get('incident_match_score')
                }

            else:
                # Create new topic
                topic_id = str(uuid.uuid4())

                # Generate Hindi title with LLM
                topic_title = self._generate_topic_title_with_llm_hindi(ner_data, text)

                # Create topic
                self._create_new_topic_with_temporal(
                    topic_id, topic_title, text, processed_text, enhanced_text,
                    query_embedding, ner_data, detected_language, lang_confidence,
                    source_type, user_id
                )

                topic_result = {
                    "action": "new_topic_created",
                    "topic_id": topic_id,
                    "topic_title": topic_title,
                    "similarity_score": 0.0,
                    "confidence": "high",
                    "boost_reasons": []
                }

            # ========== STEP 11: Filter Low Confidence Data ==========
            ner_data = self.filter_low_confidence_data(ner_data, min_confidence=0.6)

            # ========== STEP 12: Build Result (SIMPLIFIED OUTPUT) ==========
            processing_time = (time.time() - start_time) * 1000

            # ✅ Clean up NER data to avoid duplication
            cleaned_entities = {k: v for k, v in ner_data.items() if k not in [
                'temporal_info', 'advanced_sentiment', 'category_classifications',
                'primary_classification', 'incident_location_analysis'
            ]}

            result = {
                "input_text": text,  # ✅ SINGLE CLEANED TEXT (no processed/enhanced variants)
                "detected_language": detected_language,
                "language_confidence": lang_confidence,
                "action": topic_result["action"],
                "topic_title": topic_result.get("topic_title", ""),
                "topic_id": topic_result["topic_id"],
                "similarity_score": topic_result["similarity_score"],
                "confidence": topic_result["confidence"],
                "source_type": source_type,
                "embedding_model": "BAAI/bge-m3",
                "processing_time_ms": int(processing_time),

                # ✅ Root level fields (clean data)
                "temporal_info": ner_data.get('temporal_info', {}),
                "advanced_sentiment": ner_data.get('advanced_sentiment', {}),
                "category_classifications": ner_data.get('category_classifications', []),
                "primary_classification": ner_data.get('primary_classification', {}),
                "incident_location_analysis": ner_data.get('incident_location_analysis', {}),

                # ✅ Cleaned entities (no duplicates)
                "extracted_entities": cleaned_entities,

                "boost_reasons": topic_result.get("boost_reasons", []),
                "temporal_distance_days": topic_result.get("temporal_distance_days"),
                "incident_match_score": topic_result.get("incident_match_score"),
                "can_reassign": topic_result.get("user_can_reassign", False),

                # ✅ Cleaning metadata (optional, only if cleaning was applied)
                "text_cleaning": {
                    "applied": cleaning_result["cleaning_applied"],
                    "reduction_percentage": cleaning_result["reduction_percentage"],
                    "removed_noise_count": len(cleaning_result["removed_noise"])
                } if cleaning_result["cleaning_applied"] else None,

                "timestamp": time.time()
            }

            logger.info(f"✅ Processing complete: {topic_result['action']} in {processing_time:.2f}ms")

            # ========== STEP 13: Logging ==========
            self._log_processing(result, processing_time, user_id)
            self.stats['total_processed'] += 1

            return result

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Processing failed: {e}", exc_info=True)
            self.stats['errors'] += 1

            return {
                "input_text": text,
                "error": str(e),
                "action": "error",
                "processing_time_ms": int(processing_time),
                "timestamp": time.time()
            }

    def _find_similar_topics_three_phase(self,
                                         query_embedding: np.ndarray,
                                         ner_data: Dict[str, Any],
                                         language: str,
                                         source_type: str) -> List[Dict[str, Any]]:

        """THREE-PHASE MATCHING with CORRECT ChromaDB filter"""

        # ============ PHASE 1: WHERE (Location) ============
        incident_districts = ner_data.get('incident_location_analysis', {}).get('incident_districts', [])

        if not incident_districts:
            incident_districts = ner_data.get('district_names', [])

        if not incident_districts:
            logger.warning("❌ PHASE 1 FAILED: No districts identified")
            return []

        logger.info(f"✅ PHASE 1 (WHERE): Filtering by districts: {incident_districts}")

        # ============ PHASE 2: WHAT (Incident Type) ============
        current_incident_types = self._extract_incident_signatures(ner_data)

        if not current_incident_types:
            logger.warning("❌ PHASE 2 FAILED: No incident type identified")
            return []

        logger.info(f"✅ PHASE 2 (WHAT): Incident signatures: {current_incident_types}")

        # ============ PHASE 3: WHEN (Temporal) ============
        current_temporal = ner_data.get('temporal_info', {})
        current_days_ago = current_temporal.get('days_ago')

        # ✅ Handle None for days_ago
        if current_days_ago is None:
            current_days_ago = 999  # Large number if no temporal info
            logger.info(f"✅ PHASE 3 (WHEN): No temporal information provided")
        else:
            logger.info(f"✅ PHASE 3 (WHEN): Incident occurred ~{current_days_ago} days ago")

        # ============ BUILD FILTERS (FIXED FOR CHROMADB) ============
        # ✅ Build conditions using individual district fields (NOT $contains)
        district_conditions = []

        # Check up to 3 district fields
        for i, district in enumerate(incident_districts[:3]):
            field_num = i + 1
            district_conditions.append({
                f"district_{field_num}": {"$eq": district}
            })

        # Also check primary_district
        if incident_districts:
            district_conditions.append({
                "primary_district": {"$eq": incident_districts[0]}
            })

        # ✅ Build filter based on number of conditions
        if len(district_conditions) == 1:
            # Single condition
            location_filter = {
                "$and": [
                    district_conditions[0],
                    {"topic_status": {"$eq": "active"}}
                ]
            }
        elif len(district_conditions) >= 2:
            # Multiple conditions
            location_filter = {
                "$and": [
                    {"$or": district_conditions},
                    {"topic_status": {"$eq": "active"}}
                ]
            }
        else:
            logger.error("No district conditions available")
            return []

        logger.info(f"🔍 Filter structure: {location_filter}")
        logger.info(f"   Districts: {len(incident_districts)}, Conditions: {len(district_conditions)}")

        # ============ SEARCH WITH FILTERS ============
        try:
            candidates = self.vector_service.search_similar_topics(
                query_embedding=query_embedding,
                n_results=100,
                threshold=0.3,
                filters=location_filter
            )

            logger.info(f"📊 Found {len(candidates)} candidates (same district + active)")

        except Exception as e:
            logger.error(f"❌ Vector search failed: {e}")
            logger.error(f"   Filter used: {location_filter}")
            return []

        if not candidates:
            logger.info("⚠️  No candidates found matching location + active status")
            return []

        # ============ APPLY WHAT + WHEN FILTERS ============
        filtered_matches = []

        for candidate in candidates:
            metadata = candidate.get('metadata', {})

            # ✅ Verify location using individual fields OR JSON backup
            candidate_districts = []

            # Try individual fields first
            for i in range(1, 4):
                district = metadata.get(f'district_{i}', '').strip()
                if district:
                    candidate_districts.append(district)

            # Fallback to primary_district
            primary = metadata.get('primary_district', '').strip()
            if primary and primary not in candidate_districts:
                candidate_districts.append(primary)

            # Final fallback to JSON field
            if not candidate_districts:
                district_json = metadata.get('district_names_json', '[]')
                try:
                    import json
                    candidate_districts = json.loads(district_json)
                except:
                    pass

            candidate_districts = DistrictNormalizer.normalize_list(candidate_districts)
            district_overlap = set(incident_districts) & set(candidate_districts)

            if not district_overlap:
                continue

            # ========== WHAT MATCHING ==========
            candidate_incident_types = self._extract_incident_signatures_from_metadata(metadata)

            # ✅ ADD DEBUG LOGGING
            logger.info(f"🔍 DEBUG - Current incident signatures: {current_incident_types}")
            logger.info(f"🔍 DEBUG - Candidate incident signatures: {candidate_incident_types}")

            incident_match_score = self._calculate_incident_similarity(
                current_incident_types, candidate_incident_types
            )

            logger.info(f"🔍 DEBUG - Incident match score: {incident_match_score:.2f}")

            if incident_match_score < 0.5:
                logger.debug(f"⏭️  Skipping - Low incident match: {incident_match_score:.2f}")
                continue

            # ========== WHEN MATCHING ==========
            candidate_temporal = self._safe_get_dict(metadata, 'temporal_info')
            candidate_days_ago = candidate_temporal.get('days_ago')

            # ✅ Handle None for candidate days_ago
            if candidate_days_ago is None:
                candidate_days_ago = 999

            temporal_distance = abs(current_days_ago - candidate_days_ago)

            # ✅ Skip temporal check if either has no temporal info
            if current_days_ago == 999 or candidate_days_ago == 999:
                temporal_score = 0.5  # Neutral score
            elif temporal_distance > TEMPORAL_WINDOW_DAYS:
                logger.debug(f"⏭️  Skipping - Outside temporal window: {temporal_distance} days apart")
                continue
            else:
                temporal_score = 1.0 - (temporal_distance / TEMPORAL_WINDOW_DAYS)

            # ========== ENHANCED SCORING ==========
            base_similarity = candidate['similarity']

            location_boost = 0.30 * (len(district_overlap) / len(incident_districts))
            incident_boost = 0.20 * incident_match_score
            temporal_boost = 0.15 * temporal_score
            entity_boost = 0.10 if self._has_entity_overlap(ner_data, metadata) else 0.0
            sentiment_boost = 0.05 if self._has_sentiment_alignment(ner_data, metadata) else 0.0

            final_score = (
                    base_similarity + location_boost + incident_boost +
                    temporal_boost + entity_boost + sentiment_boost
            )

            if final_score > 1.0: final_score = 1.0

            threshold = self.embedding_service.get_similarity_threshold("", language, source_type)

            if final_score >= threshold:
                candidate['enhanced_similarity'] = final_score
                candidate['district_overlap'] = list(district_overlap)
                candidate['incident_match_score'] = incident_match_score
                candidate['temporal_distance_days'] = temporal_distance if temporal_distance != 999 else None
                candidate['temporal_score'] = temporal_score
                candidate['boost_reasons'] = [
                    f"location_match ({location_boost:.2f})",
                    f"incident_match ({incident_boost:.2f})",
                    f"temporal_proximity ({temporal_boost:.2f})"
                ]
                filtered_matches.append(candidate)

        filtered_matches.sort(key=lambda x: x['enhanced_similarity'], reverse=True)

        logger.info(f"✅ THREE-PHASE MATCH: {len(filtered_matches)} topics passed all filters")

        return filtered_matches

    def _extract_incident_signatures(self, ner_data: Dict[str, Any]) -> List[str]:
        """Extract incident type signatures for WHAT matching"""
        signatures = []

        # From primary classification
        primary = ner_data.get('primary_classification', {})
        if primary.get('sub_category'):
            sub_cat = primary['sub_category'].lower()
            signatures.append(sub_cat)
            logger.debug(f"   📌 From primary classification: {sub_cat}")

        if primary.get('broad_category'):
            broad_cat = primary['broad_category'].lower()
            signatures.append(broad_cat)
            logger.debug(f"   📌 From broad category: {broad_cat}")

        # From incidents - extract key terms
        for incident in ner_data.get('incidents', [])[:3]:
            # Keep full incident text
            signatures.append(incident.lower())

            # Also extract individual words (3+ chars)
            incident_clean = re.sub(r'[^\w\s]', ' ', incident.lower())
            key_terms = [w for w in incident_clean.split() if len(w) > 3]
            signatures.extend(key_terms)
            logger.debug(f"   📌 From incident: {incident} → {key_terms}")

        # From category keywords
        for cat in ner_data.get('category_classifications', [])[:2]:
            if cat.get('matched_keywords'):
                keywords = [kw.lower() for kw in cat['matched_keywords'][:3]]
                signatures.extend(keywords)
                logger.debug(f"   📌 From category keywords: {keywords}")

        signatures = list(set(signatures))  # Remove duplicates
        logger.info(f"   🎯 Final signatures: {signatures}")

        return signatures

    def _extract_incident_signatures_from_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """Extract incident signatures from stored topic metadata"""
        signatures = []

        # ✅ Get primary classification
        primary = self._safe_get_dict(metadata, 'primary_classification')
        if primary.get('sub_category'):
            signatures.append(primary['sub_category'].lower())

        # ✅ Get incidents from metadata
        incidents = self._safe_get_list(metadata, 'incidents')
        logger.debug(f"   📋 Raw incidents from metadata: {incidents}")

        for incident in incidents[:3]:
            incident_clean = re.sub(r'[^\w\s]', ' ', incident.lower())
            key_terms = [w for w in incident_clean.split() if len(w) > 3]
            signatures.extend(key_terms)

        # ✅ Get category keywords from stored classifications
        category_classifications = self._safe_get_list(metadata, 'category_classifications')
        for cat in category_classifications[:2]:
            if isinstance(cat, dict) and cat.get('matched_keywords'):
                keywords = cat['matched_keywords']
                if isinstance(keywords, list):
                    signatures.extend([kw.lower() for kw in keywords[:3]])

        signatures = list(set(signatures))  # Remove duplicates
        logger.debug(f"   🔍 Extracted signatures from metadata: {signatures}")

        return signatures

    def _calculate_incident_similarity(self, current_sigs: List[str], candidate_sigs: List[str]) -> float:
        """Calculate similarity between incident signatures using fuzzy matching"""
        if not current_sigs or not candidate_sigs:
            return 0.0

        from difflib import SequenceMatcher

        match_scores = []
        for curr_sig in current_sigs:
            best_match = 0.0
            for cand_sig in candidate_sigs:
                similarity = SequenceMatcher(None, curr_sig, cand_sig).ratio()
                best_match = max(best_match, similarity)
            match_scores.append(best_match)

        return sum(match_scores) / len(match_scores) if match_scores else 0.0

    def _safe_get_list(self, data: Dict, key: str) -> List:
        """Safely get list from dict, handling JSON strings"""
        value = data.get(key, [])
        if isinstance(value, str):
            try:
                import json
                return json.loads(value)
            except:
                return []
        return value if isinstance(value, list) else []

    def _safe_get_dict(self, data: Dict, key: str) -> Dict:
        """Safely get dict from dict, handling JSON strings"""
        value = data.get(key, {})
        if isinstance(value, str):
            try:
                import json
                return json.loads(value)
            except:
                return {}
        return value if isinstance(value, dict) else {}


    # def validate_and_correct_categories(self, category_classifications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    #     """Validate categories against allowed list"""
    #     validated = []
    #
    #     for classification in category_classifications:
    #         broad_cat = classification.get('broad_category', '').upper()
    #         sub_cat = classification.get('sub_category', '').upper()
    #
    #         if broad_cat not in ALLOWED_CATEGORIES:
    #             logger.warning(f"❌ Invalid broad category: '{broad_cat}'")
    #             broad_cat = self._find_closest_category(broad_cat, ALLOWED_CATEGORIES.keys())
    #             if not broad_cat:
    #                 continue
    #
    #         if sub_cat not in ALLOWED_CATEGORIES[broad_cat]:
    #             logger.warning(f"❌ Invalid sub category: '{sub_cat}' under '{broad_cat}'")
    #             sub_cat = self._find_closest_category(sub_cat, ALLOWED_CATEGORIES[broad_cat])
    #             if not sub_cat:
    #                 continue
    #
    #         classification['broad_category'] = broad_cat
    #         classification['sub_category'] = sub_cat
    #         validated.append(classification)
    #
    #         logger.info(f"✅ Validated: {broad_cat} > {sub_cat}")
    #
    #     if not validated:
    #         validated.append({
    #             'broad_category': 'UNCATEGORIZED',
    #             'sub_category': 'GENERAL',
    #             'confidence': 0.0,
    #             'matched_keywords': [],
    #             'reasoning': 'No categories matched allowed list'
    #         })
    #
    #     return validated
    #
    # def _find_closest_category(self, input_cat: str, allowed_cats) -> Optional[str]:
    #         """Find closest matching category using fuzzy matching"""
    #         from difflib import get_close_matches
    #
    #         allowed_list = list(allowed_cats)
    #         matches = get_close_matches(input_cat, allowed_list, n=1, cutoff=0.75)
    #
    #         return matches[0] if matches else None

    def get_or_create_unassigned_topic(self) -> str:
        """Get or create the special 'Unassigned Posts' topic"""
        try:
            with self.db_manager.get_session() as session:
                from database.models import Topic

                unassigned_topic = session.query(Topic).filter(
                    Topic.id == UNASSIGNED_TOPIC_ID
                ).first()

                if unassigned_topic:
                    return UNASSIGNED_TOPIC_ID

                # Create it
                unassigned_topic = Topic(
                    id=UNASSIGNED_TOPIC_ID,
                    title=UNASSIGNED_TOPIC_TITLE_HINDI,
                    description="पोस्ट जिनमें पर्याप्त जानकारी नहीं है या जिन्हें किसी विषय में वर्गीकृत नहीं किया जा सका",
                    primary_language="hindi",
                    content_count=0,
                    confidence_score=0.0,
                    status='active',
                    representative_text="Unassigned posts collection"
                )
                session.add(unassigned_topic)

                # Add to vector DB
                dummy_embedding = np.zeros(1024)
                self.vector_service.add_topic(
                    topic_id=UNASSIGNED_TOPIC_ID,
                    embedding=dummy_embedding,
                    metadata={
                        "topic_title": UNASSIGNED_TOPIC_TITLE_HINDI,
                        "is_special_topic": True,
                        "topic_status": "active",
                        "primary_language": "hindi"
                    },
                    document="Special topic for unassigned posts"
                )

                logger.info(f"✅ Created Unassigned Posts topic")
                return UNASSIGNED_TOPIC_ID

        except Exception as e:
            logger.error(f"Failed to create unassigned topic: {e}")
            return UNASSIGNED_TOPIC_ID

    def assign_to_unassigned_topic(self,
                                   original_text: str,
                                   ner_data: Dict[str, Any],
                                   reason: str,
                                   user_id: Optional[str] = None) -> Dict[str, Any]:
        """Assign text to Unassigned Posts topic"""
        unassigned_id = self.get_or_create_unassigned_topic()

        with self.db_manager.get_session() as session:
            from database.models import TextEntry, Topic

            text_entry = TextEntry(
                original_text=original_text,
                extracted_entities=ner_data,
                sentiment_data=ner_data.get('sentiment', {}),
                user_id=user_id,
                topic_id=unassigned_id,
                notes=f"Unassigned reason: {reason}"
            )
            session.add(text_entry)

            topic = session.query(Topic).filter(Topic.id == unassigned_id).first()
            if topic:
                topic.content_count += 1

        logger.info(f"📌 Assigned to Unassigned Posts: {reason}")

        return {
            "action": "assigned_to_unassigned",
            "topic_id": unassigned_id,
            "topic_title": UNASSIGNED_TOPIC_TITLE_HINDI,
            "similarity_score": 0.0,
            "confidence": "unassigned",
            "reason": reason,
            "boost_reasons": [],
            "user_can_reassign": True
        }

    def _generate_topic_title_with_llm_hindi(self,
                                             ner_data: Dict[str, Any],
                                             original_text: str) -> str:
        """Generate topic title ALWAYS in Hindi using LLM"""

        # Get Hindi district names
        canonical_districts = ner_data.get('district_names', [])
        hindi_districts = [DistrictNormalizer.get_hindi_name(d) for d in canonical_districts]

        prompt = f"""आप एक हिंदी समाचार शीर्षक विशेषज्ञ हैं। इस घटना के लिए एक संक्षिप्त, विशिष्ट विषय शीर्षक (10-15 शब्द) बनाएं।

    अनिवार्य आवश्यकताएं:
    1. शीर्षक केवल हिंदी में होना चाहिए (देवनागरी लिपि)
    2. स्थान (जिला/थाना) अवश्य शामिल करें
    3. विशिष्ट घटना/अपराध का प्रकार बताएं
    4. प्रारूप: [स्थान] - [विशिष्ट घटना]
    5. सामान्य शब्दों से बचें, विशिष्ट रहें

    निकाली गई जानकारी:
    - जिले: {hindi_districts}
    - थाने: {ner_data.get('thana_names', [])}
    - घटनाएं: {ner_data.get('incidents', [])}
    - कार्यक्रम: {ner_data.get('events', [])}
    - श्रेणी: {ner_data.get('primary_classification', {}).get('sub_category', 'अज्ञात')}

    मूल पाठ (संदर्भ के लिए):
    {original_text[:500]}

    अच्छे उदाहरण:
    - "लखनऊ गोमती नगर - नाबालिग से दुष्कर्म का मामला"
    - "मुरादाबाद - डकैती में तीन आरोपी गिरफ्तार"
    - "आगरा - ऑपरेशन कन्विक्शन के तहत BNS की पहली सजा"

    केवल हिंदी शीर्षक लिखें, कुछ और नहीं:"""

        try:
            if self.ner_extractor._model_loaded:
                response = self._generate_with_llm(prompt, max_tokens=100, temperature=0.3)

                title = response.strip()

                # Validation
                if len(title) < 10 or not self.ner_extractor._has_devanagari(title):
                    return self._generate_hindi_title_fallback(ner_data)

                logger.info(f"✅ LLM generated Hindi title: '{title}'")
                return title

        except Exception as e:
            logger.error(f"LLM title generation failed: {e}")

        return self._generate_hindi_title_fallback(ner_data)

    def _create_new_topic_with_temporal(self, topic_id: str, title: str,
                                        original_text: str, processed_text: str,
                                        enhanced_text: str, embedding: np.ndarray,
                                        ner_data: Dict[str, Any], language: str,
                                        lang_confidence: float, source_type: str,
                                        user_id: Optional[str]):
        """Create new topic with temporal tracking and active status"""

        from datetime import datetime
        import json

        with self.db_manager.get_session() as session:
            from database.models import Topic, TextEntry

            # Parse incident date
            temporal_info = ner_data.get('temporal_info', {})
            incident_date_str = temporal_info.get('incident_date')

            try:
                incident_date = datetime.strptime(incident_date_str, '%Y-%m-%d').date()
            except:
                incident_date = datetime.now().date()

            # Create topic
            topic = Topic(
                id=topic_id,
                title=title,
                description=ner_data.get('contextual_understanding', ''),
                primary_language=language,
                content_count=1,
                confidence_score=lang_confidence,
                representative_text=original_text[:500],
                status='active',
                first_incident_date=incident_date,
                last_incident_date=incident_date,
                incident_count=1
            )
            session.add(topic)

            # Create text entry
            text_entry = TextEntry(
                original_text=original_text,
                processed_text=processed_text,
                enhanced_text=enhanced_text,
                detected_language=language,
                language_confidence=lang_confidence,
                source_type=source_type,
                similarity_score=0.0,
                confidence_level="high",
                extracted_entities=ner_data,
                sentiment_data=ner_data.get('advanced_sentiment', {}),
                user_id=user_id,
                topic_id=topic_id
            )
            session.add(text_entry)

            # ✅ Get districts
            district_names = ner_data.get('district_names', [])

            # Vector DB metadata
            vector_metadata = {
                "topic_title": title,
                "primary_language": language,
                "source_type": source_type,
                "content_count": 1,
                "topic_status": "active",
                "first_incident_date": incident_date_str,
                "last_incident_date": incident_date_str,

                # ✅ NEW: Store as JSON string (backup)
                "district_names_json": json.dumps(district_names),

                # ✅ NEW: Store as individual searchable fields
                "primary_district": district_names[0] if len(district_names) > 0 else "",
                "district_1": district_names[0] if len(district_names) > 0 else "",
                "district_2": district_names[1] if len(district_names) > 1 else "",
                "district_3": district_names[2] if len(district_names) > 2 else "",

                "temporal_info": temporal_info,
                "incident_category": ner_data.get('incident_category', 'general'),
                "primary_classification": ner_data.get('primary_classification', {}),
                "category_classifications": ner_data.get('category_classifications', []),  # ✅ ADD THIS
                "incidents": ner_data.get('incidents', []),
                "events": ner_data.get('events', [])
            }

            self.vector_service.add_topic(
                topic_id=topic_id,
                embedding=embedding,
                metadata=vector_metadata,
                document=enhanced_text
            )

            self.stats['topics_created'] += 1
            logger.info(f"✅ Created new ACTIVE topic: '{title}'")

    def mark_topic_inactive(self,
                            topic_id: str,
                            reason: str = "Resolved",
                            user_id: Optional[str] = None) -> bool:
        """Mark a topic as inactive"""
        try:
            with self.db_manager.get_session() as session:
                from database.models import Topic
                from datetime import datetime

                topic = session.query(Topic).filter(Topic.id == topic_id).first()
                if not topic:
                    return False

                topic.status = 'inactive'
                topic.status_changed_at = datetime.now()
                topic.status_changed_by = user_id
                topic.status_reason = reason

                self.vector_service.update_topic_metadata(topic_id, {
                    "topic_status": "inactive",
                    "status_changed_at": datetime.now().isoformat()
                })

                logger.info(f"✅ Topic {topic_id} marked as INACTIVE: {reason}")
                return True

        except Exception as e:
            logger.error(f"Failed to mark topic inactive: {e}")
            return False

    def reassign_text_to_topic(self,
                               text_entry_id: int,
                               new_topic_id: str,
                               user_id: str,
                               reason: str = "Manual reassignment") -> Dict[str, Any]:
        """Allow user to move text from one topic to another"""
        try:
            with self.db_manager.get_session() as session:
                from database.models import TextEntry, Topic
                from datetime import datetime

                text_entry = session.query(TextEntry).filter(
                    TextEntry.id == text_entry_id
                ).first()

                if not text_entry:
                    return {"success": False, "error": "Text entry not found"}

                old_topic_id = text_entry.topic_id

                old_topic = session.query(Topic).filter(Topic.id == old_topic_id).first()
                new_topic = session.query(Topic).filter(Topic.id == new_topic_id).first()

                if not new_topic:
                    return {"success": False, "error": "Target topic not found"}

                # Update text entry
                text_entry.topic_id = new_topic_id
                text_entry.notes = f"{text_entry.notes or ''}\n[{datetime.now()}] Reassigned by {user_id}: {reason}"

                # Update topic counts
                if old_topic:
                    old_topic.content_count = max(0, old_topic.content_count - 1)

                new_topic.content_count += 1
                new_topic.updated_at = func.now()

                logger.info(f"✅ Text {text_entry_id} reassigned: {old_topic_id} → {new_topic_id}")

                return {
                    "success": True,
                    "text_entry_id": text_entry_id,
                    "old_topic_id": old_topic_id,
                    "new_topic_id": new_topic_id,
                    "new_topic_title": new_topic.title
                }

        except Exception as e:
            logger.error(f"Failed to reassign text: {e}")
            return {"success": False, "error": str(e)}

    def _generate_hindi_title_fallback(self, ner_data: Dict[str, Any]) -> str:
        """Fallback: Generate Hindi title using rules"""
        parts = []

        # Location in Hindi
        canonical_districts = ner_data.get('district_names', [])
        if canonical_districts:
            hindi_district = DistrictNormalizer.get_hindi_name(canonical_districts[0])
            parts.append(hindi_district)

        # Incident in Hindi
        incidents = ner_data.get('incidents', [])
        if incidents and self.ner_extractor._has_devanagari(incidents[0]):
            parts.append(incidents[0])
        else:
            category = ner_data.get('primary_classification', {}).get('sub_category', '')
            if category:
                parts.append(self._translate_category_to_hindi(category))
            else:
                parts.append("आपराधिक मामला")

        title = " - ".join(parts) if len(parts) > 1 else parts[0] if parts else "स्थानीय घटना"
        return title

    def _translate_category_to_hindi(self, category: str) -> str:
        """Translate category to Hindi"""
        translations = {
            'MURDER': 'हत्या का मामला',
            'AGAINST WOMEN': 'महिलाओं के विरुद्ध अपराध',
            'AGAINST MINORS': 'नाबालिगों के विरुद्ध अपराध',
            'ROBBERY': 'लूटपाट',
            'THEFT': 'चोरी',
            'KIDNAPPING': 'अपहरण',
            'ASSAULT': 'मारपीट',
            'TRAFFIC JAM': 'यातायात जाम',
            'ACCIDENT': 'दुर्घटना',
            'CORRUPTION': 'भ्रष्टाचार',
            'PROTEST': 'प्रदर्शन',
            'COMMUNAL': 'सांप्रदायिक मामला',
            'CASTEISM': 'जातिवाद',
        }
        return translations.get(category.upper(), 'आपराधिक मामला')

    def _generate_with_llm(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate text using loaded Mistral model"""

        if self.ner_extractor._loading_method == 'vllm':
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95
            )

            outputs = self.ner_extractor.model.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text.strip()

        elif self.ner_extractor._loading_method == 'mlx':
            from mlx_lm import generate as mlx_generate
            return mlx_generate(
                self.ner_extractor.model,
                self.ner_extractor.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature
            )

        else:  # transformers
            import torch
            inputs = self.ner_extractor.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)

            if hasattr(self.ner_extractor.model, 'device'):
                inputs = {k: v.to(self.ner_extractor.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.ner_extractor.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )

            full_response = self.ner_extractor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return full_response[len(prompt):].strip()

    # REPLACE the process_text_batch method

    async def process_text_batch(self,
                                 texts: List[str],
                                 source_type: str = "unknown",
                                 user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        ✅ FIXED: Batch processing with COMPLETE pipeline matching process_text_complete

        Pipeline Steps:
        0. Text Cleaning
        1. Validation
        2. Language & Preprocessing
        3. NER Extraction (includes temporal & sentiment)
        4. District Normalization
        5. Temporal Info Check
        6. Category Classification
        7. Enhanced Text Creation
        8. Generate Embedding
        9. Three-Phase Matching
        10. Topic Assignment
        11. Filter Low Confidence
        12. Build Result
        """
        start_time = time.time()
        results = []

        try:
            logger.info(f"🚀 Starting batch processing for {len(texts)} texts")

            # ========== STEP 0: CLEAN ALL TEXTS ==========
            logger.info("📝 STEP 0: Text cleaning...")
            cleaned_data = []

            for i, text in enumerate(texts):
                if not text or len(text.strip()) < 3:
                    cleaned_data.append({
                        "error": "Invalid text",
                        "original_text": text
                    })
                    continue

                # Clean text based on source
                cleaning_result = self.text_cleaner.clean_text(
                    text,
                    source_type=source_type,
                    use_ai=True
                )

                cleaned_text = cleaning_result["cleaned_text"]

                if not cleaned_text or len(cleaned_text) < 10:
                    cleaned_data.append({
                        "error": "Text too short after cleaning",
                        "original_text": text
                    })
                    continue

                cleaned_data.append({
                    "original_text": text,
                    "cleaned_text": cleaned_text,
                    "cleaning_metadata": cleaning_result
                })

            valid_items = [item for item in cleaned_data if "error" not in item]
            logger.info(f"✅ STEP 0 COMPLETE: Cleaned {len(valid_items)}/{len(texts)} texts")

            if not valid_items:
                logger.warning("No valid texts after cleaning")
                return [{
                    "input_text": item["original_text"],
                    "error": item.get("error", "Unknown error"),
                    "action": "error",
                    "timestamp": time.time()
                } for item in cleaned_data]

            # ========== STEP 1: VALIDATION (already done in STEP 0) ==========
            logger.info("✅ STEP 1 COMPLETE: Validation done during cleaning")

            # ========== STEP 2: LANGUAGE & PREPROCESSING ==========
            logger.info("🌍 STEP 2: Language detection and preprocessing...")

            for item in valid_items:
                cleaned_text = item["cleaned_text"]

                # Detect language
                detected_language, lang_confidence = self.embedding_service.detect_language(cleaned_text)
                item["detected_language"] = detected_language
                item["language_confidence"] = lang_confidence

                # Preprocess
                processed_text = self.embedding_service.preprocess_text(cleaned_text)
                item["processed_text"] = processed_text

            logger.info(f"✅ STEP 2 COMPLETE: Language detection done")

            # ========== STEP 3: BATCH NER EXTRACTION ==========
            logger.info("🤖 STEP 3: Batch NER extraction (includes temporal & sentiment)...")

            batch_ner_results = self.ner_extractor.extract_batch(
                [item["cleaned_text"] for item in valid_items],
                max_tokens=1500,
                temperature=0.1
            )

            # Add NER to items
            for i, item in enumerate(valid_items):
                if i < len(batch_ner_results):
                    item["ner_data"] = batch_ner_results[i]
                else:
                    item["ner_data"] = {}

            logger.info(f"✅ STEP 3 COMPLETE: NER extraction done (includes temporal & sentiment)")

            # ========== STEP 4: DISTRICT NORMALIZATION ==========
            logger.info("🗺️ STEP 4: District normalization...")

            for item in valid_items:
                ner_data = item["ner_data"]

                # Normalize districts
                ner_data['district_names'] = DistrictNormalizer.normalize_list(
                    ner_data.get('district_names', [])
                )

                # Normalize in location analysis
                if 'incident_location_analysis' in ner_data:
                    loc_analysis = ner_data['incident_location_analysis']
                    loc_analysis['incident_districts'] = DistrictNormalizer.normalize_list(
                        loc_analysis.get('incident_districts', [])
                    )
                    loc_analysis['related_districts'] = DistrictNormalizer.normalize_list(
                        loc_analysis.get('related_districts', [])
                    )

            logger.info("✅ STEP 4 COMPLETE: District normalization done")

            # ========== STEP 5: TEMPORAL INFO CHECK ==========
            logger.info("⏰ STEP 5: Temporal info validation...")

            for item in valid_items:
                ner_data = item["ner_data"]

                # Ensure temporal info exists
                if 'temporal_info' not in ner_data or not ner_data['temporal_info'].get('incident_date'):
                    ner_data['temporal_info'] = self.ner_extractor._extract_temporal_info(item["cleaned_text"])

            logger.info("✅ STEP 5 COMPLETE: Temporal info validated")

            # ========== STEP 6: CATEGORY CLASSIFICATION ==========
            logger.info("📂 STEP 6: Category classification...")

            for item in valid_items:
                cleaned_text = item["cleaned_text"]
                ner_data = item["ner_data"]

                classification_result = self.keyword_classifier.classify(cleaned_text, ner_data)
                ner_data['category_classifications'] = classification_result['category_classifications']
                ner_data['primary_classification'] = classification_result['primary_classification']

            logger.info("✅ STEP 6 COMPLETE: Category classification done")

            # ========== STEP 7: ENHANCED TEXT CREATION ==========
            logger.info("✨ STEP 7: Enhanced text creation...")

            for item in valid_items:
                processed_text = item["processed_text"]
                ner_data = item["ner_data"]

                enhanced_text = self.embedding_service.create_enhanced_text(processed_text, ner_data)
                item["enhanced_text"] = enhanced_text

            logger.info("✅ STEP 7 COMPLETE: Enhanced text created")

            # ========== STEP 8: GENERATE EMBEDDINGS ==========
            logger.info("🧮 STEP 8: Generating embeddings...")

            # Batch embedding generation
            enhanced_texts = [item["enhanced_text"] for item in valid_items]
            embeddings = self.embedding_service.generate_embeddings(enhanced_texts)

            for i, item in enumerate(valid_items):
                item["query_embedding"] = embeddings[i]

            logger.info("✅ STEP 8 COMPLETE: Embeddings generated")

            # ========== STEP 9-12: PROCESS EACH TEXT ==========
            logger.info("🔄 STEPS 9-12: Topic matching and assignment...")

            for i, item in enumerate(cleaned_data):
                try:
                    # Handle error items
                    if "error" in item:
                        results.append({
                            "input_text": item["original_text"],
                            "error": item["error"],
                            "action": "error",
                            "processing_time_ms": 0,
                            "timestamp": time.time()
                        })
                        continue

                    cleaned_text = item["cleaned_text"]
                    ner_data = item["ner_data"]
                    query_embedding = item["query_embedding"]
                    detected_language = item["detected_language"]
                    lang_confidence = item["language_confidence"]

                    # ========== STEP 9: THREE-PHASE MATCHING ==========
                    similar_topics = self._find_similar_topics_three_phase(
                        query_embedding, ner_data, detected_language, source_type
                    )

                    # ========== STEP 10: TOPIC ASSIGNMENT ==========
                    has_location = bool(ner_data.get('district_names'))
                    has_incident = bool(ner_data.get('incidents') or ner_data.get('events'))

                    information_score = (
                            (1.0 if has_location else 0.0) +
                            (1.0 if has_incident else 0.0) +
                            (0.5 if len(ner_data.get('contextual_understanding', '')) > 50 else 0.0)
                    )

                    if information_score < 1.5:
                        # Insufficient information - assign to unassigned
                        topic_result = self.assign_to_unassigned_topic(
                            cleaned_text, ner_data,
                            reason=f"Insufficient information (score: {information_score:.1f})",
                            user_id=user_id
                        )

                    elif similar_topics:
                        # Group with existing topic
                        best_match = similar_topics[0]
                        topic_id = best_match['topic_id']

                        self._update_existing_topic(topic_id, cleaned_text, ner_data, user_id)

                        topic_result = {
                            "action": "grouped",
                            "topic_id": topic_id,
                            "topic_title": best_match['metadata'].get('topic_title'),
                            "similarity_score": best_match['enhanced_similarity'],
                            "confidence": self._calculate_confidence(best_match['enhanced_similarity']),
                            "boost_reasons": best_match.get('boost_reasons', []),
                            "temporal_distance_days": best_match.get('temporal_distance_days'),
                            "incident_match_score": best_match.get('incident_match_score')
                        }

                    else:
                        # Create new topic
                        topic_id = str(uuid.uuid4())
                        topic_title = self._generate_topic_title_with_llm_hindi(ner_data, cleaned_text)

                        self._create_new_topic_with_temporal(
                            topic_id, topic_title, cleaned_text, item["processed_text"],
                            item["enhanced_text"], query_embedding, ner_data,
                            detected_language, lang_confidence, source_type, user_id
                        )

                        topic_result = {
                            "action": "new_topic_created",
                            "topic_id": topic_id,
                            "topic_title": topic_title,
                            "similarity_score": 0.0,
                            "confidence": "high",
                            "boost_reasons": []
                        }

                    # ========== STEP 11: FILTER LOW CONFIDENCE ==========
                    ner_data = self.filter_low_confidence_data(ner_data, min_confidence=0.6)

                    # ========== STEP 12: BUILD RESULT ==========
                    # Clean up NER data to avoid duplication
                    cleaned_entities = {k: v for k, v in ner_data.items() if k not in [
                        'temporal_info', 'advanced_sentiment', 'category_classifications',
                        'primary_classification', 'incident_location_analysis'
                    ]}

                    result = {
                        "input_text": item["original_text"],
                        "cleaned_text": cleaned_text,
                        "detected_language": detected_language,
                        "language_confidence": lang_confidence,
                        "action": topic_result["action"],
                        "topic_title": topic_result.get("topic_title", ""),
                        "topic_id": topic_result["topic_id"],
                        "similarity_score": topic_result["similarity_score"],
                        "confidence": topic_result["confidence"],
                        "source_type": source_type,
                        "embedding_model": "BAAI/bge-m3",
                        "processing_time_ms": 0,  # Updated below

                        # Root level fields
                        "temporal_info": ner_data.get('temporal_info', {}),
                        "advanced_sentiment": ner_data.get('advanced_sentiment', {}),
                        "category_classifications": ner_data.get('category_classifications', []),
                        "primary_classification": ner_data.get('primary_classification', {}),
                        "incident_location_analysis": ner_data.get('incident_location_analysis', {}),

                        # Cleaned entities
                        "extracted_entities": cleaned_entities,

                        "boost_reasons": topic_result.get("boost_reasons", []),
                        "temporal_distance_days": topic_result.get("temporal_distance_days"),
                        "incident_match_score": topic_result.get("incident_match_score"),
                        "can_reassign": topic_result.get("user_can_reassign", False),

                        # Cleaning metadata
                        "text_cleaning": {
                            "applied": item["cleaning_metadata"]["cleaning_applied"],
                            "reduction_pct": item["cleaning_metadata"]["reduction_percentage"]
                        } if item["cleaning_metadata"]["cleaning_applied"] else None,

                        "timestamp": time.time()
                    }

                    results.append(result)

                    if (i + 1) % 10 == 0:
                        logger.info(f"  Processed {i + 1}/{len(valid_items)} texts")

                except Exception as e:
                    logger.error(f"Failed to process batch item {i}: {e}", exc_info=True)
                    results.append({
                        "input_text": item.get("original_text", ""),
                        "error": str(e),
                        "action": "error",
                        "processing_time_ms": 0,
                        "timestamp": time.time()
                    })

            # Update processing time
            total_time = (time.time() - start_time) * 1000
            avg_time = total_time / len(results) if results else 0

            for result in results:
                if "error" not in result:
                    result["processing_time_ms"] = int(avg_time)

            logger.info(f"✅ STEPS 9-12 COMPLETE")
            logger.info(f"✅ BATCH PROCESSING COMPLETE: {len(results)} texts in {total_time:.2f}ms")

            return results

        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            return [{
                "input_text": text,
                "error": str(e),
                "action": "error",
                "processing_time_ms": 0,
                "timestamp": time.time()
            } for text in texts]



    async def _parallel_ai_clean_batch(self, texts: List[str], source_type: str) -> List[Dict[str, Any]]:
        """
        Clean multiple texts in parallel using AI (for vLLM batch mode)
        """
        if self.text_cleaner._loading_method != 'vllm':
            # Sequential for non-vLLM
            return [self.text_cleaner.clean_text(text, source_type, use_ai=True) for text in texts]

        # ✅ vLLM batch processing
        from vllm import SamplingParams

        prompts = [
            self.text_cleaner._build_cleaning_prompt(text, source_type)
            for text in texts
        ]

        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=2048,
            top_p=0.9
        )

        outputs = self.text_cleaner.model.generate(prompts, sampling_params)

        results = []
        for i, output in enumerate(outputs):
            try:
                ai_response = output.outputs[0].text.strip()
                cleaned_text = self.text_cleaner._parse_ai_response(ai_response, texts[i])

                results.append({
                    "cleaned_text": cleaned_text,
                    "original_length": len(texts[i]),
                    "cleaned_length": len(cleaned_text),
                    "reduction_percentage": round((1 - len(cleaned_text) / len(texts[i])) * 100, 1),
                    "cleaning_method": "ai_powered_batch",
                    "cleaning_applied": True
                })
            except Exception as e:
                logger.error(f"Batch cleaning failed for text {i}: {e}")
                results.append(self.text_cleaner._fallback_clean(texts[i], source_type))

        return results

    def _enhance_ner_with_categorization(self, ner_data: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """Enhance NER data with incident categorization"""
        enhanced_data = ner_data.copy()

        # Categorize the type of incident
        incident_category = self._categorize_incident(original_text)
        enhanced_data['incident_category'] = incident_category

        # Extract key phrases for better topic titles
        key_phrases = self._extract_key_phrases(original_text, ner_data)
        enhanced_data['key_phrases'] = key_phrases

        # Determine severity/priority
        severity = self._determine_severity(original_text, ner_data)
        enhanced_data['severity'] = severity

        return enhanced_data

    def _categorize_incident(self, text: str) -> str:
        """Categorize the type of incident based on content patterns"""
        text_lower = text.lower()

        category_scores = {}

        for category, patterns in self.incident_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                score += matches
            category_scores[category] = score

        # Find the category with highest score
        if category_scores:
            max_category = max(category_scores, key=category_scores.get)
            if category_scores[max_category] > 0:
                return max_category

        return 'general'

    def _extract_key_phrases(self, text: str, ner_data: Dict[str, Any]) -> List[str]:
        """Extract key phrases for topic title generation"""
        key_phrases = []

        # Crime-specific phrases
        crime_phrases = [
            r'नाबालिग से दुष्कर्म', r'आजीवन कारावास', r'मु0अ0सं0 \d+',
            r'धारा \d+', r'BNS की धाराओं', r'पोक्सो एक्ट',
            r'rape case', r'murder case', r'theft case', r'FIR number',
            r'चोरी का मामला', r'हत्या का मामला', r'गिरफ्तारी'
        ]

        for phrase_pattern in crime_phrases:
            matches = re.findall(phrase_pattern, text, re.IGNORECASE)
            key_phrases.extend(matches)

        # Operations and campaigns
        if ner_data.get('events'):
            key_phrases.extend(ner_data['events'][:2])

        return key_phrases[:5]  # Limit to 5 key phrases

    def _determine_severity(self, text: str, ner_data: Dict[str, Any]) -> str:
        """Determine incident severity"""
        text_lower = text.lower()

        high_severity_indicators = [
            'murder', 'हत्या', 'rape', 'दुष्कर्म', 'terrorist', 'आतंकवादी',
            'explosion', 'विस्फोट', 'kidnapping', 'अपहरण', 'आजीवन कारावास'
        ]

        medium_severity_indicators = [
            'theft', 'चोरी', 'assault', 'मारपीट', 'fraud', 'धोखाधड़ी',
            'accident', 'दुर्घटना', 'गिरफ्तार', 'arrest'
        ]

        for indicator in high_severity_indicators:
            if indicator in text_lower:
                return 'high'

        for indicator in medium_severity_indicators:
            if indicator in text_lower:
                return 'medium'

        return 'low'

    def _generate_intelligent_topic_title(self, ner_data: Dict[str, Any], language: str, original_text: str = "") -> str:
        """Generate intelligent, descriptive topic titles based on incident analysis"""

        # Get enhanced NER data
        incident_category = ner_data.get('incident_category', 'general')
        key_phrases = ner_data.get('key_phrases', [])
        severity = ner_data.get('severity', 'low')

        title_parts = []

        # Strategy 1: Location-based title with incident details
        location_part = self._build_location_part(ner_data)
        incident_part = self._build_incident_part(ner_data, incident_category, original_text)

        # Strategy 2: Build title based on incident category
        if incident_category == 'legal_action':
            title_parts = self._build_legal_action_title(ner_data, location_part, original_text)
        elif incident_category == 'crime':
            title_parts = self._build_crime_title(ner_data, location_part, original_text)
        elif incident_category == 'police_action':
            title_parts = self._build_police_action_title(ner_data, location_part, original_text)
        elif incident_category == 'accident':
            title_parts = self._build_accident_title(ner_data, location_part, original_text)
        elif incident_category == 'protest_political':
            title_parts = self._build_political_title(ner_data, location_part, original_text)
        elif incident_category == 'administrative':
            title_parts = self._build_administrative_title(ner_data, location_part, original_text)
        elif incident_category == 'social_incident':
            title_parts = self._build_social_title(ner_data, location_part, original_text)
        else:
            # Fallback to generic but descriptive title
            title_parts = self._build_generic_descriptive_title(ner_data, location_part, incident_part)

        # Ensure we have something meaningful
        if not title_parts:
            title_parts = self._build_fallback_title(ner_data, language, original_text)

        # Join parts and finalize
        separator = " - " if language == 'english' else " - "
        title = separator.join(filter(None, title_parts))

        # Post-process title
        title = self._post_process_title(title, language)

        # Track improvement
        if title not in ["सामान्य विषय", "General Topic", "सामान्य"]:
            self.stats['title_improvements'] += 1

        return title

    def _build_location_part(self, ner_data: Dict[str, Any]) -> str:
        """Build location part of title"""
        if ner_data.get('district_names'):
            return ner_data['district_names'][0]
        elif ner_data.get('thana_names'):
            return f"थाना {ner_data['thana_names'][0]}"
        elif ner_data.get('location_names'):
            return ner_data['location_names'][0]
        return ""

    def _build_incident_part(self, ner_data: Dict[str, Any], incident_category: str, text: str) -> str:
        """Build incident description part"""
        if ner_data.get('incidents'):
            return ner_data['incidents'][0]
        elif ner_data.get('events'):
            return ner_data['events'][0]
        elif incident_category != 'general':
            return incident_category.replace('_', ' ').title()
        return ""

    def _build_legal_action_title(self, ner_data: Dict[str, Any], location: str, text: str) -> List[str]:
        """Build title for legal action cases"""
        parts = []

        if location:
            parts.append(location)

        # Check for specific legal terms
        if 'BNS' in text or 'भारतीय न्याय संहिता' in text:
            parts.append("BNS के तहत सजा")
        elif 'आजीवन कारावास' in text:
            parts.append("आजीवन कारावास की सजा")
        elif 'सजा' in text or 'दंडित' in text:
            parts.append("न्यायालय का फैसला")
        else:
            parts.append("कानूनी कार्रवाई")

        # Add specific crime if available
        if 'दुष्कर्म' in text or 'rape' in text.lower():
            parts.append("दुष्कर्म मामला")
        elif ner_data.get('incidents'):
            parts.append(ner_data['incidents'][0])

        return parts

    def _build_crime_title(self, ner_data: Dict[str, Any], location: str, text: str) -> List[str]:
        """Build title for crime cases"""
        parts = []

        if location:
            parts.append(location)

        # Identify specific crime type
        crime_types = {
            'दुष्कर्म': 'दुष्कर्म मामला',
            'चोरी': 'चोरी की घटना',
            'हत्या': 'हत्या का मामला',
            'लूट': 'लूट की घटना',
            'डकैती': 'डकैती का मामला',
            'बलात्कार': 'बलात्कार का मामला',
            'अपहरण': 'अपहरण का मामला'
        }

        text_lower = text.lower()
        for crime, title in crime_types.items():
            if crime in text or crime.lower() in text_lower:
                parts.append(title)
                break
        else:
            if ner_data.get('incidents'):
                parts.append(ner_data['incidents'][0])
            else:
                parts.append("आपराधिक मामला")

        return parts

    def _build_police_action_title(self, ner_data: Dict[str, Any], location: str, text: str) -> List[str]:
        """Build title for police action cases"""
        parts = []

        if location:
            parts.append(location)

        if ner_data.get('events'):
            # Clean event names like "ऑपरेशन कन्विक्शन"
            event = ner_data['events'][0]
            parts.append(event)
        elif 'गिरफ्तार' in text:
            parts.append("गिरफ्तारी")
        elif 'छापेमारी' in text:
            parts.append("छापेमारी")
        else:
            parts.append("पुलिस कार्रवाई")

        return parts

    def _build_accident_title(self, ner_data: Dict[str, Any], location: str, text: str) -> List[str]:
        """Build title for accident cases"""
        parts = []

        if location:
            parts.append(location)

        if 'सड़क दुर्घटना' in text:
            parts.append("सड़क दुर्घटना")
        elif 'आग' in text:
            parts.append("आग की घटना")
        elif 'दुर्घटना' in text:
            parts.append("दुर्घटना")
        else:
            parts.append("अकस्मात घटना")

        return parts

    def _build_political_title(self, ner_data: Dict[str, Any], location: str, text: str) -> List[str]:
        """Build title for political/protest cases"""
        parts = []

        if location:
            parts.append(location)

        if 'चुनाव' in text:
            parts.append("चुनावी गतिविधि")
        elif 'प्रदर्शन' in text:
            parts.append("प्रदर्शन")
        elif 'धरना' in text:
            parts.append("धरना")
        elif 'रैली' in text:
            parts.append("रैली")
        else:
            parts.append("राजनीतिक गतिविधि")

        return parts

    def _build_administrative_title(self, ner_data: Dict[str, Any], location: str, text: str) -> List[str]:
        """Build title for administrative cases"""
        parts = []

        if location:
            parts.append(location)

        if ner_data.get('organisation_names'):
            org = ner_data['organisation_names'][0]
            parts.append(f"{org} की गतिविधि")
        elif 'योजना' in text:
            parts.append("सरकारी योजना")
        elif 'कार्यक्रम' in text:
            parts.append("सरकारी कार्यक्रम")
        else:
            parts.append("प्रशासनिक कार्य")

        return parts

    def _build_social_title(self, ner_data: Dict[str, Any], location: str, text: str) -> List[str]:
        """Build title for social incident cases"""
        parts = []

        if location:
            parts.append(location)

        if 'जातिवाद' in text:
            parts.append("जातिवाद की घटना")
        elif 'भेदभाव' in text:
            parts.append("भेदभाव का मामला")
        elif 'धार्मिक' in text:
            parts.append("धार्मिक मामला")
        else:
            parts.append("सामाजिक घटना")

        return parts

    def _build_generic_descriptive_title(self, ner_data: Dict[str, Any], location: str, incident: str) -> List[str]:
        """Build generic but descriptive title"""
        parts = []

        # Always try to include location
        if location:
            parts.append(location)

        # Add incident or event
        if incident:
            parts.append(incident)
        elif ner_data.get('incidents'):
            parts.append(ner_data['incidents'][0])
        elif ner_data.get('events'):
            parts.append(ner_data['events'][0])

        # Add organization if available
        if ner_data.get('organisation_names') and len(parts) < 2:
            org = ner_data['organisation_names'][0]
            if len(org) < 20:  # Avoid very long org names
                parts.append(org)

        return parts

    def _build_fallback_title(self, ner_data: Dict[str, Any], language: str, text: str) -> List[str]:
        """Build fallback title when other methods fail"""

        # Extract key words from contextual understanding
        context = ner_data.get('contextual_understanding', '')
        if context:
            # Extract meaningful words from context (avoid common words)
            context_words = context.split()
            meaningful_words = []

            # Common Hindi stop words to avoid
            stop_words = {'में', 'की', 'के', 'को', 'से', 'पर', 'और', 'या', 'है', 'हैं', 'था', 'थे', 'एक', 'यह', 'वह'}

            for word in context_words[:8]:  # Look at first 8 words
                if len(word) > 2 and word not in stop_words:
                    meaningful_words.append(word)
                if len(meaningful_words) >= 3:
                    break

            if meaningful_words:
                return meaningful_words

        # Location-based fallback
        if ner_data.get('district_names'):
            return [ner_data['district_names'][0], "की घटना"]
        elif ner_data.get('thana_names'):
            return [f"थाना {ner_data['thana_names'][0]}", "क्षेत्र"]
        elif ner_data.get('location_names'):
            return [ner_data['location_names'][0], "क्षेत्रीय मामला"]

        # Final fallback - but more specific than "General Topic"
        return ["स्थानीय घटना"] if language in ['hindi', 'hinglish'] else ["Local Incident"]

    def _post_process_title(self, title: str, language: str) -> str:
        """Post-process the generated title"""
        if not title or title.strip() in ["", "सामान्य विषय", "General Topic"]:
            return "स्थानीय घटना" if language in ['hindi', 'hinglish'] else "Local Incident"

        # Clean up title
        title = title.strip()

        # Remove redundant words
        title = re.sub(r'\b(की|के|में|से|पर)\b\s*-\s*', ' - ', title)
        title = re.sub(r'\s+-\s*$', '', title)  # Remove trailing dash
        title = re.sub(r'^\s*-\s*', '', title)  # Remove leading dash

        # Limit title length
        if len(title) > 80:
            parts = title.split(' - ')
            if len(parts) > 1:
                title = ' - '.join(parts[:2])
            else:
                words = title.split()[:12]
                title = ' '.join(words)

        return title

    def _find_similar_topics_with_ner(self,
                                      query_embedding: np.ndarray,
                                      ner_data: Dict[str, Any],
                                      language: str,
                                      source_type: str) -> List[Dict[str, Any]]:
        """Find similar topics using both semantic similarity and NER filtering"""

        threshold = self.embedding_service.get_similarity_threshold("", language, source_type)
        filters = {"primary_language": language}

        semantic_matches = self.vector_service.search_similar_topics(
            query_embedding=query_embedding,
            n_results=20,
            threshold=threshold * 0.7,
            filters=filters
        )

        enhanced_matches = []
        for match in semantic_matches:
            enhanced_score = self._calculate_enhanced_similarity(match, ner_data)

            if enhanced_score >= threshold:
                match['enhanced_similarity'] = enhanced_score
                match['boost_reasons'] = self._get_boost_reasons(match, ner_data)
                enhanced_matches.append(match)

        enhanced_matches.sort(key=lambda x: x['enhanced_similarity'], reverse=True)
        logger.info(f"Found {len(enhanced_matches)} similar topics after NER enhancement")
        return enhanced_matches

    def _calculate_enhanced_similarity(self, match: Dict[str, Any], ner_data: Dict[str, Any]) -> float:
        """Calculate enhanced similarity score using NER data"""
        base_score = match['similarity']
        boost = 0.0

        match_meta_data = match.get('metadata', {})

        if self._has_geographic_overlap(ner_data, match_meta_data):
            boost += 0.15

        if self._has_incident_overlap(ner_data, match_meta_data):
            boost += 0.20

        if self._has_entity_overlap(ner_data, match_meta_data):
            boost += 0.10

        if self._has_sentiment_alignment(ner_data, match_meta_data):
            boost += 0.05

        if self._has_source_alignment(ner_data, match_meta_data):
            boost += 0.03

        return min(1.0, base_score + boost)

    def _has_geographic_overlap(self, ner_data: Dict[str, Any], topic_meta_data: Dict[str, Any]) -> bool:
        """Check geographic relevance between NER data and topic"""
        def safe_get_list(data: Dict[str, Any], key: str) -> List[str]:
            value = data.get(key, [])
            if isinstance(value, str):
                try:
                    import json
                    return json.loads(value)
                except:
                    return []
            return value if isinstance(value, list) else []

        ner_districts = set(ner_data.get('district_names', []))
        topic_districts = set(safe_get_list(topic_meta_data, 'district_names'))

        ner_thanas = set(ner_data.get('thana_names', []))
        topic_thanas = set(safe_get_list(topic_meta_data, 'thana_names'))

        ner_locations = set(ner_data.get('location_names', []))
        topic_locations = set(safe_get_list(topic_meta_data, 'location_names'))

        return bool(
            (ner_districts & topic_districts) or
            (ner_thanas & topic_thanas) or
            (ner_locations & topic_locations)
        )

    def _has_incident_overlap(self, ner_data: Dict[str, Any], topic_meta_data: Dict[str, Any]) -> bool:
        """Check incident type similarity"""
        def safe_get_list(data: Dict[str, Any], key: str) -> List[str]:
            value = data.get(key, [])
            if isinstance(value, str):
                try:
                    import json
                    return json.loads(value)
                except:
                    return []
            return value if isinstance(value, list) else []

        ner_incidents = set(ner_data.get('incidents', []))
        topic_incidents = set(safe_get_list(topic_meta_data, 'incidents'))

        ner_events = set(ner_data.get('events', []))
        topic_events = set(safe_get_list(topic_meta_data, 'events'))

        return bool(
            (ner_incidents & topic_incidents) or
            (ner_events & topic_events)
        )

    def _has_entity_overlap(self, ner_data: Dict[str, Any], topic_meta_data: Dict[str, Any]) -> bool:
        """Check entity overlap"""
        def safe_get_list(data: Dict[str, Any], key: str) -> List[str]:
            value = data.get(key, [])
            if isinstance(value, str):
                try:
                    import json
                    return json.loads(value)
                except:
                    return []
            return value if isinstance(value, list) else []

        ner_persons = set(ner_data.get('person_names', []))
        topic_persons = set(safe_get_list(topic_meta_data, 'person_names'))

        ner_orgs = set(ner_data.get('organisation_names', []))
        topic_orgs = set(safe_get_list(topic_meta_data, 'organisation_names'))

        return bool(
            (ner_persons & topic_persons) or
            (ner_orgs & topic_orgs)
        )

    def _has_sentiment_alignment(self, ner_data: Dict[str, Any], topic_meta_data: Dict[str, Any]) -> bool:
        """Check sentiment alignment"""
        ner_sentiment = ner_data.get('sentiment', {}).get('label', 'neutral')

        topic_sentiment_data = topic_meta_data.get('sentiment')
        if isinstance(topic_sentiment_data, str):
            try:
                import json
                topic_sentiment_data = json.loads(topic_sentiment_data)
            except:
                topic_sentiment_data = {}

        topic_sentiment = topic_sentiment_data.get('label', 'neutral') if isinstance(topic_sentiment_data, dict) else 'neutral'
        return ner_sentiment == topic_sentiment

    def _has_source_alignment(self, ner_data: Dict[str, Any], topic_meta_data: Dict[str, Any]) -> bool:
        """Check source type alignment between current text and existing topic"""
        try:
            # Get current source type from the processing context or default
            current_source = getattr(self, '_current_source_type', 'unknown')

            # Get topic's source type from metadata
            topic_source = topic_meta_data.get('source_type', 'unknown')

            # Direct match
            if current_source == topic_source:
                return True

            # Define source type compatibility groups
            social_media_sources = {'social_media', 'twitter', 'facebook', 'instagram', 'whatsapp'}
            news_sources = {'news', 'newspaper', 'online_news', 'press_release'}
            official_sources = {'government', 'police', 'court', 'official'}

            # Check if both sources belong to the same category
            source_groups = [social_media_sources, news_sources, official_sources]

            for group in source_groups:
                if current_source in group and topic_source in group:
                    return True

            # Special alignments for mixed sources
            mixed_alignments = {
                ('police', 'official'): True,
                ('government', 'official'): True,
                ('court', 'official'): True,
                ('press_release', 'news'): True,
                ('online_news', 'news'): True
            }

            alignment_key = tuple(sorted([current_source, topic_source]))
            if alignment_key in mixed_alignments:
                return mixed_alignments[alignment_key]

            # If we can't determine alignment, consider it neutral (no boost/penalty)
            return False

        except Exception as e:
            logger.warning(f"Error in source alignment check: {e}")
            return False

    def _get_boost_reasons(self, match: Dict[str, Any], ner_data: Dict[str, Any]) -> List[str]:
        """Get reasons for similarity boost"""
        reasons = []
        match_meta_data = match.get('metadata', {})

        if self._has_geographic_overlap(ner_data, match_meta_data):
            reasons.append("geographic_match")

        if self._has_incident_overlap(ner_data, match_meta_data):
            reasons.append("incident_type_match")

        if self._has_entity_overlap(ner_data, match_meta_data):
            reasons.append("entity_overlap")

        if self._has_sentiment_alignment(ner_data, match_meta_data):
            reasons.append("sentiment_alignment")

        return reasons

    def _assign_or_create_topic(self,
                                original_text: str,
                                processed_text: str,
                                enhanced_text: str,
                                embedding: np.ndarray,
                                ner_data: Dict[str, Any],
                                similar_topics: List[Dict[str, Any]],
                                language: str,
                                lang_confidence: float,
                                source_type: str,
                                user_id: Optional[str]) -> Dict[str, Any]:
        """Assign text to existing topic or create new one with intelligent title generation"""

        if similar_topics:
            # Assign to best matching topic
            best_match = similar_topics[0]
            topic_id = best_match['topic_id']

            # Update topic in database
            self._update_existing_topic(topic_id, original_text, ner_data, user_id)

            return {
                "action": "grouped",
                "topic_id": topic_id,
                "topic_title": best_match['metadata'].get('topic_title', ''),
                "similarity_score": best_match['enhanced_similarity'],
                "confidence": self._calculate_confidence(best_match['enhanced_similarity']),
                "boost_reasons": best_match.get('boost_reasons', [])
            }
        else:
            # Create new topic with intelligent title
            topic_id = str(uuid.uuid4())
            topic_title = self._generate_intelligent_topic_title(ner_data, language, original_text)

            # Save to databases
            self._create_new_topic(
                topic_id, topic_title, original_text, processed_text, enhanced_text,
                embedding, ner_data, language, lang_confidence, source_type, user_id
            )

            return {
                "action": "new_topic_created",
                "topic_id": topic_id,
                "topic_title": topic_title,
                "similarity_score": 0.0,
                "confidence": "high",
                "boost_reasons": []
            }

    def _create_new_topic(self, topic_id: str, title: str, original_text: str,
                          processed_text: str, enhanced_text: str, embedding: np.ndarray,
                          ner_data: Dict[str, Any], language: str, lang_confidence: float,
                          source_type: str, user_id: Optional[str]):
        """Create new topic in both MySQL and vector database"""

        with self.db_manager.get_session() as session:
            from database.models import Topic, TextEntry

            # Create topic record
            topic = Topic(
                id=topic_id,
                title=title,
                description=ner_data.get('contextual_understanding', ''),
                primary_language=language,
                content_count=1,
                confidence_score=lang_confidence,
                representative_text=original_text[:500]
            )
            session.add(topic)

            # Create text entry record
            text_entry = TextEntry(
                original_text=original_text,
                processed_text=processed_text,
                enhanced_text=enhanced_text,
                detected_language=language,
                language_confidence=lang_confidence,
                source_type=source_type,
                similarity_score=0.0,
                confidence_level="high",
                extracted_entities=ner_data,
                sentiment_data=ner_data.get('sentiment', {}),
                user_id=user_id,
                topic_id=topic_id
            )
            session.add(text_entry)

            # Add to vector database with enhanced metadata
            vector_meta_data = {
                "topic_title": title,
                "primary_language": language,
                "source_type": source_type,
                "content_count": 1,
                "incident_category": ner_data.get('incident_category', 'general'),
                "severity": ner_data.get('severity', 'low'),
                **{k: v for k, v in ner_data.items() if
                   k not in ['contextual_understanding', 'incident_category', 'severity', 'key_phrases']}
            }

            success = self.vector_service.add_topic(
                topic_id=topic_id,
                embedding=embedding,
                metadata=vector_meta_data,
                document=enhanced_text
            )

            if not success:
                logger.error(f"Failed to add topic {topic_id} to vector database")
                raise Exception("Vector database operation failed")

            self.stats['topics_created'] += 1
            logger.info(f"Created new topic: {topic_id} with title: '{title}'")

    def _update_existing_topic(self, topic_id: str, original_text: str,
                               ner_data: Dict[str, Any], user_id: Optional[str]):
        """Update existing topic with new text entry"""

        with self.db_manager.get_session() as session:
            from database.models import Topic, TextEntry

            # Update topic count and timestamp
            topic = session.query(Topic).filter(Topic.id == topic_id).first()
            if topic:
                topic.content_count += 1
                topic.updated_at = func.now()
                logger.info(f"Updated topic {topic_id} - new count: {topic.content_count}")

            # Create new text entry
            text_entry = TextEntry(
                original_text=original_text,
                extracted_entities=ner_data,
                sentiment_data=ner_data.get('sentiment', {}),
                user_id=user_id,
                topic_id=topic_id
            )
            session.add(text_entry)

    def _calculate_confidence(self, similarity_score: float) -> str:
        """Calculate confidence level based on similarity score"""
        if similarity_score >= 0.9:
            return "high"
        elif similarity_score >= 0.75:
            return "medium"
        else:
            return "low"

    def _build_result(self, original_text: str, processed_text: str, enhanced_text: str,
                      topic_result: Dict[str, Any], ner_data: Dict[str, Any],
                      language: str, lang_confidence: float, source_type: str,
                      processing_time: float) -> Dict[str, Any]:
        """Build comprehensive result object with enhanced NER data"""

        # Ensure all new fields are present in ner_data
        enhanced_ner_result = self._ensure_enhanced_fields(ner_data)

        return {
            "input_text": original_text,
            "processed_text": processed_text,
            "enhanced_text": enhanced_text,
            "detected_language": language,
            "language_confidence": lang_confidence,
            "action": topic_result["action"],
            "topic_title": topic_result.get("topic_title", ""),
            "topic_id": topic_result["topic_id"],
            "similarity_score": topic_result["similarity_score"],
            "confidence": topic_result["confidence"],
            "source_type": source_type,
            "embedding_model": "BAAI/bge-m3",
            "processing_time_ms": int(processing_time),
            "extracted_entities": enhanced_ner_result,  # This now includes all new fields
            "boost_reasons": topic_result.get("boost_reasons", []),
            "timestamp": time.time()
        }

    def _ensure_enhanced_fields(self, ner_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all enhanced fields are present in the NER result"""

        # Create a copy to avoid modifying the original
        enhanced_result = ner_data.copy()

        # Ensure category_classifications field exists and is properly structured
        if "category_classifications" not in enhanced_result:
            enhanced_result["category_classifications"] = []

        # Convert to proper format if needed
        classifications = enhanced_result["category_classifications"]
        if isinstance(classifications, list):
            formatted_classifications = []
            for cls in classifications:
                if isinstance(cls, dict):
                    formatted_classifications.append({
                        "broad_category": cls.get("broad_category", ""),
                        "sub_category": cls.get("sub_category", ""),
                        "confidence": float(cls.get("confidence", 0.0)),
                        "matched_keywords": cls.get("matched_keywords", []),
                        "reasoning": cls.get("reasoning", "")
                    })
            enhanced_result["category_classifications"] = formatted_classifications

        # Ensure primary_classification field exists
        if "primary_classification" not in enhanced_result:
            enhanced_result["primary_classification"] = {
                "broad_category": "",
                "sub_category": "",
                "confidence": 0.0
            }

        # Format primary classification
        primary = enhanced_result["primary_classification"]
        if isinstance(primary, dict):
            enhanced_result["primary_classification"] = {
                "broad_category": primary.get("broad_category", ""),
                "sub_category": primary.get("sub_category", ""),
                "confidence": float(primary.get("confidence", 0.0))
            }

        # Ensure incident_location_analysis field exists
        if "incident_location_analysis" not in enhanced_result:
            enhanced_result["incident_location_analysis"] = {
                "incident_districts": [],
                "related_districts": [],
                "incident_thanas": [],
                "related_thanas": [],
                "primary_location": {}
            }

        # Format incident location analysis
        location_analysis = enhanced_result["incident_location_analysis"]
        if isinstance(location_analysis, dict):
            enhanced_result["incident_location_analysis"] = {
                "incident_districts": location_analysis.get("incident_districts", []),
                "related_districts": location_analysis.get("related_districts", []),
                "incident_thanas": location_analysis.get("incident_thanas", []),
                "related_thanas": location_analysis.get("related_thanas", []),
                "primary_location": location_analysis.get("primary_location", {})
            }

        return enhanced_result


    def _build_error_result(self, text: str, error: str, processing_time: float) -> Dict[str, Any]:
        """Build error result object"""
        return {
            "input_text": text,
            "error": error,
            "action": "error",
            "processing_time_ms": int(processing_time),
            "timestamp": time.time()
        }

    def _update_statistics(self, processing_time: float, action: str):
        """Update processing statistics"""
        self.stats['total_processed'] += 1
        self.stats['total_processing_time'] += processing_time

        if action == "new_topic_created":
            self.stats['topics_created'] += 1
        elif action == "merged_topics":
            self.stats['topics_merged'] += 1

    def _log_processing(self, result: Dict[str, Any], processing_time: float, user_id: Optional[str]):
        """Log processing operation"""
        try:
            with self.db_manager.get_session() as session:
                from database.models import ProcessingLog

                log_entry = ProcessingLog(
                    operation_type="text_clustering",
                    status="success" if "error" not in result else "failure",
                    processing_time_ms=int(processing_time),
                    error_message=result.get("error", ""),
                    user_id=user_id,
                    metadata_info={
                        "language": result.get("detected_language"),
                        "action": result.get("action"),
                        "confidence": result.get("confidence"),
                        "topic_id": result.get("topic_id"),
                        "topic_title": result.get("topic_title"),
                        "incident_category": result.get("extracted_entities", {}).get("incident_category")
                    }
                )
                session.add(log_entry)
        except Exception as e:
            logger.error(f"Failed to log processing: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        avg_time = (self.stats['total_processing_time'] / self.stats['total_processed']
                    if self.stats['total_processed'] > 0 else 0)

        improvement_rate = (self.stats['title_improvements'] / max(self.stats['topics_created'], 1)) * 100

        return {
            **self.stats,
            'average_processing_time_ms': avg_time,
            'error_rate': (self.stats['errors'] / max(self.stats['total_processed'], 1)) * 100,
            'title_improvement_rate': improvement_rate
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        health = {
            "status": "healthy",
            "components": {}
        }

        try:
            # Check embedding services
            model_info = self.embedding_service.get_model_info()
            health["components"]["embedding_service"] = {
                "status": "healthy" if model_info["model_loaded"] else "unhealthy",
                "details": model_info
            }
        except Exception as e:
            health["components"]["embedding_service"] = {
                "status": "unhealthy",
                "error": str(e)
            }

        try:
            # Check NER extractor
            ner_info = self.ner_extractor.get_model_info()
            health["components"]["ner_extractor"] = {
                "status": "healthy" if ner_info["model_loaded"] else "unhealthy",
                "details": ner_info
            }
        except Exception as e:
            health["components"]["ner_extractor"] = {
                "status": "unhealthy",
                "error": str(e)
            }

        try:
            # Check vector services
            vector_health = self.vector_service.health_check()
            health["components"]["vector_service"] = vector_health
        except Exception as e:
            health["components"]["vector_service"] = {
                "status": "unhealthy",
                "error": str(e)
            }

        try:
            # Check database connection
            db_healthy = self.db_manager.test_connection()
            health["components"]["database"] = {
                "status": "healthy" if db_healthy else "unhealthy"
            }
        except Exception as e:
            health["components"]["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }

        # Determine overall status
        component_statuses = [comp["status"] for comp in health["components"].values()]
        if all(status == "healthy" for status in component_statuses):
            health["status"] = "healthy"
        elif any(status == "healthy" for status in component_statuses):
            health["status"] = "degraded"
        else:
            health["status"] = "unhealthy"

        # Add system metrics
        health["metrics"] = self.get_statistics()
        health["timestamp"] = time.time()

        return health

    def reset_statistics(self):
        """Reset processing statistics"""
        self.stats = {
            'total_processed': 0,
            'total_processing_time': 0,
            'topics_created': 0,
            'topics_merged': 0,
            'errors': 0,
            'title_improvements': 0
        }
        logger.info("Statistics reset successfully")

    def get_topic_details(self, topic_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific topic"""
        try:
            with self.db_manager.get_session() as session:
                from database.models import Topic, TextEntry

                topic = session.query(Topic).filter(Topic.id == topic_id).first()
                if not topic:
                    return None

                # Get recent text entries
                text_entries = session.query(TextEntry).filter(
                    TextEntry.topic_id == topic_id
                ).order_by(TextEntry.created_at.desc()).limit(10).all()

                return {
                    "topic_id": topic.id,
                    "title": topic.title,
                    "description": topic.description,
                    "primary_language": topic.primary_language,
                    "content_count": topic.content_count,
                    "confidence_score": topic.confidence_score,
                    "created_at": topic.created_at.isoformat() if topic.created_at else None,
                    "updated_at": topic.updated_at.isoformat() if topic.updated_at else None,
                    "representative_text": topic.representative_text,
                    "recent_entries": [
                        {
                            "text": entry.original_text[:200] + "..." if len(
                                entry.original_text) > 200 else entry.original_text,
                            "created_at": entry.created_at.isoformat() if entry.created_at else None,
                            "sentiment": entry.sentiment_data,
                            "source_type": entry.source_type,
                            "extracted_entities": entry.extracted_entities
                        } for entry in text_entries
                    ]
                }
        except Exception as e:
            logger.error(f"Failed to get topic details for {topic_id}: {e}")
            return None

    def merge_topics(self, primary_topic_id: str, secondary_topic_id: str, user_id: Optional[str] = None) -> Dict[
        str, Any]:
        """Merge two topics together"""
        try:
            with self.db_manager.get_session() as session:
                from database.models import Topic, TextEntry

                # Get both topics
                primary_topic = session.query(Topic).filter(Topic.id == primary_topic_id).first()
                secondary_topic = session.query(Topic).filter(Topic.id == secondary_topic_id).first()

                if not primary_topic or not secondary_topic:
                    return {"success": False, "error": "One or both topics not found"}

                # Update all text entries from secondary to primary
                session.query(TextEntry).filter(
                    TextEntry.topic_id == secondary_topic_id
                ).update({"topic_id": primary_topic_id})

                # Update primary topic stats
                primary_topic.content_count += secondary_topic.content_count
                primary_topic.updated_at = func.now()

                # Delete secondary topic
                session.delete(secondary_topic)

                # Remove secondary topic from vector database
                self.vector_service.delete_topic(secondary_topic_id)

                self.stats['topics_merged'] += 1

                logger.info(f"Successfully merged topic {secondary_topic_id} into {primary_topic_id}")

                return {
                    "success": True,
                    "primary_topic_id": primary_topic_id,
                    "merged_topic_id": secondary_topic_id,
                    "new_content_count": primary_topic.content_count
                }

        except Exception as e:
            logger.error(f"Failed to merge topics {primary_topic_id} and {secondary_topic_id}: {e}")
            return {"success": False, "error": str(e)}

    def regenerate_topic_title(self, topic_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Regenerate topic title for an existing topic using latest NER analysis"""
        try:
            with self.db_manager.get_session() as session:
                from database.models import Topic, TextEntry

                # Get topic and recent entries
                topic = session.query(Topic).filter(Topic.id == topic_id).first()
                if not topic:
                    return {"success": False, "error": "Topic not found"}

                # Get recent text entries to analyze
                text_entries = session.query(TextEntry).filter(
                    TextEntry.topic_id == topic_id
                ).order_by(TextEntry.created_at.desc()).limit(5).all()

                if not text_entries:
                    return {"success": False, "error": "No text entries found for topic"}

                # Use the most recent entry for NER analysis
                latest_entry = text_entries[0]

                # Re-extract NER if needed
                if latest_entry.extracted_entities:
                    ner_data = latest_entry.extracted_entities
                else:
                    ner_data = self.ner_extractor.extract(latest_entry.original_text)

                # Enhance with categorization
                enhanced_ner_data = self._enhance_ner_with_categorization(ner_data, latest_entry.original_text)

                # Generate new title
                old_title = topic.title
                new_title = self._generate_intelligent_topic_title(
                    enhanced_ner_data,
                    topic.primary_language,
                    latest_entry.original_text
                )

                # Update topic
                topic.title = new_title
                topic.updated_at = func.now()

                # Update vector database metadata
                self.vector_service.update_topic_metadata(topic_id, {"topic_title": new_title})

                logger.info(f"Regenerated title for topic {topic_id}: '{old_title}' -> '{new_title}'")

                return {
                    "success": True,
                    "topic_id": topic_id,
                    "old_title": old_title,
                    "new_title": new_title
                }

        except Exception as e:
            logger.error(f"Failed to regenerate title for topic {topic_id}: {e}")
            return {"success": False, "error": str(e)}

    def bulk_regenerate_titles(self, limit: int = 50, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Bulk regenerate titles for topics with generic titles"""
        try:
            with self.db_manager.get_session() as session:
                from database.models import Topic

                # Find topics with generic titles
                generic_titles = ["सामान्य विषय", "General Topic", "सामान्य", "Local Incident", "स्थानीय घटना"]

                topics_to_update = session.query(Topic).filter(
                    Topic.title.in_(generic_titles)
                ).limit(limit).all()

                updated_count = 0
                results = []

                for topic in topics_to_update:
                    result = self.regenerate_topic_title(topic.id, user_id)
                    if result.get("success"):
                        updated_count += 1
                    results.append({
                        "topic_id": topic.id,
                        "result": result
                    })

                logger.info(f"Bulk title regeneration completed: {updated_count}/{len(topics_to_update)} updated")

                return {
                    "success": True,
                    "total_processed": len(topics_to_update),
                    "updated_count": updated_count,
                    "results": results
                }

        except Exception as e:
            logger.error(f"Bulk title regeneration failed: {e}")
            return {"success": False, "error": str(e)}

    def get_topics_by_category(self, incident_category: str = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Get topics filtered by incident category"""
        try:
            with self.db_manager.get_session() as session:
                from database.models import Topic, TextEntry

                query = session.query(Topic)

                if incident_category:
                    # Join with TextEntry to filter by incident category in extracted_entities
                    query = query.join(TextEntry).filter(
                        TextEntry.extracted_entities.contains({'incident_category': incident_category})
                    )

                topics = query.order_by(Topic.updated_at.desc()).limit(limit).all()

                return [
                    {
                        "topic_id": topic.id,
                        "title": topic.title,
                        "description": topic.description,
                        "content_count": topic.content_count,
                        "primary_language": topic.primary_language,
                        "created_at": topic.created_at.isoformat() if topic.created_at else None,
                        "updated_at": topic.updated_at.isoformat() if topic.updated_at else None
                    } for topic in topics
                ]

        except Exception as e:
            logger.error(f"Failed to get topics by category {incident_category}: {e}")
            return []

    def analyze_topic_trends(self, days: int = 30) -> Dict[str, Any]:
        """Analyze trending topics and incident categories"""
        try:
            from datetime import datetime, timedelta

            with self.db_manager.get_session() as session:
                from database.models import Topic, TextEntry

                # Calculate date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)

                # Get recent topics
                recent_topics = session.query(Topic).filter(
                    Topic.created_at >= start_date
                ).all()

                # Get recent text entries with extracted entities
                recent_entries = session.query(TextEntry).filter(
                    TextEntry.created_at >= start_date
                ).all()

                # Analyze incident categories
                category_counts = {}
                location_counts = {}

                for entry in recent_entries:
                    if entry.extracted_entities:
                        entities = entry.extracted_entities

                        # Count incident categories
                        category = entities.get('incident_category', 'general')
                        category_counts[category] = category_counts.get(category, 0) + 1

                        # Count locations
                        districts = entities.get('district_names', [])
                        for district in districts:
                            location_counts[district] = location_counts.get(district, 0) + 1

                # Sort by frequency
                top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                top_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:10]

                return {
                    "analysis_period_days": days,
                    "total_topics_created": len(recent_topics),
                    "total_text_entries": len(recent_entries),
                    "top_incident_categories": [
                        {"category": cat, "count": count} for cat, count in top_categories
                    ],
                    "top_locations": [
                        {"location": loc, "count": count} for loc, count in top_locations
                    ],
                    "daily_topic_creation": self._calculate_daily_trends(recent_topics, days),
                    "generated_at": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to analyze topic trends: {e}")
            return {"error": str(e)}

    def _calculate_daily_trends(self, topics: List, days: int) -> List[Dict[str, Any]]:
        """Calculate daily topic creation trends"""
        from datetime import datetime, timedelta
        from collections import defaultdict

        daily_counts = defaultdict(int)

        for topic in topics:
            if topic.created_at:
                date_key = topic.created_at.strftime('%Y-%m-%d')
                daily_counts[date_key] += 1

        # Generate complete date range
        end_date = datetime.now()
        trends = []

        for i in range(days):
            date = end_date - timedelta(days=i)
            date_key = date.strftime('%Y-%m-%d')
            trends.append({
                "date": date_key,
                "topics_created": daily_counts.get(date_key, 0)
            })

        return sorted(trends, key=lambda x: x['date'])
"""
Enhanced topic clustering service with intelligent topic title generation
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
from sqlalchemy.sql import func
import re

logger = logging.getLogger(__name__)


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

    def process_text(self,
                     text: str,
                     source_type: str = "unknown",
                     user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main function to process incoming text and assign to topics with enhanced title generation
        """
        start_time = time.time()

        try:
            # Step 1: Validate input
            if not text or not text.strip():
                raise ValueError("Empty text provided")

            if len(text.strip()) < 3:
                raise ValueError("Text must be at least 3 characters long")

            # Step 2: Language detection and preprocessing
            detected_language, lang_confidence = self.embedding_service.detect_language(text)
            processed_text = self.embedding_service.preprocess_text(text)

            logger.info(f"Processing text: language={detected_language}, confidence={lang_confidence:.2f}")

            # Step 3: Enhanced NER extraction with incident categorization
            ner_data = self.ner_extractor.extract(text)

            # Step 4: Enhance NER data with incident categorization
            enhanced_ner_data = self._enhance_ner_with_categorization(ner_data, text)

            # Step 5: Create enhanced text representation
            enhanced_text = self.embedding_service.create_enhanced_text(processed_text, enhanced_ner_data)

            # Step 6: Generate embeddings
            embeddings = self.embedding_service.generate_embeddings([enhanced_text])
            query_embedding = embeddings[0]

            # Step 7: Find similar topics with enhanced NER-based filtering
            similar_topics = self._find_similar_topics_with_ner(
                query_embedding, enhanced_ner_data, detected_language, source_type
            )

            # Step 8: Decide on topic assignment with intelligent title generation
            topic_result = self._assign_or_create_topic(
                text, processed_text, enhanced_text, query_embedding, enhanced_ner_data,
                similar_topics, detected_language, lang_confidence, source_type, user_id
            )

            processing_time = (time.time() - start_time) * 1000

            # Step 9: Build comprehensive result
            result = self._build_result(
                text, processed_text, enhanced_text, topic_result,
                enhanced_ner_data, detected_language, lang_confidence,
                source_type, processing_time
            )

            # Step 10: Update statistics and log
            self._update_statistics(processing_time, topic_result['action'])
            self._log_processing(result, processing_time, user_id)

            return result

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Text processing failed: {e}")
            self.stats['errors'] += 1
            return self._build_error_result(text, str(e), processing_time)

    async def process_text_batch(self,
                                 texts: List[str],
                                 source_type: str = "unknown",
                                 user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process multiple texts in batch for improved performance

        Args:
            texts: List of text strings to process
            source_type: Source type for all texts
            user_id: Optional user identifier

        Returns:
            List of processing results
        """
        start_time = time.time()

        try:
            # Step 1: Batch language detection and preprocessing
            batch_processed_data = []
            for text in texts:
                if not text or not text.strip() or len(text.strip()) < 3:
                    batch_processed_data.append({
                        "error": "Invalid text",
                        "original_text": text
                    })
                    continue

                detected_language, lang_confidence = self.embedding_service.detect_language(text)
                processed_text = self.embedding_service.preprocess_text(text)

                batch_processed_data.append({
                    "original_text": text,
                    "processed_text": processed_text,
                    "detected_language": detected_language,
                    "language_confidence": lang_confidence
                })

            # Step 2: Batch NER extraction
            valid_texts = [item for item in batch_processed_data if "error" not in item]
            if valid_texts:
                logger.info(f"Starting batch NER extraction for {len(valid_texts)} texts")

                # Use the batch extraction method from NER extractor
                batch_ner_results = self.ner_extractor.extract_batch(
                    [item["original_text"] for item in valid_texts]
                )

                # Add NER results to processed data
                for i, item in enumerate(valid_texts):
                    item["ner_data"] = batch_ner_results[i] if i < len(batch_ner_results) else {}
                    item["enhanced_ner_data"] = self._enhance_ner_with_categorization(
                        item["ner_data"], item["original_text"]
                    )

            # Step 3: Create enhanced texts and batch embedding generation
            enhanced_texts = []
            valid_items = []

            for item in batch_processed_data:
                if "error" not in item:
                    enhanced_text = self.embedding_service.create_enhanced_text(
                        item["processed_text"], item["enhanced_ner_data"]
                    )
                    item["enhanced_text"] = enhanced_text
                    enhanced_texts.append(enhanced_text)
                    valid_items.append(item)

            # Generate embeddings in batch
            if enhanced_texts:
                logger.info(f"Generating embeddings for {len(enhanced_texts)} texts")
                batch_embeddings = self.embedding_service.generate_embeddings(
                    enhanced_texts,
                    batch_size=len(enhanced_texts)  # Process all at once
                )

                # Add embeddings to items
                for i, item in enumerate(valid_items):
                    item["embedding"] = batch_embeddings[i] if i < len(batch_embeddings) else None

            # Step 4: Batch similarity search and topic assignment
            batch_results = []

            for item in batch_processed_data:
                if "error" in item:
                    batch_results.append(self._build_error_result(
                        item["original_text"], item["error"], 0
                    ))
                    continue

                try:
                    # Find similar topics
                    similar_topics = self._find_similar_topics_with_ner(
                        item["embedding"],
                        item["enhanced_ner_data"],
                        item["detected_language"],
                        source_type
                    )

                    # Assign or create topic
                    topic_result = self._assign_or_create_topic(
                        item["original_text"],
                        item["processed_text"],
                        item["enhanced_text"],
                        item["embedding"],
                        item["enhanced_ner_data"],
                        similar_topics,
                        item["detected_language"],
                        item["language_confidence"],
                        source_type,
                        user_id
                    )

                    # Build result
                    result = self._build_result(
                        item["original_text"],
                        item["processed_text"],
                        item["enhanced_text"],
                        topic_result,
                        item["enhanced_ner_data"],
                        item["detected_language"],
                        item["language_confidence"],
                        source_type,
                        0  # Individual processing time will be calculated differently
                    )

                    batch_results.append(result)

                except Exception as e:
                    logger.error(f"Error processing text in batch: {e}")
                    batch_results.append(self._build_error_result(
                        item["original_text"], str(e), 0
                    ))

            # Update batch processing time for all results
            total_processing_time = (time.time() - start_time) * 1000
            avg_processing_time = total_processing_time / len(batch_results) if batch_results else 0

            for result in batch_results:
                if "error" not in result:
                    result["processing_time_ms"] = int(avg_processing_time)

            # Update statistics
            self.stats['total_processed'] += len(texts)
            self.stats['total_processing_time'] += total_processing_time

            logger.info(f"Batch processing completed: {len(texts)} texts in {total_processing_time:.2f}ms")
            return batch_results

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Return error results for all texts
            return [self._build_error_result(text, str(e), 0) for text in texts]

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
        """Build comprehensive result object"""

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
            "extracted_entities": ner_data,
            "boost_reasons": topic_result.get("boost_reasons", []),
            "timestamp": time.time()
        }

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
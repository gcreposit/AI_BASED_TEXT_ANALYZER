"""
Main topic clustering services that combines BGE-M3 embeddings with Mistral NER
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
from sqlalchemy.sql import func

logger = logging.getLogger(__name__)


class TopicClusteringService:
    """
    Main services that orchestrates the multilingual topic clustering pipeline
    combining BGE-M3 embeddings with Mistral 24B NER extraction
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

        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'total_processing_time': 0,
            'topics_created': 0,
            'topics_merged': 0,
            'errors': 0
        }

    def process_text(self,
                     text: str,
                     source_type: str = "unknown",
                     user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main function to process incoming text and assign to topics

        Args:
            text: Input text to process
            source_type: Source type of the text
            user_id: Optional user identifier

        Returns:
            Dictionary containing processing results
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

            # Step 3: Parallel processing - NER extraction
            ner_data = self.ner_extractor.extract(text)

            # Step 4: Create enhanced text representation
            enhanced_text = self.embedding_service.create_enhanced_text(processed_text, ner_data)

            # Step 5: Generate embeddings
            embeddings = self.embedding_service.generate_embeddings([enhanced_text])
            query_embedding = embeddings[0]

            # Step 6: Find similar topics with NER-based filtering
            similar_topics = self._find_similar_topics_with_ner(
                query_embedding, ner_data, detected_language, source_type
            )

            # Step 7: Decide on topic assignment
            topic_result = self._assign_or_create_topic(
                text, processed_text, enhanced_text, query_embedding, ner_data,
                similar_topics, detected_language, lang_confidence, source_type, user_id
            )

            processing_time = (time.time() - start_time) * 1000

            # Step 8: Build comprehensive result
            result = self._build_result(
                text, processed_text, enhanced_text, topic_result,
                ner_data, detected_language, lang_confidence,
                source_type, processing_time
            )

            # Step 9: Update statistics and log
            self._update_statistics(processing_time, topic_result['action'])
            self._log_processing(result, processing_time, user_id)

            return result

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Text processing failed: {e}")
            self.stats['errors'] += 1
            return self._build_error_result(text, str(e), processing_time)

    def _find_similar_topics_with_ner(self,
                                      query_embedding: np.ndarray,
                                      ner_data: Dict[str, Any],
                                      language: str,
                                      source_type: str) -> List[Dict[str, Any]]:
        """Find similar topics using both semantic similarity and NER filtering"""

        # Get dynamic threshold based on content characteristics
        threshold = self.embedding_service.get_similarity_threshold("", language, source_type)

        # Prepare filters for vector search
        filters = {
            "primary_language": language
        }

        # Base semantic similarity search
        semantic_matches = self.vector_service.search_similar_topics(
            query_embedding=query_embedding,
            n_results=20,
            threshold=threshold * 0.7,  # Lower threshold for initial search
            filters=filters
        )

        # Enhance matches with NER-based scoring
        enhanced_matches = []
        for match in semantic_matches:
            enhanced_score = self._calculate_enhanced_similarity(match, ner_data)

            if enhanced_score >= threshold:
                match['enhanced_similarity'] = enhanced_score
                match['boost_reasons'] = self._get_boost_reasons(match, ner_data)
                enhanced_matches.append(match)

        # Sort by enhanced similarity
        enhanced_matches.sort(key=lambda x: x['enhanced_similarity'], reverse=True)

        logger.info(f"Found {len(enhanced_matches)} similar topics after NER enhancement")
        return enhanced_matches

    def _calculate_enhanced_similarity(self, match: Dict[str, Any], ner_data: Dict[str, Any]) -> float:
        """Calculate enhanced similarity score using NER data"""
        base_score = match['similarity']
        boost = 0.0

        match_meta_data = match.get('meta_data', {})

        # Geographic relevance boost
        if self._has_geographic_overlap(ner_data, match_meta_data):
            boost += 0.15

        # Incident type boost (highest priority)
        if self._has_incident_overlap(ner_data, match_meta_data):
            boost += 0.20

        # Entity overlap boost
        if self._has_entity_overlap(ner_data, match_meta_data):
            boost += 0.10

        # Sentiment alignment boost
        if self._has_sentiment_alignment(ner_data, match_meta_data):
            boost += 0.05

        # Source type alignment
        if self._has_source_alignment(ner_data, match_meta_data):
            boost += 0.03

        return min(1.0, base_score + boost)

    def _has_geographic_overlap(self, ner_data: Dict[str, Any], topic_meta_data: Dict[str, Any]) -> bool:
        """Check geographic relevance between NER data and topic"""

        # Parse stored JSON strings back to lists if needed
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

        topic_sentiment = topic_sentiment_data.get('label', 'neutral') if isinstance(topic_sentiment_data,
                                                                                     dict) else 'neutral'

        return ner_sentiment == topic_sentiment

    def _has_source_alignment(self, ner_data: Dict[str, Any], topic_meta_data: Dict[str, Any]) -> bool:
        """Check source type alignment"""
        # This would need to be passed in the meta_data, simplified for now
        return True

    def _get_boost_reasons(self, match: Dict[str, Any], ner_data: Dict[str, Any]) -> List[str]:
        """Get reasons for similarity boost"""
        reasons = []
        match_meta_data = match.get('meta_data', {})

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
        """Assign text to existing topic or create new one"""

        if similar_topics:
            # Assign to best matching topic
            best_match = similar_topics[0]
            topic_id = best_match['topic_id']

            # Update topic in database
            self._update_existing_topic(topic_id, original_text, ner_data, user_id)

            return {
                "action": "grouped",
                "topic_id": topic_id,
                "topic_title": best_match['meta_data'].get('topic_title', ''),
                "similarity_score": best_match['enhanced_similarity'],
                "confidence": self._calculate_confidence(best_match['enhanced_similarity']),
                "boost_reasons": best_match.get('boost_reasons', [])
            }
        else:
            # Create new topic
            topic_id = str(uuid.uuid4())
            topic_title = self._generate_topic_title(ner_data, language)

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

    def _generate_topic_title(self, ner_data: Dict[str, Any], language: str) -> str:
        """Generate concise topic title using NER data"""
        title_parts = []

        # Prioritize geographic information
        if ner_data.get('district_names'):
            title_parts.append(ner_data['district_names'][0])
        elif ner_data.get('location_names'):
            title_parts.append(ner_data['location_names'][0])

        # Add incident/event information
        if ner_data.get('incidents'):
            title_parts.append(ner_data['incidents'][0])
        elif ner_data.get('events'):
            title_parts.append(ner_data['events'][0])
        elif ner_data.get('organisation_names'):
            title_parts.append(ner_data['organisation_names'][0])

        # Fallback to contextual understanding
        if not title_parts and ner_data.get('contextual_understanding'):
            context = ner_data['contextual_understanding']
            # Extract first few meaningful words
            words = context.split()[:4]
            title_parts.extend(words)

        # Final fallback based on language
        if not title_parts:
            fallbacks = {
                'hindi': "सामान्य विषय",
                'hinglish': "General Topic",
                'english': "General Topic"
            }
            title_parts = [fallbacks.get(language, "General Topic")]

        # Join with appropriate separator and limit length
        separator = " — " if language in ['hindi', 'hinglish'] else " - "
        title = separator.join(title_parts[:3])  # Max 3 parts for readability

        # Ensure title doesn't exceed 10 words
        if len(title.split()) > 10:
            words = title.split()[:10]
            title = " ".join(words)

        return title

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

            # Add to vector database
            vector_meta_data = {
                "topic_title": title,
                "primary_language": language,
                "source_type": source_type,
                "content_count": 1,
                **{k: v for k, v in ner_data.items() if k != 'contextual_understanding'}
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
                    meta_data={
                        "language": result.get("detected_language"),
                        "action": result.get("action"),
                        "confidence": result.get("confidence"),
                        "topic_id": result.get("topic_id")
                    }
                )
                session.add(log_entry)
        except Exception as e:
            logger.error(f"Failed to log processing: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        avg_time = (self.stats['total_processing_time'] / self.stats['total_processed']
                    if self.stats['total_processed'] > 0 else 0)

        return {
            **self.stats,
            'average_processing_time_ms': avg_time,
            'error_rate': (self.stats['errors'] / max(self.stats['total_processed'], 1)) * 100
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
            'errors': 0
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
                            "source_type": entry.source_type
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
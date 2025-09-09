"""
Embedding services using BGE-M3 model for multilingual text processing
"""

import numpy as np
import re
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    BGE-M3 embedding services for multilingual text processing
    Optimized for Hindi, English, and Hinglish content
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self.model = None
        self.embedding_dimension = None
        self._load_model()

    def _load_model(self):
        """Load the BGE-M3 embedding model with optimization"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")

            self.model = SentenceTransformer(self.model_name)

            # Optimize for social media content
            self.model.max_seq_length = 512

            # Get embedding dimension
            test_embedding = self.model.encode(["test"], show_progress_bar=False)
            self.embedding_dimension = test_embedding.shape[1]

            logger.info(f"Successfully loaded embedding model: {self.model_name}")
            logger.info(f"Embedding dimension: {self.embedding_dimension}")

        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise

    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language of input text

        Args:
            text: Input text

        Returns:
            Tuple of (language, confidence)
        """
        try:
            detected_lang = detect(text)
            confidence = 0.9  # langdetect doesn't provide confidence

            # Map to our supported languages
            if detected_lang in ['hi', 'ur', 'pa']:  # Hindi, Urdu, Punjabi
                return 'hindi', confidence
            elif detected_lang == 'en':
                return 'english', confidence
            else:
                # Check for Hinglish patterns
                if self._is_hinglish(text):
                    return 'hinglish', 0.8
                return 'english', 0.6  # Default to English with lower confidence

        except LangDetectException:
            # If detection fails, check for scripts
            if self._has_devanagari(text):
                return 'hindi', 0.7
            elif self._is_hinglish(text):
                return 'hinglish', 0.6
            else:
                return 'english', 0.5

    def _is_hinglish(self, text: str) -> bool:
        """Detect if text is Hinglish (code-mixed Hindi-English)"""
        # Check for both Latin and Devanagari scripts
        has_latin = bool(re.search(r'[a-zA-Z]', text))
        has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))

        # Additional Hinglish patterns
        hinglish_patterns = [
            r'\b(hai|hain|tha|thi|ka|ki|ke|mein|ko|se|par|aur|ya|to|yeh|woh)\b',
            r'\b(kya|kaise|kahan|kab|kyun|kaun|kitna)\b',
            r'\b(gaya|gayi|hua|hui|kiya|kar|karne)\b'
        ]

        has_hinglish_words = any(re.search(pattern, text, re.IGNORECASE) for pattern in hinglish_patterns)

        return (has_latin and has_devanagari) or (has_latin and has_hinglish_words)

    def _has_devanagari(self, text: str) -> bool:
        """Check if text contains Devanagari script"""
        return bool(re.search(r'[\u0900-\u097F]', text))

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better embeddings

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        # Basic cleaning
        text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'#(\w+)', r'\1', text)  # Convert hashtags to words
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s\u0900-\u097F\u0600-\u06FF.,!?।॥]', ' ', text)  # Keep essential punctuation

        # Normalize Devanagari text
        if self._has_devanagari(text):
            text = self._normalize_devanagari(text)

        return text.strip()

    def _normalize_devanagari(self, text: str) -> str:
        """Normalize Devanagari text for better processing"""
        # Basic Devanagari normalization
        normalizations = {
            'क़': 'क', 'ख़': 'ख', 'ग़': 'ग', 'ज़': 'ज', 'ड़': 'ड', 'ढ़': 'ढ',
            'फ़': 'फ', 'य़': 'य', 'ऱ': 'र', 'ऴ': 'ल', 'ळ': 'ल', 'ॡ': 'ऋ'
        }

        for original, normalized in normalizations.items():
            text = text.replace(original, normalized)

        return text

    def create_enhanced_text(self, original_text: str, ner_data: Dict[str, Any]) -> str:
        """
        Create enhanced text representation using NER data for better embeddings

        Args:
            original_text: Original input text
            ner_data: NER extraction results

        Returns:
            Enhanced text with entity information
        """
        enhanced_parts = [original_text]

        # Add key entities as context
        if ner_data.get('incidents'):
            incidents = ', '.join(ner_data['incidents'][:3])  # Top 3 incidents
            enhanced_parts.append(f"घटनाएं: {incidents}")

        if ner_data.get('location_names'):
            locations = ', '.join(ner_data['location_names'][:2])  # Top 2 locations
            enhanced_parts.append(f"स्थान: {locations}")

        if ner_data.get('district_names'):
            districts = ', '.join(ner_data['district_names'][:2])  # Top 2 districts
            enhanced_parts.append(f"जिला: {districts}")

        if ner_data.get('organisation_names'):
            orgs = ', '.join(ner_data['organisation_names'][:2])  # Top 2 organizations
            enhanced_parts.append(f"संगठन: {orgs}")

        # Add contextual understanding if available
        if ner_data.get('contextual_understanding'):
            enhanced_parts.append(ner_data['contextual_understanding'])

        return " | ".join(enhanced_parts)

    def generate_embeddings(self,
                            texts: List[str],
                            instruction: str = "Represent this text for clustering: ",
                            batch_size: Optional[int] = None,
                            normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of input texts
            instruction: Instruction prefix for better performance
            batch_size: Batch size for processing (None for auto)
            normalize: Whether to normalize embeddings

        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])

        start_time = time.time()

        try:
            # Add instruction prefix for better performance
            if instruction:
                instructed_texts = [f"{instruction}{text}" for text in texts]
            else:
                instructed_texts = texts

            # Determine batch size
            if batch_size is None:
                batch_size = min(32, len(texts))  # Auto batch size

            # Generate embeddings
            embeddings = self.model.encode(
                instructed_texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=len(texts) > 10,
                convert_to_numpy=True
            )

            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Generated embeddings for {len(texts)} texts in {processing_time:.2f}ms")

            return embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    def generate_single_embedding(self,
                                  text: str,
                                  instruction: str = "Represent this text for clustering: ",
                                  normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text

        Args:
            text: Input text
            instruction: Instruction prefix
            normalize: Whether to normalize embedding

        Returns:
            Numpy array embedding
        """
        embeddings = self.generate_embeddings([text], instruction, batch_size=1, normalize=normalize)
        return embeddings[0] if len(embeddings) > 0 else np.array([])

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Ensure embeddings are normalized
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

            # Clamp to [0, 1] range
            return max(0.0, min(1.0, similarity))

        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0

    def get_similarity_threshold(self,
                                 content_type: str,
                                 language: str,
                                 source: str) -> float:
        """
        Get dynamic similarity threshold based on content characteristics

        Args:
            content_type: Type of content
            language: Detected language
            source: Source type

        Returns:
            Optimal similarity threshold
        """
        base_threshold = 0.80

        # Language-based adjustments
        language_adjustments = {
            'hinglish': -0.05,  # Lower threshold for code-mixed text
            'hindi': -0.02,  # Slightly lower for Hindi
            'english': 0.0  # No adjustment for English
        }

        # Source-based adjustments
        source_adjustments = {
            'social_media': -0.05,  # More lenient for social media
            'whatsapp': -0.08,  # More flexible for chat
            'news': +0.05,  # Stricter for formal content
            'blog': +0.02,  # Slightly stricter for blogs
            'other': 0.0
        }

        # Content length adjustments
        text_length = len(content_type) if content_type else 0
        length_adjustment = 0.0
        if text_length < 50:
            length_adjustment = -0.10  # More lenient for short text
        elif text_length > 500:
            length_adjustment = +0.05  # Stricter for long text

        # Calculate final threshold
        final_threshold = base_threshold
        final_threshold += language_adjustments.get(language, 0.0)
        final_threshold += source_adjustments.get(source, 0.0)
        final_threshold += length_adjustment

        # Clamp to reasonable range
        return max(0.6, min(0.95, final_threshold))

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "max_sequence_length": self.model.max_seq_length if self.model else None,
            "model_loaded": self.model is not None,
            "supported_languages": ["hindi", "english", "hinglish"],
            "optimization": "social_media_content"
        }

    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """
        Validate embedding array

        Args:
            embedding: Embedding to validate

        Returns:
            True if valid, False otherwise
        """
        if embedding is None or embedding.size == 0:
            return False

        if len(embedding.shape) != 1:
            return False

        if self.embedding_dimension and embedding.shape[0] != self.embedding_dimension:
            return False

        # Check for NaN or infinite values
        if not np.isfinite(embedding).all():
            return False

        return True
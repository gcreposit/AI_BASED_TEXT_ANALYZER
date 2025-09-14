"""
Enhanced Embedding services supporting BGE-M3, Jina v3, and Jina v4 models
with full backward compatibility for existing codebase and proper multi-GPU support
"""

import numpy as np
import re
import time
import logging
import os
import torch
from typing import List, Dict, Any, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Enhanced embedding service with multi-model support and proper GPU distribution
    while maintaining full backward compatibility with existing code
    """

    # Model configuration mapping
    MODEL_CONFIGS = {
        'BAAI/bge-m3': {
            'type': 'bge-m3',
            'supports_tasks': False,
            'trust_remote_code': True,
            'default_instruction': "Represent this text for clustering: ",
            'max_seq_length': 8192
        },
        'jinaai/jina-embeddings-v3': {
            'type': 'jina-v3',
            'supports_tasks': False,
            'trust_remote_code': False,
            'default_instruction': "",
            'max_seq_length': 8192
        },
        'jinaai/jina-embeddings-v4': {
            'type': 'jina-v4',
            'supports_tasks': True,
            'available_tasks': ['retrieval', 'text-matching', 'code'],  # Updated with actual v4 tasks
            'trust_remote_code': False,
            'default_instruction': "",
            'max_seq_length': 8192
        }
    }

    def __init__(self, model_name: str = None):
        """
        Initialize embedding service with backward compatibility

        Args:
            model_name: Model name (maintains original interface)
        """
        # Handle environment variable or parameter
        self.model_name = model_name or os.getenv('EMBEDDING_MODEL_NAME', 'BAAI/bge-m3')

        # Initialize model storage
        self.model = None  # Main model (for backward compatibility)
        self.models = {}   # Task-specific models (for v4)
        self.embedding_dimension = None

        # Configuration
        self.current_config = self._get_model_config()

        # Task management for v4
        self.load_all_tasks = os.getenv('EMBEDDING_LOAD_ALL_TASKS', 'false').lower() == 'true'
        self.primary_tasks = [task.strip() for task in os.getenv('EMBEDDING_PRIMARY_TASKS', 'retrieval,text-matching').split(',')]

        # Load the model(s)
        self._load_model()

    def _get_model_config(self) -> Dict[str, Any]:
        """Get configuration for the current model"""
        for model_pattern, config in self.MODEL_CONFIGS.items():
            if model_pattern in self.model_name:
                return config

        # Fallback - treat as BGE-like model
        return {
            'type': 'custom',
            'supports_tasks': False,
            'trust_remote_code': True,
            'default_instruction': "Represent this text for clustering: ",
            'max_seq_length': 8192
        }

    def _load_model(self):
        """Load the embedding model based on type with comprehensive fallback handling"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")

            if self.current_config.get('supports_tasks', False):
                # Try to load Jina v4 with task support
                try:
                    self._load_task_specific_models()
                except Exception as v4_error:
                    logger.error(f"Jina v4 task loading failed: {v4_error}")
                    logger.info("Attempting to fall back to Jina v3...")

                    # Try fallback to Jina v3
                    try:
                        self.model_name = "jinaai/jina-embeddings-v3"
                        self.current_config = self.MODEL_CONFIGS.get("jinaai/jina-embeddings-v3", {
                            'type': 'jina-v3',
                            'supports_tasks': False,
                            'trust_remote_code': False,
                            'default_instruction': "",
                            'max_seq_length': 8192
                        })
                        self._load_single_model()
                        logger.info("✅ Successfully fell back to Jina v3")
                    except Exception as v3_error:
                        logger.error(f"Jina v3 fallback failed: {v3_error}")
                        logger.info("Attempting final fallback to BGE-M3...")

                        # Final fallback to BGE-M3
                        try:
                            self.model_name = "BAAI/bge-m3"
                            self.current_config = self.MODEL_CONFIGS.get("BAAI/bge-m3", {
                                'type': 'bge-m3',
                                'supports_tasks': False,
                                'trust_remote_code': True,
                                'default_instruction': "Represent this text for clustering: ",
                                'max_seq_length': 8192
                            })
                            self._load_single_model()
                            logger.info("✅ Successfully fell back to BGE-M3")
                        except Exception as bge_error:
                            logger.error(f"All fallbacks failed. BGE-M3 error: {bge_error}")
                            raise Exception(f"All models failed to load. Last error: {bge_error}")
            else:
                self._load_single_model()

        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise

    def _load_single_model(self):
        """Load single model (BGE-M3, Jina v3, or other)"""
        try:
            # Prepare loading parameters
            load_kwargs = {}
            if self.current_config.get('trust_remote_code'):
                load_kwargs['trust_remote_code'] = True

            # Load the model
            self.model = SentenceTransformer(self.model_name,
                trust_remote_code=True)

            # Set max sequence length
            max_seq_length = self.current_config.get('max_seq_length', 8192)
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = max_seq_length

            # Get embedding dimension
            test_embedding = self.model.encode(["test"], show_progress_bar=False)
            self.embedding_dimension = test_embedding.shape[1]

            logger.info(f"Successfully loaded embedding model: {self.model_name}")
            logger.info(f"Embedding dimension: {self.embedding_dimension}")

        except Exception as e:
            logger.error(f"Failed to load single model: {e}")
            raise

    def _load_task_specific_models(self):
        """Load task-specific models for Jina v4 with multi-GPU distribution and fallback handling"""
        try:
            available_tasks = self.current_config.get('available_tasks', [])

            # Determine which tasks to load
            if self.load_all_tasks:
                tasks_to_load = available_tasks
            else:
                # Filter tasks to only include available ones
                tasks_to_load = [task.strip() for task in self.primary_tasks if task.strip() in available_tasks]

            logger.info(f"Loading Jina v4 tasks: {tasks_to_load}")
            logger.info(f"Available tasks in model: {available_tasks}")

            # Get number of available GPUs
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            logger.info(f"Available GPUs: {num_gpus}")

            # First try fallback mode (no task specification) - most reliable
            fallback_success = False
            try:
                logger.info("Attempting fallback: loading Jina v4 without task specification...")
                fallback_model = SentenceTransformer(self.model_name, trust_remote_code=True)

                # Test the fallback model
                test_embedding = fallback_model.encode(["test"], show_progress_bar=False)

                # If successful, use this as the main model
                self.model = fallback_model
                self.embedding_dimension = test_embedding.shape[1]

                # For backward compatibility, assign to all requested tasks
                for task in tasks_to_load:
                    self.models[task] = fallback_model

                fallback_success = True
                logger.warning(f"⚠️  Using Jina v4 in fallback mode (no task specification)")
                logger.info(f"✅ Fallback model loaded successfully with dimension {self.embedding_dimension}")

            except Exception as fallback_error:
                logger.error(f"Fallback loading failed: {fallback_error}")

            # If fallback didn't work, try sequential task loading with GPU distribution
            if not fallback_success and tasks_to_load:
                for idx, task in enumerate(tasks_to_load):
                    try:
                        # Determine target GPU (round-robin if multiple GPUs available)
                        target_gpu = idx % max(1, num_gpus) if num_gpus > 0 else 0

                        logger.info(f"Loading model for task: {task}")
                        if num_gpus > 0:
                            logger.info(f"Target GPU for '{task}': cuda:{target_gpu}")

                        # Set specific GPU before loading (if multiple GPUs)
                        if num_gpus > 1:
                            torch.cuda.set_device(target_gpu)

                        task_model = SentenceTransformer(
                            self.model_name,
                            trust_remote_code=True,
                            model_kwargs={'default_task': task}
                        )

                        # Move to specific GPU if available
                        if num_gpus > 0:
                            device = torch.device(f'cuda:{target_gpu}')
                            task_model = task_model.to(device)

                        # Test the model
                        test_embedding = task_model.encode(["test"], show_progress_bar=False)

                        # Store the model
                        self.models[task] = task_model

                        # Set embedding dimension from first successful model
                        if self.embedding_dimension is None:
                            self.embedding_dimension = test_embedding.shape[1]

                        # Set the main model to the first loaded task for backward compatibility
                        if self.model is None:
                            self.model = task_model

                        logger.info(f"✅ Loaded task '{task}' successfully on GPU {target_gpu if num_gpus > 0 else 'CPU'}")

                    except Exception as e:
                        logger.error(f"❌ Failed to load task '{task}': {e}")
                        continue

                # Reset to GPU 0 after loading
                if num_gpus > 0:
                    torch.cuda.set_device(0)

            # Check if we have any working model
            if not self.model and not self.models:
                logger.error("All Jina v4 loading attempts failed.")
                raise ValueError("No task models loaded successfully")

            if fallback_success:
                logger.info(f"✅ Using Jina v4 in unified mode for tasks: {list(self.models.keys())}")
            else:
                logger.info(f"✅ Loaded {len(self.models)} task-specific models: {list(self.models.keys())}")
                if num_gpus > 1:
                    gpu_distribution = {}
                    for task, model in self.models.items():
                        try:
                            device = next(model.parameters()).device
                            gpu_distribution[task] = str(device)
                        except:
                            gpu_distribution[task] = "unknown"
                    logger.info(f"GPU distribution: {gpu_distribution}")

        except Exception as e:
            logger.error(f"Failed to load task-specific models: {e}")
            raise

    def get_model_for_task(self, task: str = None) -> SentenceTransformer:
        """Get appropriate model for task (internal method)"""
        if not self.current_config.get('supports_tasks', False):
            return self.model

        if task and task in self.models:
            return self.models[task]

        # Try to load task on demand
        if task and task in self.current_config.get('available_tasks', []):
            if self._load_task_on_demand(task):
                return self.models[task]

        # Fallback to main model or first available
        if self.model:
            return self.model
        elif self.models:
            return list(self.models.values())[0]

        raise ValueError(f"No model available for task: {task}")

    def _load_task_on_demand(self, task: str) -> bool:
        """Load a task on demand for v4"""
        if task in self.models:
            return True

        try:
            logger.info(f"Loading task '{task}' on-demand...")

            task_model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
                model_kwargs={'default_task': task}
            )

            # Test the model
            test_embedding = task_model.encode(["test"], show_progress_bar=False)

            self.models[task] = task_model
            logger.info(f"✅ Task '{task}' loaded on-demand")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load task '{task}' on-demand: {e}")
            return False

    # ==================== ORIGINAL INTERFACE METHODS ====================
    # These maintain exact compatibility with existing code

    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language of input text
        (Original method - unchanged)
        """
        try:
            detected_lang = detect(text)
            confidence = 0.9

            if detected_lang in ['hi', 'ur', 'pa']:
                return 'hindi', confidence
            elif detected_lang == 'en':
                return 'english', confidence
            else:
                if self._is_hinglish(text):
                    return 'hinglish', 0.8
                return 'english', 0.6

        except LangDetectException:
            if self._has_devanagari(text):
                return 'hindi', 0.7
            elif self._is_hinglish(text):
                return 'hinglish', 0.6
            else:
                return 'english', 0.5

    def _is_hinglish(self, text: str) -> bool:
        """Detect if text is Hinglish (Original method)"""
        has_latin = bool(re.search(r'[a-zA-Z]', text))
        has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))

        hinglish_patterns = [
            r'\b(hai|hain|tha|thi|ka|ki|ke|mein|ko|se|par|aur|ya|to|yeh|woh)\b',
            r'\b(kya|kaise|kahan|kab|kyun|kaun|kitna)\b',
            r'\b(gaya|gayi|hua|hui|kiya|kar|karne)\b'
        ]

        has_hinglish_words = any(re.search(pattern, text, re.IGNORECASE) for pattern in hinglish_patterns)

        return (has_latin and has_devanagari) or (has_latin and has_hinglish_words)

    def _has_devanagari(self, text: str) -> bool:
        """Check if text contains Devanagari script (Original method)"""
        return bool(re.search(r'[\u0900-\u097F]', text))

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better embeddings
        (Original method - unchanged)
        """
        if not text:
            return ""

        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\u0900-\u097F\u0600-\u06FF.,!?।॥]', ' ', text)

        if self._has_devanagari(text):
            text = self._normalize_devanagari(text)

        return text.strip()

    def _normalize_devanagari(self, text: str) -> str:
        """Normalize Devanagari text (Original method)"""
        normalizations = {
            'क़': 'क', 'ख़': 'ख', 'ग़': 'ग', 'ज़': 'ज', 'ड़': 'ड', 'ढ़': 'ढ',
            'फ़': 'फ', 'य़': 'य', 'ऱ': 'र', 'ऴ': 'ल', 'ळ': 'ल', 'ॡ': 'ऋ'
        }

        for original, normalized in normalizations.items():
            text = text.replace(original, normalized)

        return text

    def create_enhanced_text(self, original_text: str, ner_data: Dict[str, Any]) -> str:
        """
        Create enhanced text representation (Original method - unchanged)
        """
        enhanced_parts = [original_text]

        if ner_data.get('incidents'):
            incidents = ', '.join(ner_data['incidents'][:3])
            enhanced_parts.append(f"घटनाएं: {incidents}")

        if ner_data.get('location_names'):
            locations = ', '.join(ner_data['location_names'][:2])
            enhanced_parts.append(f"स्थान: {locations}")

        if ner_data.get('district_names'):
            districts = ', '.join(ner_data['district_names'][:2])
            enhanced_parts.append(f"जिला: {districts}")

        if ner_data.get('organisation_names'):
            orgs = ', '.join(ner_data['organisation_names'][:2])
            enhanced_parts.append(f"संगठन: {orgs}")

        if ner_data.get('contextual_understanding'):
            enhanced_parts.append(ner_data['contextual_understanding'])

        return " | ".join(enhanced_parts)

    def generate_embeddings(self,
                            texts: List[str],
                            instruction: str = "Represent this text for clustering: ",
                            batch_size: Optional[int] = None,
                            normalize: bool = True,
                            task: str = None) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        Enhanced with optional task parameter while maintaining backward compatibility
        """
        if not texts:
            return np.array([])

        start_time = time.time()

        try:
            # Get appropriate model
            model = self.get_model_for_task(task)

            # Handle instruction based on model type
            if instruction is None:
                instruction = self.current_config.get('default_instruction', '')

            # Prepare texts with instruction (only for BGE models)
            if instruction and self.current_config.get('type') == 'bge-m3':
                instructed_texts = [f"{instruction}{text}" for text in texts]
            else:
                instructed_texts = texts

            # Determine batch size
            if batch_size is None:
                batch_size = min(32, len(texts))

            # Prepare encoding parameters
            encode_params = {
                'sentences': instructed_texts,
                'batch_size': batch_size,
                'show_progress_bar': len(texts) > 10,
                'convert_to_numpy': True
            }

            # Add normalize_embeddings if supported
            try:
                # Test if the model supports normalize_embeddings parameter
                embeddings = model.encode(**encode_params, normalize_embeddings=normalize)
            except TypeError:
                # Fallback for models that don't support normalize_embeddings
                embeddings = model.encode(
                    instructed_texts,
                    batch_size=batch_size,
                    show_progress_bar=len(texts) > 10,
                    convert_to_numpy=True
                )

                # Manual normalization if needed
                if normalize:
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    norms[norms == 0] = 1
                    embeddings = embeddings / norms

            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Generated embeddings for {len(texts)} texts in {processing_time:.2f}ms")

            return embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    def generate_single_embedding(self,
                                  text: str,
                                  instruction: str = "Represent this text for clustering: ",
                                  normalize: bool = True,
                                  task: str = None) -> np.ndarray:
        """
        Generate embedding for a single text
        Enhanced with optional task parameter
        """
        embeddings = self.generate_embeddings([text], instruction, batch_size=1, normalize=normalize, task=task)
        return embeddings[0] if len(embeddings) > 0 else np.array([])

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        (Original method - unchanged)
        """
        try:
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return max(0.0, min(1.0, similarity))

        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0

    def get_similarity_threshold(self,
                                 content_type: str,
                                 language: str,
                                 source: str) -> float:
        """
        Get dynamic similarity threshold (Original method - unchanged)
        """
        base_threshold = 0.80

        language_adjustments = {
            'hinglish': -0.05,
            'hindi': -0.02,
            'english': 0.0
        }

        source_adjustments = {
            'social_media': -0.05,
            'whatsapp': -0.08,
            'news': +0.05,
            'blog': +0.02,
            'other': 0.0
        }

        text_length = len(content_type) if content_type else 0
        length_adjustment = 0.0
        if text_length < 50:
            length_adjustment = -0.10
        elif text_length > 500:
            length_adjustment = +0.05

        final_threshold = base_threshold
        final_threshold += language_adjustments.get(language, 0.0)
        final_threshold += source_adjustments.get(source, 0.0)
        final_threshold += length_adjustment

        return max(0.6, min(0.95, final_threshold))

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model(s)"""
        base_info = {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "max_sequence_length": self.model.max_seq_length if self.model and hasattr(self.model, 'max_seq_length') else None,
            "model_loaded": self.model is not None,
            "supported_languages": ["hindi", "english", "hinglish"],
            "optimization": "social_media_content",
            "model_type": self.current_config.get('type', 'unknown')
        }

        # Add task-specific info for v4
        if self.current_config.get('supports_tasks', False):
            base_info.update({
                "supports_tasks": True,
                "available_tasks": self.current_config.get('available_tasks', []),
                "loaded_tasks": list(self.models.keys()),
                "primary_tasks": self.primary_tasks,
                "load_all_tasks": self.load_all_tasks
            })
        else:
            base_info["supports_tasks"] = False

        return base_info

    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """
        Validate embedding array (Original method - unchanged)
        """
        if embedding is None or embedding.size == 0:
            return False

        if len(embedding.shape) != 1:
            return False

        if self.embedding_dimension and embedding.shape[0] != self.embedding_dimension:
            return False

        if not np.isfinite(embedding).all():
            return False

        return True

    # ==================== NEW ENHANCED METHODS ====================
    # These are new methods that leverage v4 capabilities

    def generate_embeddings_for_retrieval(self,
                                        texts: List[str],
                                        instruction: str = None,
                                        batch_size: Optional[int] = None,
                                        normalize: bool = True) -> np.ndarray:
        """Generate embeddings optimized for search and retrieval"""
        return self.generate_embeddings(texts, instruction, batch_size, normalize, task='retrieval')

    def generate_embeddings_for_clustering(self,
                                         texts: List[str],
                                         instruction: str = None,
                                         batch_size: Optional[int] = None,
                                         normalize: bool = True) -> np.ndarray:
        """Generate embeddings optimized for clustering and topic modeling"""
        # Note: 'clustering' task doesn't exist in Jina v4, falls back to general model
        return self.generate_embeddings(texts, instruction, batch_size, normalize, task='clustering')

    def generate_embeddings_for_classification(self,
                                             texts: List[str],
                                             instruction: str = None,
                                             batch_size: Optional[int] = None,
                                             normalize: bool = True) -> np.ndarray:
        """Generate embeddings optimized for text classification"""
        # Note: 'classification' task doesn't exist in Jina v4, falls back to general model
        return self.generate_embeddings(texts, instruction, batch_size, normalize, task='classification')

    def generate_embeddings_for_similarity(self,
                                         texts: List[str],
                                         instruction: str = None,
                                         batch_size: Optional[int] = None,
                                         normalize: bool = True) -> np.ndarray:
        """Generate embeddings optimized for semantic similarity matching"""
        return self.generate_embeddings(texts, instruction, batch_size, normalize, task='text-matching')

    def find_similar_texts(self,
                          query: str,
                          candidates: List[str],
                          task: str = 'text-matching',
                          top_k: int = 5,
                          threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find most similar texts using specified task"""
        if not candidates:
            return []

        query_embedding = self.generate_single_embedding(query, task=task)
        candidate_embeddings = self.generate_embeddings(candidates, task=task)

        similarities = []
        for i, candidate_emb in enumerate(candidate_embeddings):
            similarity = self.calculate_similarity(query_embedding, candidate_emb)
            if similarity >= threshold:
                similarities.append({
                    'text': candidates[i],
                    'similarity': similarity,
                    'index': i,
                    'task_used': task
                })

        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        memory_info = {
            "model_type": self.current_config.get('type', 'unknown'),
            "model_name": self.model_name
        }

        if self.current_config.get('supports_tasks', False):
            memory_info.update({
                "loaded_tasks": list(self.models.keys()),
                "estimated_memory_per_task": "~8GB",
                "total_estimated_memory": f"~{len(self.models) * 8}GB"
            })
        else:
            model_type = self.current_config.get('type', 'unknown')
            if model_type == 'bge-m3':
                memory_info["estimated_memory"] = "~2GB"
            elif model_type == 'jina-v3':
                memory_info["estimated_memory"] = "~1.5GB"
            else:
                memory_info["estimated_memory"] = "unknown"

        return memory_info

    def unload_task(self, task: str):
        """Unload a specific task to free memory (for v4 only)"""
        if task in self.models:
            del self.models[task]
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded task: {task}")

    def cleanup(self):
        """Clean up all models and free memory"""
        logger.info("Cleaning up embedding service...")

        if hasattr(self, 'models'):
            self.models.clear()

        self.model = None

        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Cleanup completed")
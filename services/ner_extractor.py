import os
import re
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Required imports for vLLM, MLX and Hugging Face models
try:
    from vllm import LLM, SamplingParams
    from vllm.transformers_utils.tokenizer import get_tokenizer

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("vLLM not available. Install with: pip install vllm")

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load as mlx_load, generate as mlx_generate

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("MLX not available. Install with: pip install mlx mlx-lm")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers torch")

try:
    from huggingface_hub import snapshot_download

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Hugging Face Hub not available. Install with: pip install huggingface-hub")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MistralNERExtractor:
    """
    NER extractor using Mistral 24B model for Hindi/English/Hinglish text processing
    Optimized for efficient model loading and reuse with vLLM, MLX, and Transformers support
    """

    def __init__(self, model_id: str = None, model_path: Optional[str] = None, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/mistral_ner")
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        self._loading_method = None  # Track which loading method worked: 'vllm', 'mlx', 'transformers'

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        if model_path:
            # Direct model path provided
            self.model_path = model_path
            self.model_id = model_path.split('/')[-1]
            logger.info(f"Using provided model path: {self.model_path}")

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        else:
            # Use model_id with caching logic
            self.model_id = model_id or "mlx-community/Dolphin-Mistral-24B-Venice-Edition-4bit"
            self.model_path = self._get_or_download_model()

        # Domain-specific vocabularies and patterns
        self.religions = [
            "हिन्दू", "हिंदू", "मुस्लिम", "इस्लाम", "ईसाई", "क्रिश्चियन", "सिख", "बौद्ध", "जैन",
            "Hindu", "Muslim", "Islam", "Christian", "Sikh", "Buddhist", "Jain"
        ]

        self.castes = [
            "SC", "ST", "OBC", "ब्राह्मण", "ठाकुर", "राजपूत", "यादव", "दलित", "कुर्मी"
        ]

        self.up_districts = [
            "आगरा", "अलीगढ़", "अयोध्या", "बांदा", "बरेली", "लखनऊ", "वाराणसी",
            "गोरखपुर", "कानपुर", "मेरठ", "प्रयागराज", "फैजाबाद", "गाजियाबाद"
        ]

        self.thana_patterns = [
            r"(?:थाना|कोतवाली|Kotwali|Thana|PS)\s+([^\s,।:-]+(?:\s+[^\s,।:-]+)?)"
        ]

        # JSON schema for consistent output
        self.json_schema = {
            "person_names": [],
            "organisation_names": [],
            "location_names": [],
            "district_names": [],
            "thana_names": [],
            "incidents": [],
            "caste_names": [],
            "religion_names": [],
            "hashtags": [],
            "mention_ids": [],
            "events": [],
            "sentiment": {"label": "neutral", "confidence": 0.5},
            "contextual_understanding": ""
        }

        # Load model during initialization for efficiency
        self._load_model()

    def _get_or_download_model(self) -> str:
        """Check if model exists locally in cache."""
        logger.info(f"Checking local cache for model {self.model_id} ...")

        try:
            local_path = snapshot_download(
                repo_id=self.model_id,
                cache_dir=self.cache_dir,
                local_files_only=False
            )
            logger.info(f"Model available at: {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Failed to download or find model: {e}")
            raise

    def _load_model(self, max_retries: int = 3, retry_wait: int = 5) -> bool:
        """Load the Mistral model with vLLM, MLX, and Transformers support."""
        if self._model_loaded and self.model is not None and self.tokenizer is not None:
            logger.info("Model already loaded, skipping reload")
            return True

        retry_count = 0
        while retry_count < max_retries:
            try:
                local_path = self.model_path
                logger.info(f"Loading model from: {local_path}")

                # Try vLLM loading first (fastest inference)
                if VLLM_AVAILABLE:
                    try:
                        logger.info("Attempting vLLM model loading...")

                        # Initialize vLLM model with optimized settings
                        self.model = LLM(
                            model=local_path,
                            tokenizer=local_path,
                            trust_remote_code=True,
                            dtype="auto",  # Let vLLM choose optimal dtype
                            gpu_memory_utilization=0.85,  # Use 85% of GPU memory
                            max_model_len=4096,  # Reasonable context length
                            tensor_parallel_size=1,  # Single GPU
                            disable_log_stats=True,  # Reduce logging noise
                            enforce_eager=False,  # Use CUDA graphs for better performance
                        )

                        # Get tokenizer for vLLM
                        self.tokenizer = get_tokenizer(
                            tokenizer_name=local_path,
                            trust_remote_code=True
                        )

                        # Set pad token if not available
                        if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is None:
                            self.tokenizer.pad_token = self.tokenizer.eos_token

                        self._loading_method = 'vllm'
                        logger.info(f"Successfully loaded vLLM model: {self.model_id}")
                        self._model_loaded = True
                        return True

                    except Exception as e:
                        logger.warning(f"vLLM loading failed: {e}")
                        self._loading_method = None

                # Try MLX loading second (for Apple Silicon)
                if MLX_AVAILABLE:
                    try:
                        logger.info("Attempting MLX model loading...")
                        self.model, self.tokenizer = mlx_load(local_path)
                        self._loading_method = 'mlx'
                        logger.info(f"Successfully loaded MLX model: {self.model_id}")
                        self._model_loaded = True
                        return True
                    except Exception as e:
                        logger.warning(f"MLX loading failed: {e}")
                        self._loading_method = None

                # Fallback to standard transformers loading
                if TRANSFORMERS_AVAILABLE:
                    try:
                        logger.info("Attempting Transformers model loading...")
                        self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)

                        # Set pad token if not available
                        if self.tokenizer.pad_token is None:
                            self.tokenizer.pad_token = self.tokenizer.eos_token

                        self.model = AutoModelForCausalLM.from_pretrained(
                            local_path,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None,
                            trust_remote_code=True
                        )

                        self._loading_method = 'transformers'
                        logger.info(f"Successfully loaded Transformers model: {self.model_id}")
                        self._model_loaded = True
                        return True
                    except Exception as e:
                        logger.warning(f"Transformers loading failed: {e}")
                        self._loading_method = None

                # If we reach here, no loading method worked
                raise Exception("No compatible model loading framework available or all methods failed")

            except Exception as e:
                retry_count += 1
                logger.warning(f"Failed to load model (attempt {retry_count}): {e}")

                if retry_count >= max_retries:
                    logger.error(f"Failed to load model after {max_retries} attempts: {e}")
                    self._model_loaded = False
                    self._loading_method = None
                    return False

                logger.info(f"Retrying in {retry_wait} seconds...")
                time.sleep(retry_wait)

        return False

    def _dedupe(self, seq: List[str]) -> List[str]:
        """Remove duplicates while preserving order"""
        seen = set()
        result = []
        for item in seq:
            if not item:
                continue
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                result.append(cleaned)
                seen.add(cleaned)
        return result

    def _find_hashtags(self, text: str) -> List[str]:
        """Extract hashtags including Devanagari script"""
        pattern = re.compile(r"#([A-Za-z0-9_\.\u0900-\u097F]+)")
        hashtags = [match.group(0) for match in pattern.finditer(text)]
        return self._dedupe(hashtags)

    def _find_mentions(self, text: str) -> List[str]:
        """Extract @mentions"""
        pattern = re.compile(r"@([A-Za-z0-9_\.]+)")
        mentions = [match.group(0) for match in pattern.finditer(text)]
        return self._dedupe(mentions)

    def _find_districts(self, text: str) -> List[str]:
        """Find district names in text"""
        found_districts = []
        text_lower = text.lower()

        for district in self.up_districts:
            if district in text or district.lower() in text_lower:
                found_districts.append(district)

        return self._dedupe(found_districts)

    def _find_thana(self, text: str) -> List[str]:
        """Extract police station (thana) names using patterns"""
        found_thanas = []

        for pattern in self.thana_patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                name = match.group(1).strip()
                # Clean up punctuation
                name = name.rstrip(".,:;|/\\-—–")
                if name:
                    found_thanas.append(name)

        return self._dedupe(found_thanas)

    def _find_keywords(self, text: str, vocabulary: List[str]) -> List[str]:
        """Find keywords from vocabulary in text"""
        found_keywords = []
        text_lower = text.lower()

        for keyword in vocabulary:
            if keyword in text or keyword.lower() in text_lower:
                found_keywords.append(keyword)

        return self._dedupe(found_keywords)

    def _build_instruction_prompt(self, text: str) -> str:
        """Build instruction prompt that adapts to model type"""

        # Check if model is instruction-tuned based on model name
        is_instruct_model = self._is_instruction_tuned_model()

        instructions = f"""आप एक डेटा-एक्सट्रैक्शन सहायक हैं। नीचे दिए गए हिन्दी टेक्स्ट से पार्स कर के केवल वैध JSON लौटाइए। 
    JSON की कुंजियाँ और फ़ॉर्मेट बिल्कुल इस स्कीमा जैसा होना चाहिए:
    {json.dumps(self.json_schema, ensure_ascii=False, indent=2)}

    नियम:
    - सभी सूचियाँ unique और साफ़ strings हों।
    - "person_names" में व्यक्ति (@handles नहीं), "organisation_names" में संगठन/विभाग/कंपनी,
      "location_names" में शहर/कस्बा/इलाका/राज्य (जिले/थाने अलग keys में हैं) डालें।
    - "incidents" और "events" में 3-7 शब्दों के संक्षिप्त वाक्यांश रखें (जैसे "मारपीट", "सड़क दुर्घटना", "प्रदर्शन", "एफआईआर दर्ज")।
    - "sentiment" में label = positive|negative|neutral और confidence 0..1 दें।
    - "contextual_understanding" में 1-3 वाक्य का सार दें (हिन्दी में)।
    - कोड-फेंस (```), अतिरिक्त टेक्स्ट, या टिप्पणियाँ न जोड़ें—सिर्फ JSON लौटाएँ।

    टेक्स्ट:
    {text.strip()}"""

        if is_instruct_model:
            # Use instruction format for instruction-tuned models
            return f"<s>[INST]{instructions.strip()}[/INST]"
        else:
            # Use completion format for base models
            return f"{instructions.strip()}\n\nJSON Output:"

    def _is_instruction_tuned_model(self) -> bool:
        """Check if the loaded model is instruction-tuned based on model name"""
        if not self.model_id:
            return False

        # Convert to lowercase for case-insensitive matching
        model_name = self.model_id.lower()

        # List of patterns that indicate instruction-tuned models
        instruct_patterns = [
            'instruct',
            'instruction',
            'chat',
            'dolphin',
            'vicuna',
            'alpaca',
            'wizard',
            'openchat',
            'airoboros',
            'nous-hermes',
            'guanaco',
            'orca',
            'platypus',
            'samantha',
            'manticore'
        ]

        # Check if any instruction pattern is in the model name
        for pattern in instruct_patterns:
            if pattern in model_name:
                return True

        # Additional check for common instruction model naming conventions
        if any(suffix in model_name for suffix in ['-it', '-sft', '-dpo', '-rlhf']):
            return True

        # Default to False for base models
        return False

    def _safe_json_parse(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response with error handling and cleanup"""
        text = response_text.strip()

        # Remove code fences if present
        if text.startswith("```"):
            text = text.strip("`")
            text = re.sub(r"^json", "", text, flags=re.IGNORECASE).strip()

        # Try to parse the JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find the last complete JSON object
            try:
                last_brace = text.rfind("}")
                if last_brace != -1:
                    return json.loads(text[:last_brace + 1])
            except json.JSONDecodeError:
                pass

            logger.warning(f"Failed to parse JSON response: {text[:200]}...")
            return {}

    def _merge_results(self, llm_json: Dict[str, Any], regex_extractions: Dict[str, Any]) -> Dict[str, Any]:
        """Merge LLM results with regex-based extractions"""
        result = dict(self.json_schema)

        # List fields to merge
        list_fields = [
            "person_names", "organisation_names", "location_names", "district_names",
            "thana_names", "incidents", "caste_names", "religion_names", "hashtags",
            "mention_ids", "events"
        ]

        # Merge list fields
        for field in list_fields:
            combined_values = []

            # Add LLM results
            llm_values = llm_json.get(field, [])
            if isinstance(llm_values, list):
                combined_values.extend([str(val).strip() for val in llm_values if val])

            # Add regex results
            regex_values = regex_extractions.get(field, [])
            if isinstance(regex_values, list):
                combined_values.extend([str(val).strip() for val in regex_values if val])

            result[field] = self._dedupe(combined_values)

        # Handle sentiment with better error handling
        sentiment = llm_json.get("sentiment", {})
        if isinstance(sentiment, dict) and "label" in sentiment and "confidence" in sentiment:
            try:
                # Safely parse confidence, default to 0.5 if invalid
                confidence_val = sentiment.get("confidence", 0.5)
                if isinstance(confidence_val, str):
                    confidence_val = float(confidence_val) if confidence_val.strip() else 0.5
                elif confidence_val is None:
                    confidence_val = 0.5

                result["sentiment"] = {
                    "label": str(sentiment.get("label", "neutral")),
                    "confidence": float(confidence_val)
                }
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid sentiment confidence value: {sentiment.get('confidence')} - using default")
                result["sentiment"] = {
                    "label": str(sentiment.get("label", "neutral")),
                    "confidence": 0.5
                }
        else:
            # Default sentiment
            result["sentiment"] = {
                "label": "neutral",
                "confidence": 0.5
            }

        # Handle contextual understanding
        context = llm_json.get("contextual_understanding", "")
        result["contextual_understanding"] = str(context).strip()

        return result

    def extract(self, text: str, max_tokens: int = 1024, temperature: float = 0.2) -> Dict[str, Any]:
        """Extract entities and information from text using Mistral model"""
        if not text or not text.strip():
            logger.warning("Empty text provided for extraction")
            return dict(self.json_schema)

        start_time = time.time()

        try:
            # Step 1: Regex-based extractions (fast and reliable)
            regex_extractions = {
                "hashtags": self._find_hashtags(text),
                "mention_ids": self._find_mentions(text),
                "district_names": self._find_districts(text),
                "thana_names": self._find_thana(text),
                "caste_names": self._find_keywords(text, self.castes),
                "religion_names": self._find_keywords(text, self.religions),
            }

            # Step 2: LLM-based extraction
            llm_json = {}

            if self._model_loaded and self.model is not None:
                prompt = self._build_instruction_prompt(text)

                try:
                    raw_response = ""

                    # Try vLLM generation first (fastest)
                    if self._loading_method == 'vllm' and VLLM_AVAILABLE:
                        try:
                            # Create sampling parameters for vLLM
                            sampling_params = SamplingParams(
                                temperature=temperature,
                                max_tokens=max_tokens,
                                stop=[self.tokenizer.eos_token] if hasattr(self.tokenizer, 'eos_token') else None,
                                top_p=0.9,
                                frequency_penalty=0.1
                            )

                            # Generate with vLLM
                            outputs = self.model.generate([prompt], sampling_params)
                            raw_response = outputs[0].outputs[0].text.strip()
                            logger.debug("Used vLLM generation")

                        except Exception as e:
                            logger.warning(f"vLLM generation failed: {e}")
                            raw_response = ""

                    # Try MLX generation second
                    elif self._loading_method == 'mlx' and MLX_AVAILABLE:
                        try:
                            raw_response = mlx_generate(
                                self.model,
                                self.tokenizer,
                                prompt=prompt,
                                max_tokens=max_tokens,
                                temp=temperature,  # Fixed: use 'temp' not 'temperature'
                                verbose=False
                            )
                            logger.debug("Used MLX generation")
                        except Exception as e:
                            logger.warning(f"MLX generation failed: {e}")
                            raw_response = ""

                    # Fallback to transformers generation
                    elif self._loading_method == 'transformers' and TRANSFORMERS_AVAILABLE:
                        try:
                            # Improved transformers generation
                            inputs = self.tokenizer(
                                prompt,
                                return_tensors="pt",
                                truncation=True,
                                max_length=2048,
                                padding=True
                            )

                            # Move to appropriate device
                            if hasattr(self.model, 'device'):
                                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                            with torch.no_grad():
                                outputs = self.model.generate(
                                    **inputs,
                                    max_new_tokens=max_tokens,
                                    temperature=temperature,
                                    do_sample=True if temperature > 0 else False,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    attention_mask=inputs.get('attention_mask')  # Fixed: include attention mask
                                )

                            # Decode and remove the original prompt
                            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                            raw_response = full_response[len(prompt):].strip()
                            logger.debug("Used Transformers generation")

                        except Exception as e:
                            logger.warning(f"Transformers generation failed: {e}")
                            raw_response = ""

                    if raw_response:
                        llm_json = self._safe_json_parse(raw_response)
                        logger.info("LLM extraction successful")
                    else:
                        logger.info("No LLM response available, using regex extractions only")
                        llm_json = {}

                except Exception as e:
                    logger.error(f"LLM generation failed: {e}")
                    llm_json = {}
            else:
                logger.warning("Model not available, using regex extractions only")

            # Step 3: Merge results
            final_result = self._merge_results(llm_json, regex_extractions)

            processing_time = (time.time() - start_time) * 1000
            logger.info(f"NER extraction completed in {processing_time:.2f}ms")

            return final_result

        except Exception as e:
            logger.error(f"NER extraction failed for text: {text[:100]}... Error: {e}")
            return dict(self.json_schema)

    def extract_batch(self, texts: List[str], max_tokens: int = 1024, temperature: float = 0.2) -> List[Dict[str, Any]]:
        """Extract entities from multiple texts efficiently"""
        if not self._model_loaded or self.model is None:
            logger.warning("Model not available for batch processing")
            return [dict(self.json_schema) for _ in texts]

        results = []
        total_start = time.time()

        # Special handling for vLLM batch processing
        if self._loading_method == 'vllm' and VLLM_AVAILABLE:
            try:
                # Prepare all prompts
                prompts = [self._build_instruction_prompt(text) for text in texts]

                # Create sampling parameters
                sampling_params = SamplingParams(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=[self.tokenizer.eos_token] if hasattr(self.tokenizer, 'eos_token') else None,
                    top_p=0.9,
                    frequency_penalty=0.1
                )

                # Batch generation with vLLM
                outputs = self.model.generate(prompts, sampling_params)

                # Process results
                for i, output in enumerate(outputs):
                    try:
                        text = texts[i]
                        raw_response = output.outputs[0].text.strip()

                        # Regex extractions
                        regex_extractions = {
                            "hashtags": self._find_hashtags(text),
                            "mention_ids": self._find_mentions(text),
                            "district_names": self._find_districts(text),
                            "thana_names": self._find_thana(text),
                            "caste_names": self._find_keywords(text, self.castes),
                            "religion_names": self._find_keywords(text, self.religions),
                        }

                        # Parse LLM response
                        llm_json = self._safe_json_parse(raw_response) if raw_response else {}

                        # Merge and add result
                        final_result = self._merge_results(llm_json, regex_extractions)
                        results.append(final_result)

                    except Exception as e:
                        logger.error(f"Failed to process batch item {i}: {e}")
                        results.append(dict(self.json_schema))

                total_time = time.time() - total_start
                logger.info(f"vLLM batch extraction completed: {len(texts)} texts in {total_time:.2f}s")
                return results

            except Exception as e:
                logger.error(f"vLLM batch processing failed: {e}")
                # Fall back to sequential processing

        # Sequential processing for MLX and Transformers
        for i, text in enumerate(texts):
            try:
                result = self.extract(text, max_tokens, temperature)
                results.append(result)

                if (i + 1) % 10 == 0:
                    elapsed = time.time() - total_start
                    avg_time = elapsed / (i + 1)
                    eta = avg_time * (len(texts) - i - 1)
                    logger.info(f"Processed {i + 1}/{len(texts)} texts in {elapsed:.2f}s (ETA: {eta:.2f}s)")

            except Exception as e:
                logger.error(f"Failed to process text {i + 1}: {e}")
                results.append(dict(self.json_schema))

        total_time = time.time() - total_start
        logger.info(f"Sequential batch extraction completed: {len(texts)} texts in {total_time:.2f}s")
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "model_loaded": self._model_loaded,
            "loading_method": self._loading_method,
            "cache_dir": self.cache_dir,
            "tokenizer_loaded": self.tokenizer is not None,
            "supported_languages": ["hindi", "english", "hinglish"],
            "extraction_capabilities": list(self.json_schema.keys()),
            "vllm_available": VLLM_AVAILABLE,
            "mlx_available": MLX_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE
        }

    def test_extraction(self, sample_text: str = None) -> Dict[str, Any]:
        """Test the extraction with a sample text"""
        if sample_text is None:
            sample_text = "लखनऊ के गोमती नगर थाने में पुलिस कदाचार की शिकायत दर्ज की गई। राम शर्मा नाम के व्यक्ति पर अत्याचार हुआ। #यूपीपुलिस #न्याय"

        logger.info("Running extraction test...")
        result = self.extract(sample_text)

        # Log test results
        logger.info(f"Test completed. Found {len(result.get('person_names', []))} persons, "
                    f"{len(result.get('incidents', []))} incidents, "
                    f"{len(result.get('hashtags', []))} hashtags")

        return result

    def reload_model(self) -> bool:
        """Force reload the model"""
        logger.info("Reloading model...")
        self._model_loaded = False
        self.model = None
        self.tokenizer = None
        self._loading_method = None
        return self._load_model()

    def unload_model(self):
        """Unload model to free memory"""
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        self._loading_method = None
        logger.info("Model unloaded from memory")

    def add_custom_district(self, district_name: str):
        """Add a custom district to the recognition list"""
        if district_name and district_name not in self.up_districts:
            self.up_districts.append(district_name)
            logger.info(f"Added custom district: {district_name}")

    def add_custom_keywords(self, keywords: List[str], category: str):
        """Add custom keywords to existing categories"""
        if category == "religions" and hasattr(self, 'religions'):
            new_keywords = [kw for kw in keywords if kw not in self.religions]
            self.religions.extend(new_keywords)
            logger.info(f"Added {len(new_keywords)} new keywords to religions")
        elif category == "castes" and hasattr(self, 'castes'):
            new_keywords = [kw for kw in keywords if kw not in self.castes]
            self.castes.extend(new_keywords)
            logger.info(f"Added {len(new_keywords)} new keywords to castes")
        else:
            logger.warning(f"Unknown category: {category}")

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about the extractor's capabilities"""
        return {
            "total_districts": len(self.up_districts),
            "total_religions": len(self.religions),
            "total_castes": len(self.castes),
            "pattern_count": len(self.thana_patterns),
            "schema_fields": len(self.json_schema),
            "model_loaded": self._model_loaded,
            "loading_method": self._loading_method,
            "cache_dir": self.cache_dir,
            "vllm_available": VLLM_AVAILABLE,
            "mlx_available": MLX_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE
        }


# Debugging helper function
def debug_model_loading(model_path: str):
    """Debug function to check model loading issues with all methods"""
    logger.info("Starting comprehensive model loading debug...")

    # Check path exists
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        return False

    logger.info(f"Model path exists: {model_path}")

    # List contents
    try:
        contents = os.listdir(model_path)
        logger.info(f"Model directory contents: {contents}")

        # Check for required files
        required_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
        for file in required_files:
            if file in contents:
                logger.info(f"Found {file}")
            else:
                logger.warning(f"Missing {file}")

    except Exception as e:
        logger.error(f"Error reading model directory: {e}")
        return False

    # Test vLLM loading first
    if VLLM_AVAILABLE:
        try:
            logger.info("Testing vLLM loading...")
            model = LLM(
                model=model_path,
                tokenizer=model_path,
                trust_remote_code=True,
                dtype="auto",
                gpu_memory_utilization=0.85,
                max_model_len=2048,  # Smaller for testing
                tensor_parallel_size=1,
                disable_log_stats=True,
                enforce_eager=False,
            )
            tokenizer = get_tokenizer(
                tokenizer_name=model_path,
                trust_remote_code=True
            )
            logger.info("vLLM loading successful")
            return True
        except Exception as e:
            logger.error(f"vLLM loading failed: {e}")
    else:
        logger.info("vLLM not available, skipping test")

    # Test MLX loading
    if MLX_AVAILABLE:
        try:
            logger.info("Testing MLX loading...")
            model, tokenizer = mlx_load(model_path)
            logger.info("MLX loading successful")
            return True
        except Exception as e:
            logger.error(f"MLX loading failed: {e}")
    else:
        logger.info("MLX not available, skipping test")

    # Test Transformers loading
    if TRANSFORMERS_AVAILABLE:
        try:
            logger.info("Testing Transformers loading...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            logger.info("Tokenizer loading successful")

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            logger.info("Model loading successful")
            return True
        except Exception as e:
            logger.error(f"Transformers loading failed: {e}")
    else:
        logger.info("Transformers not available, skipping test")

    logger.error("All loading methods failed or unavailable")
    return False


def check_dependencies():
    """Check which dependencies are available"""
    dependencies = {
        "vLLM": VLLM_AVAILABLE,
        "MLX": MLX_AVAILABLE,
        "Transformers": TRANSFORMERS_AVAILABLE,
        "HuggingFace Hub": HF_HUB_AVAILABLE
    }

    logger.info("Dependency Status:")
    for name, available in dependencies.items():
        status = "Available" if available else "Not Available"
        logger.info(f"  {name}: {status}")

    return dependencies


if __name__ == "__main__":
    print("Initializing Complete Mistral NER Extractor with vLLM Support...")

    # Check dependencies first
    print("\nChecking dependencies...")
    deps = check_dependencies()

    if not any(deps[key] for key in ["vLLM", "MLX", "Transformers"]):
        print("ERROR: No model loading framework available!")
        print("Please install at least one of: vllm, mlx, or transformers")
        exit(1)

    # Model path from your setup
    model_path = "/Users/pankajkumar/.cache/huggingface/hub/models--mlx-community--Dolphin-Mistral-24B-Venice-Edition-4bit/snapshots/7674b37fe24022cf79e77d204fac5b9582b0dc40"

    # Debug model loading first
    print("\nDebugging model loading...")
    debug_success = debug_model_loading(model_path)

    if not debug_success:
        print("Model loading debug failed. Please check the issues above.")
        exit(1)

    # Initialize extractor
    try:
        print("\nInitializing extractor...")
        extractor = MistralNERExtractor(
            model_path=model_path,
            model_id="mlx-community/Dolphin-Mistral-24B-Venice-Edition-4bit"
        )

        # Check model info
        model_info = extractor.get_model_info()
        print("\nModel Info:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")

        # Test extraction
        print("\nTesting extraction...")
        test_result = extractor.test_extraction()
        print("Test Result:")
        print(json.dumps(test_result, ensure_ascii=False, indent=2))

        # Test batch processing with small batch
        print("\nTesting batch processing...")
        batch_texts = [
            "लखनऊ में प्रदर्शन हुआ।",
            "दिल्ली के कनॉट प्लेस में ट्रैफिक जाम।",
            "आगरा के सदर थाने में चोरी की रिपोर्ट दर्ज की गई। #आगरापुलिस"
        ]

        batch_results = extractor.extract_batch(batch_texts)
        print(f"Processed {len(batch_results)} texts successfully!")

        # Show extraction stats
        print("\nExtraction Statistics:")
        stats = extractor.get_extraction_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\nExtractor is working properly!")
        print("The system is ready for production use.")

        # Performance recommendation
        if model_info.get("loading_method") == "vllm":
            print("\nPERFORMANCE: Using vLLM - optimal for production inference!")
        elif model_info.get("loading_method") == "mlx":
            print("\nPERFORMANCE: Using MLX - optimized for Apple Silicon!")
        elif model_info.get("loading_method") == "transformers":
            print("\nPERFORMANCE: Using Transformers - consider installing vLLM for better performance!")

    except Exception as e:
        print(f"Extractor initialization failed: {e}")
        import traceback

        traceback.print_exc()
        print("\nTroubleshooting Tips:")
        print("1. Check if the model path exists and is accessible")
        print("2. Ensure you have sufficient GPU memory (if using vLLM)")
        print("3. Try installing missing dependencies:")
        print("   pip install vllm  # For GPU acceleration")
        print("   pip install mlx mlx-lm  # For Apple Silicon")
        print("   pip install transformers torch  # Fallback option")
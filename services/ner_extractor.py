import os
import re
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Required imports for MLX and Hugging Face models
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
    Optimized for efficient model loading and reuse with MLX support
    """

    def __init__(self, model_id: str = None, model_path: Optional[str] = None, cache_dir: Optional[str] = None):
        # 1. Resolve model id
        # self.model_id = model_id or "dphn/Dolphin-Mistral-24B-Venice-Edition"
        self.model_id = model_id or "mistralai/Mistral-7B-Instruct-v0.3"
        # self.model_id = model_id or "mistralai/Mixtral-8x22B-Instruct-v0.1"
        # self.model_id = model_id or "mistralai/Mixtral-8x7B-Instruct-v0.1"
        # self.model_id = model_id or "mistralai/Mixtral-8x7B-v0.1"
        # self.model_id = model_id or "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
        # self.model_id = model_id or "dphn/Dolphin-Mistral-24B-Venice-Edition"
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/mistral_ner")

        # 2. Resolve model path
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Provided model path does not exist: {model_path}")
            self.model_path = model_path
            logger.info(f"Using provided model path: {self.model_path}")
        else:
            # Ensure cache dir exists
            os.makedirs(self.cache_dir, exist_ok=True)
            self.model_path = self._get_or_download_model()

        # Initialize placeholders
        self.model = None
        self.tokenizer = None
        self._model_loaded = False

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
        """
        Check if model exists locally in cache.
        If not, download from Hugging Face Hub and return local path.
        """
        logger.info(f"Checking local cache for model {self.model_id} ...")

        # Use HuggingFace cache
        try:
            local_path = snapshot_download(
                repo_id=self.model_id,
                cache_dir=self.cache_dir,
                local_files_only=False  # Falls back to local if already cached
            )
            logger.info(f"Model available at: {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Failed to download or find model: {e}")
            raise

    def _get_model_cache_path(self) -> str:
        """Get the local cache path for the model"""
        if self.model_path:
            return self.model_path
        model_name = self.model_id.replace("/", "_")
        return os.path.join(self.cache_dir, model_name)

    def _is_model_cached(self) -> bool:
        """Check if model is already cached locally"""
        cache_path = self._get_model_cache_path()
        return os.path.exists(cache_path) and (
                os.path.isfile(cache_path) or
                (os.path.isdir(cache_path) and os.listdir(cache_path))
        )

    def _load_model(self, max_retries: int = 3, retry_wait: int = 5) -> bool:
        """
        Load the Mistral model with retry logic and caching.
        Supports both MLX and Hugging Face models.
        Returns True if successful, False otherwise.
        """
        if self._model_loaded and self.model is not None and self.tokenizer is not None:
            logger.info("Model already loaded, skipping reload")
            return True

        retry_count = 0
        while retry_count < max_retries:
            try:
                # Use direct path if provided, otherwise check cache or download
                if self.model_path:
                    local_path = self.model_path
                    logger.info(f"Loading model from provided path: {local_path}")
                elif self._is_model_cached():
                    local_path = self._get_model_cache_path()
                    logger.info(f"Loading model from local cache: {local_path}")
                else:
                    logger.info(f"Downloading model: {self.model_id}")
                    if not HF_HUB_AVAILABLE:
                        raise ImportError("huggingface_hub not available for downloading models")
                    # Download to our cache directory
                    local_path = snapshot_download(
                        repo_id=self.model_id,
                        cache_dir=self.cache_dir,
                        local_files_only=False
                    )

                # Try MLX loading first (for 4-bit models), then fallback to standard loading
                if MLX_AVAILABLE:
                    try:
                        self.model, self.tokenizer = mlx_load(local_path)
                        logger.info(f"Successfully loaded MLX model: {self.model_id}")
                        self._model_loaded = True
                        return True
                    except Exception as e:
                        logger.warning(f"MLX loading failed: {e}")

                # Fallback to standard transformers loading
                if TRANSFORMERS_AVAILABLE:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(local_path)
                        self.model = AutoModelForCausalLM.from_pretrained(
                            local_path,
                            torch_dtype=torch.float16,
                            device_map="auto" if torch.cuda.is_available() else None
                        )
                        logger.info(f"Successfully loaded transformers model: {self.model_id}")
                        self._model_loaded = True
                        return True
                    except Exception as e:
                        logger.warning(f"Transformers loading failed: {e}")

                # If we reach here, no loading method worked
                raise Exception("No compatible model loading framework available")

            except Exception as e:
                retry_count += 1
                logger.warning(f"Failed to load model (attempt {retry_count}): {e}")

                if retry_count >= max_retries:
                    logger.error(f"Failed to load model after {max_retries} attempts: {e}")
                    self._model_loaded = False
                    return False

                logger.info(f"Retrying in {retry_wait} seconds...")
                time.sleep(retry_wait)

        return False

    def _ensure_model_loaded(self) -> bool:
        """Ensure model is loaded before processing"""
        if not self._model_loaded:
            logger.info("Model not loaded, attempting to load...")
            return self._load_model()
        return True

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
        """Build instruction prompt for Mistral model"""
        instructions = f"""
आप एक डेटा-एक्सट्रैक्शन सहायक हैं। नीचे दिए गए हिन्दी टेक्स्ट से पार्स कर के केवल वैध JSON लौटाइए। 
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
{text.strip()}
"""
        return f"<s>[INST]{instructions.strip()}[/INST]"

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

        # Handle sentiment
        sentiment = llm_json.get("sentiment", {})
        if isinstance(sentiment, dict) and "label" in sentiment and "confidence" in sentiment:
            result["sentiment"] = {
                "label": str(sentiment.get("label", "neutral")),
                "confidence": float(sentiment.get("confidence", 0.5))
            }

        # Handle contextual understanding
        context = llm_json.get("contextual_understanding", "")
        result["contextual_understanding"] = str(context).strip()

        return result

    def extract(self, text: str, max_tokens: int = 1024, temperature: float = 0.2) -> Dict[str, Any]:
        """
        Extract entities and information from text using Mistral model

        Args:
            text: Input text to process
            max_tokens: Maximum tokens for model generation
            temperature: Sampling temperature for model

        Returns:
            Dictionary containing extracted entities and information
        """
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

            # Step 2: LLM-based extraction (comprehensive but slower)
            llm_json = {}

            if self._ensure_model_loaded():
                prompt = self._build_instruction_prompt(text)

                try:
                    # Try MLX generation first, then fallback to standard methods
                    raw_response = ""

                    if MLX_AVAILABLE and self.model is not None and self._model_loaded:
                        try:
                            raw_response = mlx_generate(
                                self.model,
                                self.tokenizer,
                                prompt=prompt,
                                max_tokens=max_tokens,
                                # temperature=temperature,
                                verbose=False
                            )
                            logger.debug("Used MLX generation")
                        except Exception as e:
                            logger.warning(f"MLX generation failed: {e}")
                            raw_response = ""

                    # Fallback to transformers if MLX failed or not available
                    if not raw_response and TRANSFORMERS_AVAILABLE and self.model is not None:
                        try:
                            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

                            # Move to appropriate device
                            # if hasattr(self.model, 'device'):
                            #     inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                            #
                            # with torch.no_grad():
                            #     outputs = self.model.generate(
                            #         **inputs,
                            #         max_new_tokens=max_tokens,
                            #         temperature=temperature,
                            #         do_sample=True if temperature > 0 else False,
                            #         pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.pad_token_id,
                            #         eos_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.pad_token_id
                            #     )

                            # Handle both MLX tokenizer wrapper and standard tokenizer
                            if hasattr(self.tokenizer, 'encode'):
                                # For MLX tokenizer wrapper
                                inputs = self.tokenizer.encode(prompt)
                                if not isinstance(inputs, torch.Tensor):
                                    inputs = torch.tensor(inputs).unsqueeze(0)
                            else:
                                # For standard tokenizer
                                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                                inputs = inputs.input_ids

                            # Move to appropriate device
                            if hasattr(self.model, 'device'):
                                inputs = inputs.to(self.model.device)

                            with torch.no_grad():
                                outputs = self.model.generate(
                                    inputs,
                                    max_new_tokens=max_tokens,
                                    temperature=temperature,
                                    do_sample=True if temperature > 0 else False,
                                    pad_token_id=getattr(self.tokenizer, 'eos_token_id', 0),
                                    eos_token_id=getattr(self.tokenizer, 'eos_token_id', 0)
                                )

                            # Decode and remove the original prompt
                            # full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                            # Handle both MLX and standard tokenizer decode
                            if hasattr(self.tokenizer, 'decode'):
                                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                            else:
                                # Fallback for MLX tokenizer wrapper
                                full_response = self.tokenizer.detokenize(outputs[0].tolist())

                            raw_response = full_response[len(prompt):].strip()
                            logger.debug("Used Transformers generation")

                        except Exception as e:
                            logger.warning(f"Transformers generation failed: {e}")
                            raw_response = ""

                    # If model is None (tokenizer-only mode), skip LLM generation
                    if self.model is None:
                        logger.info("Model not loaded, skipping LLM generation")
                        raw_response = ""

                    if raw_response:
                        llm_json = self._safe_json_parse(raw_response)
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
        """
        Extract entities from multiple texts efficiently

        Args:
            texts: List of input texts
            max_tokens: Maximum tokens per generation
            temperature: Sampling temperature

        Returns:
            List of extraction results
        """
        if not self._ensure_model_loaded():
            logger.warning("Model not available for batch processing")
            return [dict(self.json_schema) for _ in texts]

        results = []
        total_start = time.time()

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
        logger.info(f"Batch extraction completed: {len(texts)} texts in {total_time:.2f}s")

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_id": self.model_id,
            "model_loaded": self._model_loaded,
            "model_cached": self._is_model_cached(),
            "cache_dir": self.cache_dir,
            "tokenizer_loaded": self.tokenizer is not None,
            "supported_languages": ["hindi", "english", "hinglish"],
            "extraction_capabilities": list(self.json_schema.keys()),
            "mlx_available": MLX_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE
        }

    def validate_extraction_result(self, result: Dict[str, Any]) -> bool:
        """Validate extraction result against schema"""
        try:
            # Check if all required keys are present
            for key in self.json_schema.keys():
                if key not in result:
                    logger.warning(f"Missing key in result: {key}")
                    return False

            # Check data types
            list_fields = [k for k, v in self.json_schema.items() if isinstance(v, list)]
            for field in list_fields:
                if not isinstance(result[field], list):
                    logger.warning(f"Invalid type for field {field}: expected list, got {type(result[field])}")
                    return False

            # Check sentiment structure
            sentiment = result.get("sentiment", {})
            if not isinstance(sentiment, dict) or "label" not in sentiment or "confidence" not in sentiment:
                logger.warning("Invalid sentiment structure")
                return False

            return True

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def test_extraction(self, sample_text: str = None) -> Dict[str, Any]:
        """
        Test the extraction with a sample text

        Args:
            sample_text: Optional sample text, uses default if not provided

        Returns:
            Test extraction result
        """
        if sample_text is None:
            sample_text = "लखनऊ के गोमती नगर थाने में पुलिस कदाचार की शिकायत दर्ज की गई। राम शर्मा नाम के व्यक्ति पर अत्याचार हुआ। #यूपीपुलिस #न्याय"

        logger.info("Running extraction test...")
        result = self.extract(sample_text)

        # Validate result
        is_valid = self.validate_extraction_result(result)

        # Log test results
        logger.info(f"Test completed. Valid: {is_valid}. Found {len(result.get('person_names', []))} persons, "
                    f"{len(result.get('incidents', []))} incidents, "
                    f"{len(result.get('hashtags', []))} hashtags")

        return result

    def reload_model(self) -> bool:
        """Force reload the model"""
        self._model_loaded = False
        self.model = None
        self.tokenizer = None
        return self._load_model()

    def unload_model(self):
        """Unload model to free memory"""
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        logger.info("Model unloaded from memory")

    def add_custom_district(self, district_name: str):
        """Add a custom district to the recognition list"""
        if district_name and district_name not in self.up_districts:
            self.up_districts.append(district_name)
            logger.info(f"Added custom district: {district_name}")

    def add_custom_keywords(self, keywords: List[str], category: str):
        """
        Add custom keywords to existing categories

        Args:
            keywords: List of keywords to add
            category: Category name (religions, castes, etc.)
        """
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
            return

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about the extractor's capabilities"""
        return {
            "total_districts": len(self.up_districts),
            "total_religions": len(self.religions),
            "total_castes": len(self.castes),
            "pattern_count": len(self.thana_patterns),
            "schema_fields": len(self.json_schema),
            "model_loaded": self._model_loaded,
            "model_cached": self._is_model_cached(),
            "cache_dir": self.cache_dir,
            "mlx_available": MLX_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE
        }


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Initializing Mistral NER Extractor...")

    # Initialize with the exact path that works in your script
    extractor = MistralNERExtractor(
        model_path="/Users/pankajkumar/.cache/huggingface/hub/models--mlx-community--Dolphin-Mistral-24B-Venice-Edition-4bit/snapshots/7674b37fe24022cf79e77d204fac5b9582b0dc40",
        model_id="mlx-community/Dolphin-Mistral-24B-Venice-Edition-4bit"
    )

    # Check model info
    model_info = extractor.get_model_info()
    print("📊 Model Info:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    print()

    # Test with sample text
    print("🧪 Running test extraction...")
    test_result = extractor.test_extraction()
    print("✅ Test Result:")
    print(json.dumps(test_result, ensure_ascii=False, indent=2))
    print()

    # Test with custom text
    custom_text = "मुंबई के बांद्रा थाने में राहुल शर्मा नाम के व्यक्ति पर हमला हुआ। पुलिस ने FIR दर्ज की है। #मुंबईपुलिस #न्याय @MumbaiPolice"
    print("🔍 Testing with custom text:")
    print(f"Input: {custom_text}")

    custom_result = extractor.extract(custom_text)
    print("Result:")
    print(json.dumps(custom_result, ensure_ascii=False, indent=2))
    print()

    # Test batch processing
    print("📦 Testing batch processing...")
    batch_texts = [
        "लखनऊ में प्रदर्शन हुआ।",
        "दिल्ली के कनॉट प्लेस में ट्रैफिक जाम।",
        "आगरा के सदर थाने में चोरी की रिपोर्ट दर्ज की गई। #आगराpolic"
    ]

    batch_results = extractor.extract_batch(batch_texts)
    print(f"✅ Processed {len(batch_results)} texts successfully!")

    # Show extraction stats
    print("\n📈 Extraction Statistics:")
    stats = extractor.get_extraction_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n🎉 All tests completed successfully!")
    print("💡 The model is now loaded and ready for production use!")
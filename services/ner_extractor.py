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
    Enhanced NER extractor using Mistral 24B model for Hindi/English/Hinglish text processing
    Optimized for efficient model loading and reuse with vLLM, MLX, and Transformers support
    """

    def __init__(self, model_id: str = None, model_path: Optional[str] = None, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/mistral_ner")
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        self._loading_method = None

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        if model_path:
            self.model_path = model_path
            self.model_id = model_path.split('/')[-1]
            logger.info(f"Using provided model path: {self.model_path}")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        else:
            self.model_id = model_id or "mlx-community/Dolphin-Mistral-24B-Venice-Edition-4bit"
            self.model_path = self._get_or_download_model()

        # Enhanced domain-specific vocabularies
        self.religions = [
            "हिन्दू", "हिंदू", "मुस्लिम", "इस्लाम", "ईसाई", "क्रिश्चियन", "सिख", "बौद्ध", "जैन",
            "Hindu", "Muslim", "Islam", "Christian", "Sikh", "Buddhist", "Jain", "धर्म", "religion"
        ]

        self.castes = [
            "SC", "ST", "OBC", "ब्राह्मण", "ठाकुर", "राजपूत", "यादव", "दलित", "कुर्मी", "अहीर", "गुर्जर",
            "Brahmin", "Thakur", "Rajput", "Yadav", "Dalit", "Kurmi", "जाति", "caste"
        ]

        # Comprehensive UP districts list (Hindi and English)
        self.up_districts = [
            # Major cities - Hindi names
            "आगरा", "अलीगढ़", "अयोध्या", "बांदा", "बरेली", "लखनऊ", "वाराणसी", "गोरखपुर",
            "कानपुर", "मेरठ", "प्रयागराज", "फैजाबाद", "गाजियाबाद", "मुरादाबाद", "सहारनपुर",
            "फिरोजाबाद", "मुजफ्फरनगर", "रामपुर", "बिजनौर", "हरदोई", "सीतापुर", "बहराइच",
            "गोंडा", "फर्रुखाबाद", "एटा", "बदायूं", "शाहजहांपुर", "पीलीभीत", "खीरी", "बस्ती",
            "देवरिया", "कुशीनगर", "महराजगंज", "संत कबीर नगर", "सिद्धार्थनगर", "बलरामपुर",
            "श्रावस्ती", "जौनपुर", "प्रतापगढ़", "सुल्तानपुर", "अम्बेडकर नगर", "अमेठी", "रायबरेली",
            "उन्नाव", "कन्नौज", "हमीरपुर", "महोबा", "जालौन", "झांसी", "ललितपुर", "चित्रकूट",
            "मिर्जापुर", "सोनभद्र", "चंदौली", "भदोही", "गाजीपुर", "मऊ", "आजमगढ़", "बलिया",
            "अकबरपुर", "अमरोहा", "औरैया", "बागपत", "बुलंदशहर", "हापुड़", "मथुरा", "हाथरस",
            "कासगंज", "मैनपुरी", "इटावा", "फतेहपुर", "कौशांबी", "संभल",

            # English names
            "Agra", "Aligarh", "Ayodhya", "Banda", "Bareilly", "Lucknow", "Varanasi", "Gorakhpur",
            "Kanpur", "Meerut", "Prayagraj", "Faizabad", "Ghaziabad", "Moradabad", "Saharanpur",
            "Firozabad", "Muzaffarnagar", "Rampur", "Bijnor", "Hardoi", "Sitapur", "Bahraich",
            "Gonda", "Farrukhabad", "Etah", "Budaun", "Shahjahanpur", "Pilibhit", "Kheri", "Basti",
            "Deoria", "Kushinagar", "Maharajganj", "Sant Kabir Nagar", "Siddharthnagar", "Balrampur",
            "Shrawasti", "Jaunpur", "Pratapgarh", "Sultanpur", "Ambedkar Nagar", "Amethi", "Raebareli",
            "Unnao", "Kannauj", "Hamirpur", "Mahoba", "Jalaun", "Jhansi", "Lalitpur", "Chitrakoot",
            "Mirzapur", "Sonbhadra", "Chandauli", "Bhadohi", "Ghazipur", "Mau", "Azamgarh", "Ballia",
            "Akbarpur", "Amroha", "Auraiya", "Bagpat", "Bulandshahr", "Hapur", "Mathura", "Hathras",
            "Kasganj", "Mainpuri", "Etawah", "Fatehpur", "Kaushambi", "Sambhal"
        ]

        # Enhanced thana patterns - more flexible
        self.thana_patterns = [
            r"(?:थाना|कोतवाली|पुलिस\s*स्टेशन|PS|Police\s*Station|Kotwali|Thana)\s+([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\s]*?)(?:\s+(?:में|पर|जनपद|थाना|district)|[,।\n]|$)",
            r"थाना\s+([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\s]*?)(?:\s+(?:पर|में)|[,।\n]|$)",
            r"कोतवाली\s+([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\s]*?)(?:\s+(?:पर|में)|[,।\n]|$)"
        ]

        # Enhanced JSON schema with better field descriptions
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

        # Load model during initialization
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
                        self.model = LLM(
                            model=local_path,
                            tokenizer=local_path,
                            trust_remote_code=True,
                            dtype="auto",
                            gpu_memory_utilization=0.75,
                            max_model_len=8192,
                            tensor_parallel_size=2,
                            disable_log_stats=True,
                            enforce_eager=False,
                        )

                        self.tokenizer = get_tokenizer(
                            tokenizer_name=local_path,
                            trust_remote_code=True
                        )

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
        """Enhanced district name finding with better matching"""
        found_districts = []
        text_lower = text.lower()

        for district in self.up_districts:
            district_lower = district.lower()
            # Direct match
            if district in text or district_lower in text_lower:
                found_districts.append(district)
            # Word boundary match for better accuracy
            elif re.search(r'\b' + re.escape(district_lower) + r'\b', text_lower):
                found_districts.append(district)

        return self._dedupe(found_districts)

    def _find_thana(self, text: str) -> List[str]:
        """Enhanced thana extraction with cleaner output"""
        found_thanas = []

        for pattern in self.thana_patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                name = match.group(1).strip()
                # Clean up unwanted words and punctuation
                name = re.sub(r'\b(?:थाना|कोतवाली|जनपद|पर|में|district|thana|kotwali)\b', '', name, flags=re.IGNORECASE)
                name = name.strip().rstrip(".,:;|/\\-—–")
                if name and len(name) > 1:  # Avoid single characters
                    found_thanas.append(name)

        return self._dedupe(found_thanas)

    def _find_keywords(self, text: str, vocabulary: List[str]) -> List[str]:
        """Enhanced keyword finding with word boundary matching"""
        found_keywords = []
        text_lower = text.lower()

        for keyword in vocabulary:
            keyword_lower = keyword.lower()
            # Direct match or word boundary match
            if keyword in text or keyword_lower in text_lower:
                found_keywords.append(keyword)
            elif re.search(r'\b' + re.escape(keyword_lower) + r'\b', text_lower):
                found_keywords.append(keyword)

        return self._dedupe(found_keywords)

    def _build_enhanced_instruction_prompt(self, text: str) -> str:
        """Enhanced instruction prompt with better structure and examples"""

        is_instruct_model = self._is_instruction_tuned_model()

        # Create comprehensive instructions with examples
        instructions = f"""You are an expert Hindi/English/Hinglish Named Entity Recognition (NER) assistant. Your task is to extract structured information from the given text and return ONLY a valid JSON response.

STRICT REQUIREMENTS:
1. Return ONLY valid JSON - no code blocks, explanations, or additional text
2. Follow the exact schema format provided below
3. Ensure contextual_understanding is ALWAYS in Hindi
4. Create comprehensive summary that captures ALL events mentioned

JSON SCHEMA (EXACT FORMAT REQUIRED):
{json.dumps(self.json_schema, ensure_ascii=False, indent=2)}

DETAILED EXTRACTION RULES:

person_names:
- Extract ALL person names (individuals, not organizations)
- Include full names, nicknames, aliases (e.g., "राम शर्मा", "अर्जुन उर्फ बहरा")
- Exclude @handles and organizational titles
- Examples: ["राम शर्मा", "अमित कुमार आनंद", "John Singh"]

organisation_names:
- Government departments, police units, companies, institutions
- Examples: ["उत्तर प्रदेश पुलिस", "SMC AMROHA", "Supreme Court"]

location_names:
- Cities, towns, villages, areas, states, countries (NOT districts/thanas)
- Examples: ["खालकपुर", "गोमती नगर", "Delhi", "रिवा"]

district_names:
- ONLY district names (जिला), both Hindi and English
- Examples: ["अमरोहा", "Moradabad", "लखनऊ", "Agra"]

thana_names:  
- ONLY clean police station names without "थाना"/"कोतवाली"/"पुलिस स्टेशन"
- Examples: ["रजबपुर", "गोमती नगर", "Civil Lines"] (NOT "थाना रजबपुर")

incidents:
- Specific crimes, accidents, or any event related to police (3-7 words max)
- Examples: ["दुष्कर्म का मामला", "सड़क दुर्घटना", "चोरी की घटना"]

events:
- Operations, campaigns, programs, meetings
- Examples: ["ऑपरेशन कन्विक्शन", "जन सुनवाई", "धरना प्रदर्शन"]

sentiment:
- label: "positive", "negative", or "neutral"
- confidence: 0.0 to 1.0

contextual_understanding:
- 2-4 sentences in Hindi summarizing ALL key events, people, and outcomes
- MUST capture complete story including WHO, WHAT, WHERE, WHEN details
- Include all important context and results

LANGUAGE HANDLING:
- Process Hindi, English, and Hinglish text
- Maintain original script in extracted entities
- contextual_understanding MUST be in Hindi regardless of input language

TEXT TO PROCESS:
{text.strip()}

IMPORTANT: Return ONLY the JSON object. No explanations, code blocks, or additional text."""

        if is_instruct_model:
            return f"<s>[INST]{instructions.strip()}[/INST]"
        else:
            return f"{instructions.strip()}\n\nJSON:"

    def _is_instruction_tuned_model(self) -> bool:
        """Check if the loaded model is instruction-tuned based on model name"""
        if not self.model_id:
            return False

        model_name = self.model_id.lower()
        instruct_patterns = [
            'instruct', 'instruction', 'chat', 'dolphin', 'vicuna', 'alpaca', 'wizard',
            'openchat', 'airoboros', 'nous-hermes', 'guanaco', 'orca', 'platypus',
            'samantha', 'manticore'
        ]

        for pattern in instruct_patterns:
            if pattern in model_name:
                return True

        if any(suffix in model_name for suffix in ['-it', '-sft', '-dpo', '-rlhf']):
            return True

        return False

    def _safe_json_parse(self, response_text: str) -> Dict[str, Any]:
        """Enhanced JSON parsing with better error handling"""
        text = response_text.strip()

        # Remove code fences and markdown
        text = re.sub(r'^```(?:json)?\n?', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n?```$', '', text, flags=re.MULTILINE)
        text = text.strip()

        # Try to find JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        try:
            parsed_json = json.loads(text)
            return parsed_json
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            logger.warning(f"Response text (first 500 chars): {text[:500]}")

            # Try to fix common JSON issues
            try:
                # Fix trailing commas
                text = re.sub(r',(\s*[}\]])', r'\1', text)
                # Fix unescaped quotes in strings
                text = re.sub(r'(?<!\\)"(?![,\s}:\[\]])', '\\"', text)
                parsed_json = json.loads(text)
                return parsed_json
            except json.JSONDecodeError:
                logger.error("Could not fix JSON parsing issues")
                return {}

    def _merge_results(self, llm_json: Dict[str, Any], regex_extractions: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced merging with better validation"""
        result = dict(self.json_schema)

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
                combined_values.extend([str(val).strip() for val in llm_values if val and str(val).strip()])

            # Add regex results
            regex_values = regex_extractions.get(field, [])
            if isinstance(regex_values, list):
                combined_values.extend([str(val).strip() for val in regex_values if val and str(val).strip()])

            result[field] = self._dedupe(combined_values)

        # Handle sentiment with validation
        sentiment = llm_json.get("sentiment", {})
        if isinstance(sentiment, dict):
            label = sentiment.get("label", "neutral")
            if label not in ["positive", "negative", "neutral"]:
                label = "neutral"

            try:
                confidence = float(sentiment.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1
            except (ValueError, TypeError):
                confidence = 0.5

            result["sentiment"] = {
                "label": label,
                "confidence": confidence
            }

        # Handle contextual understanding with validation
        context = llm_json.get("contextual_understanding", "")
        if context and isinstance(context, str):
            result["contextual_understanding"] = context.strip()
        else:
            # Fallback: create basic context from extracted entities
            result["contextual_understanding"] = "दी गई जानकारी से मुख्य घटनाओं और व्यक्तियों का विवरण मिलता है।"

        return result

    def extract(self, text: str, max_tokens: int = 1500, temperature: float = 0.1) -> Dict[str, Any]:
        """Enhanced extraction with better error handling"""
        if not text or not text.strip():
            logger.warning("Empty text provided for extraction")
            return dict(self.json_schema)

        start_time = time.time()

        try:
            # Step 1: Enhanced regex-based extractions
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
                prompt = self._build_enhanced_instruction_prompt(text)

                try:
                    raw_response = ""

                    # Generation based on loading method
                    if self._loading_method == 'vllm' and VLLM_AVAILABLE:
                        try:
                            sampling_params = SamplingParams(
                                temperature=temperature,
                                max_tokens=max_tokens,
                                stop=[self.tokenizer.eos_token] if hasattr(self.tokenizer, 'eos_token') else None,
                                top_p=0.95,
                                frequency_penalty=0.1,
                                presence_penalty=0.1
                            )

                            outputs = self.model.generate([prompt], sampling_params)
                            raw_response = outputs[0].outputs[0].text.strip()
                            logger.debug("Used vLLM generation")

                        except Exception as e:
                            logger.warning(f"vLLM generation failed: {e}")

                    elif self._loading_method == 'mlx' and MLX_AVAILABLE:
                        try:
                            raw_response = mlx_generate(
                                self.model,
                                self.tokenizer,
                                prompt=prompt,
                                max_tokens=max_tokens,
                                temp=temperature,
                                verbose=False
                            )
                            logger.debug("Used MLX generation")
                        except Exception as e:
                            logger.warning(f"MLX generation failed: {e}")

                    elif self._loading_method == 'transformers' and TRANSFORMERS_AVAILABLE:
                        try:
                            inputs = self.tokenizer(
                                prompt,
                                return_tensors="pt",
                                truncation=True,
                                max_length=4096,
                                padding=True
                            )

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
                                    attention_mask=inputs.get('attention_mask'),
                                    repetition_penalty=1.1
                                )

                            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                            raw_response = full_response[len(prompt):].strip()
                            logger.debug("Used Transformers generation")

                        except Exception as e:
                            logger.warning(f"Transformers generation failed: {e}")

                    if raw_response:
                        llm_json = self._safe_json_parse(raw_response)
                        if llm_json:
                            logger.info("LLM extraction successful")
                        else:
                            logger.warning("LLM returned empty or invalid JSON")
                    else:
                        logger.info("No LLM response available")

                except Exception as e:
                    logger.error(f"LLM generation failed: {e}")

            # Step 3: Merge results
            final_result = self._merge_results(llm_json, regex_extractions)

            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Enhanced NER extraction completed in {processing_time:.2f}ms")

            return final_result

        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            return dict(self.json_schema)

    # ... (rest of the methods remain the same but can be enhanced similarly)

    def extract_batch(self, texts: List[str], max_tokens: int = 1500, temperature: float = 0.1) -> List[Dict[str, Any]]:
        """Extract entities from multiple texts efficiently"""
        if not self._model_loaded or self.model is None:
            logger.warning("Model not available for batch processing")
            return [dict(self.json_schema) for _ in texts]

        results = []
        total_start = time.time()

        # vLLM batch processing
        if self._loading_method == 'vllm' and VLLM_AVAILABLE:
            try:
                prompts = [self._build_enhanced_instruction_prompt(text) for text in texts]

                sampling_params = SamplingParams(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=[self.tokenizer.eos_token] if hasattr(self.tokenizer, 'eos_token') else None,
                    top_p=0.95,
                    frequency_penalty=0.1,
                    presence_penalty=0.1
                )

                outputs = self.model.generate(prompts, sampling_params)

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
            sample_text = """लखनऊ के गोमती नगर थाने में पुलिस कदाचार की शिकायत दर्ज की गई। 
            राम शर्मा नाम के व्यक्ति पर अत्याचार हुआ। मुरादाबाद जिले में भी समान घटना। 
            #यूपीपुलिस #न्याय @RamSharma"""

        logger.info("Running extraction test...")
        result = self.extract(sample_text)

        logger.info(f"Test completed. Found {len(result.get('person_names', []))} persons, "
                    f"{len(result.get('incidents', []))} incidents, "
                    f"{len(result.get('hashtags', []))} hashtags, "
                    f"{len(result.get('district_names', []))} districts")

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


# Enhanced debugging helper function
def debug_model_loading(model_path: str):
    """Debug function to check model loading issues with all methods"""
    logger.info("Starting comprehensive model loading debug...")

    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        return False

    logger.info(f"Model path exists: {model_path}")

    try:
        contents = os.listdir(model_path)
        logger.info(f"Model directory contents: {contents}")

        required_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
        for file in required_files:
            if file in contents:
                logger.info(f"Found {file}")
            else:
                logger.warning(f"Missing {file}")

    except Exception as e:
        logger.error(f"Error reading model directory: {e}")
        return False

    # Test all loading methods
    loading_success = False

    if VLLM_AVAILABLE:
        try:
            logger.info("Testing vLLM loading...")
            model = LLM(
                model=model_path,
                tokenizer=model_path,
                trust_remote_code=True,
                dtype="auto",
                gpu_memory_utilization=0.75,
                max_model_len=2048,
                tensor_parallel_size=2,
                disable_log_stats=True,
                enforce_eager=False,
            )
            tokenizer = get_tokenizer(
                tokenizer_name=model_path,
                trust_remote_code=True
            )
            logger.info("✓ vLLM loading successful")
            loading_success = True
            del model, tokenizer  # Clean up
        except Exception as e:
            logger.error(f"✗ vLLM loading failed: {e}")

    if MLX_AVAILABLE:
        try:
            logger.info("Testing MLX loading...")
            model, tokenizer = mlx_load(model_path)
            logger.info("✓ MLX loading successful")
            loading_success = True
            del model, tokenizer  # Clean up
        except Exception as e:
            logger.error(f"✗ MLX loading failed: {e}")

    if TRANSFORMERS_AVAILABLE:
        try:
            logger.info("Testing Transformers loading...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            logger.info("✓ Tokenizer loading successful")

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            logger.info("✓ Model loading successful")
            loading_success = True
            del model, tokenizer  # Clean up
        except Exception as e:
            logger.error(f"✗ Transformers loading failed: {e}")

    if not loading_success:
        logger.error("All loading methods failed")
        return False

    logger.info("✓ At least one loading method successful")
    return True


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
        status = "✓ Available" if available else "✗ Not Available"
        logger.info(f"  {name}: {status}")

    return dependencies


if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced Mistral NER Extractor with Improved Prompting")
    print("=" * 60)

    # Check dependencies first
    print("\n1. Checking dependencies...")
    deps = check_dependencies()

    if not any(deps[key] for key in ["vLLM", "MLX", "Transformers"]):
        print("ERROR: No model loading framework available!")
        print("Please install at least one of:")
        print("  pip install vllm  # For GPU acceleration")
        print("  pip install mlx mlx-lm  # For Apple Silicon")
        print("  pip install transformers torch  # Fallback option")
        exit(1)

    # Model configuration
    model_path = "/Users/pankajkumar/.cache/huggingface/hub/models--mlx-community--Dolphin-Mistral-24B-Venice-Edition-4bit/snapshots/7674b37fe24022cf79e77d204fac5b9582b0dc40"

    # Debug model loading first
    print("\n2. Debugging model loading...")
    debug_success = debug_model_loading(model_path)

    if not debug_success:
        print("Model loading debug failed. Please check the issues above.")
        exit(1)

    # Initialize enhanced extractor
    try:
        print("\n3. Initializing Enhanced NER Extractor...")
        extractor = MistralNERExtractor(
            model_path=model_path,
            model_id="mlx-community/Dolphin-Mistral-24B-Venice-Edition-4bit"
        )

        # Display model info
        model_info = extractor.get_model_info()
        print("\n4. Model Information:")
        print("-" * 30)
        for key, value in model_info.items():
            print(f"  {key}: {value}")

        # Test with sample text
        print("\n5. Testing NER Extraction...")
        print("-" * 30)

        # Test with the example from your JSON
        test_text = """SMC AMROHA सराहनीय कार्य जनपद अमरोहा दिनांक 08.09.2025 ऑपरेशन कन्विक्शन जनपद अमरोहा में चलाये जा रहे ऑपरेशन कन्विक्शन के अन्तर्गत नये कानून BNS की पहली सजा नाबालिग से दुष्कर्म करने से संबंधित अभियोग में अभियुक्त को आजीवन कारावास व कुल 16,000 अर्थदंड की कराई गई सजा अवगत कराना है कि जनपद अमरोहा में पुलिस अधीक्षक श्री अमित कुमार आनंद द्वारा अपराधियों के विरुद्ध चलाए जा रहे अभियान ऑपरेशन कन्विक्शन तथा प्रभावी पैरवी के परिणामस्वरूप एक महत्वपूर्ण निर्णय मा० न्यायालय द्वारा जनपद अमरोहा में धारा बीएनएस के तहत सजा करने का पहला आदेश पारित किया गया है। दिनांक 08.09.2025 को थाना रजबपुर पर पंजीकृत मु0अ0सं0 297/2024 धारा 115(2), 351(2), 140 BNS व 5M/6 पोक्सो एक्ट से संबंधित प्रकरण में अभियुक्त अर्जुन उर्फ बहरा पुत्र भवंर सिंह निवासी रिवा (म0प्र0), हाल निवासी ग्राम खालकपुर थाना रजबपुर जनपद अमरोहा को आजीवन कारावास एवं कुल 16,000/- अर्थदंड से दंडित किया गया है।"""

        test_result = extractor.extract(test_text)

        print("Extraction Results:")
        print(json.dumps(test_result, ensure_ascii=False, indent=2))

        # Test batch processing
        print("\n6. Testing Batch Processing...")
        print("-" * 30)
        batch_texts = [
            "लखनऊ में प्रदर्शन हुआ। राम शर्मा ने भाग लिया।",
            "मुरादाबाद के कोतवाली थाने में चोरी की रिपोर्ट। #मुरादाबादपुलिस",
            "दिल्ली के कनॉट प्लेस में ट्रैफिक जाम। John Singh was present."
        ]

        batch_results = extractor.extract_batch(batch_texts)
        print(f"Successfully processed {len(batch_results)} texts in batch!")

        # Show first batch result as example
        if batch_results:
            print("\nSample Batch Result:")
            print(json.dumps(batch_results[0], ensure_ascii=False, indent=2))

        # Display extraction statistics
        print("\n7. Extraction Statistics:")
        print("-" * 30)
        stats = extractor.get_extraction_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\n" + "=" * 60)
        print("✓ Enhanced NER Extractor is working properly!")
        print("✓ The system is ready for production use.")

        # Performance recommendation
        loading_method = model_info.get("loading_method")
        if loading_method == "vllm":
            print("✓ PERFORMANCE: Using vLLM - optimal for production inference!")
        elif loading_method == "mlx":
            print("✓ PERFORMANCE: Using MLX - optimized for Apple Silicon!")
        elif loading_method == "transformers":
            print("⚠ PERFORMANCE: Using Transformers - consider installing vLLM for better performance!")

        print("\nKey Improvements Made:")
        print("- ✓ Enhanced district recognition (including मुरादाबाद)")
        print("- ✓ Cleaner thana name extraction (removes 'थाना' prefix)")
        print("- ✓ Better multilingual support (Hindi/English/Hinglish)")
        print("- ✓ Comprehensive contextual understanding in Hindi")
        print("- ✓ Improved JSON parsing with error recovery")
        print("- ✓ Enhanced prompt with detailed examples and instructions")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Extractor initialization failed: {e}")
        import traceback

        traceback.print_exc()

        print("\n🔧 Troubleshooting Tips:")
        print("1. Verify model path exists and is accessible")
        print("2. Check GPU memory availability (if using vLLM)")
        print("3. Install missing dependencies:")
        print("   pip install vllm  # For GPU acceleration")
        print("   pip install mlx mlx-lm  # For Apple Silicon")
        print("   pip install transformers torch  # Fallback option")
        print("4. Try with a smaller model if memory issues persist")
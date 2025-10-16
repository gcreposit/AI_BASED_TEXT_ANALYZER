import datetime
import os
import re
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from district_normalizer import DistrictNormalizer
from sentiment_analyzer import AdvancedSentimentAnalyzer
from datetime import datetime, timedelta
import traceback

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
            "कासगंज", "मैनपुरी", "इटावा", "फतेहपुर", "कौशांबी", "संभल", "बाराबंकी", "गौतम बुद्ध नगर", "कानपुर देहात", "शामली",
            "इलाहाबाद", "प्रयागराज", "फैज़ाबाद", "अयोध्या", "ज्योतिबा फुले नगर", "अमरोहा", "पंचशील नगर", "हापुड़",
            "भीम नगर", "संभल", "संत रविदास नगर", "भदोही"


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
            "Kasganj", "Mainpuri", "Etawah", "Fatehpur", "Kaushambi", "Sambhal", "Barabanki", "Gautam Buddha Nagar", "Kanpur Dehat", "Shamli",
            "Allahabad", "Prayagraj", "Faizabad", "Ayodhya", "J. P. Nagar", "Amroha", "Panchsheel Nagar", "Hapur",
            "Bhim Nagar", "Sambhal", "Sant Ravidas Nagar", "Bhadohi"

        ]

        # Enhanced thana patterns - more flexible
        self.thana_patterns = [
            r"(?:थाना|कोतवाली|पुलिस\s*स्टेशन|PS|Police\s*Station|Kotwali|Thana)\s+([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\s]*?)(?:\s+(?:में|पर|जनपद|थाना|district)|[,।\n]|$)",
            r"थाना\s+([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\s]*?)(?:\s+(?:पर|में)|[,।\n]|$)",
            r"कोतवाली\s+([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\s]*?)(?:\s+(?:पर|में)|[,।\n]|$)"
        ]

        # NEW: Category classification system
        self.category_keywords = {
            "CRIME": {
                "MURDER": [
                    "हत्या", "murder", "खून", "killing", "मारना", "जान से मारना", "हत्या का मामला",
                    "मृत्यु", "death", "मौत", "killed", "मारा गया", "मरा हुआ", "लाश", "शव"
                ],
                "AGAINST WOMEN": [
                    "महिलाओं के विरुद्ध", "against women", "दुष्कर्म", "rape", "छेड़छाड़", "harassment",
                    "बलात्कार", "यौन उत्पीड़न", "sexual assault", "महिला हिंसा", "दहेज", "dowry"
                ],
                "COMMUNAL": [
                    "सांप्रदायिक", "communal", "धर्म", "religion", "दंगा", "riots", "हिंसा",
                    "सांप्रदायिक तनाव", "communal tension", "religious violence", "धार्मिक हिंसा"
                ],
                "CASTEISM": [
                    "जातिवाद", "casteism", "जाति", "caste", "दलित", "अत्याचार", "atrocity",
                    "जातीय हिंसा", "caste violence", "अनुसूचित जाति"
                ],
                "AGAINST MINORS": [
                    "नाबालिग", "minor", "बच्चों के विरुद्ध", "child abuse", "पोक्सो", "POCSO",
                    "बाल यौन शोषण", "child sexual abuse", "नाबालिग से दुष्कर्म"
                ],
                "LOVE JIHAAD": [
                    "लव जिहाद", "love jihad", "धर्म परिवर्तन", "forced conversion",
                    "प्रेम प्रसंग", "interfaith marriage"
                ],
                "AGAINST COW": [
                    "गौ हत्या", "cow slaughter", "गाय", "cow", "गौ रक्षा", "cow protection",
                    "गौ तस्करी", "cattle smuggling"
                ],
                "ROBBERY": [
                    "लूट", "robbery", "डकैती", "dacoity", "चोरी", "theft", "सेंधमारी",
                    "burglary", "लूटपाट", "सम्पत्ति अपराध"
                ],
                "LOOT": ["लूटपाट", "looting", "सामान लूटना", "property theft", "धन लूटना"],
                "THEFT": ["चोरी", "theft", "सेंधमारी", "burglary", "चुराना", "stealing"],
                "KIDNAPPING": ["अपहरण", "kidnapping", "गुमशुदगी", "missing", "बंधक", "hostage"],
                "ASSAULT": ["मारपीट", "assault", "हमला", "attack", "पिटाई", "beating", "शारीरिक हिंसा"],
                "AGAINST ANIMAL": ["पशु हिंसा", "animal cruelty", "पशुओं के विरुद्ध", "against animals"],
                "PETA": ["पेटा", "PETA", "पशु अधिकार", "animal rights"]
            },
            "TRAFFIC RELATED": {
                "TRAFFIC JAM": ["ट्रैफिक जाम", "traffic jam", "यातायात बाधा", "traffic congestion"],
                "DIVERSION": ["मार्ग परिवर्तन", "traffic diversion", "रास्ता बंद", "road closure"],
                "TRAFFIC RULES VIOLATION": ["ट्रैफिक नियम उल्लंघन", "traffic violation", "चालान", "fine"]
            },
            "RAILWAY RELATED": {
                "INVOLVING RAILWAYS": ["रेलवे", "railway", "ट्रेन", "train", "स्टेशन", "station"]
            },
            "POLICE MISCONDUCT": {
                "CORRUPTION": ["भ्रष्टाचार", "corruption", "रिश्वत", "bribe", "घूसखोरी"],
                "MISCONDUCT": ["कदाचार", "misconduct", "पुलिस कदाचार", "police misconduct"]
            },
            "GRIEVANCE": {
                "EMERGENCY": ["आपातकाल", "emergency", "आपातकालीन सेवा", "emergency service"],
                "FIR RELATED": ["एफआईआर", "FIR", "प्राथमिकी", "complaint", "शिकायत"],
                "COMPLAINTS": ["शिकायत", "complaint", "निवेदन", "grievance", "समस्या"],
                "OFFICIAL SERVICE RELATED": ["सरकारी सेवा", "government service", "अधिकारी", "official"],
                "FIRE RELATED": ["आग", "fire", "अग्निकांड", "fire incident"],
                "ACCIDENT": ["दुर्घटना", "accident", "हादसा", "mishap"]
            },
            "CYBER CRIME": {
                "HACKING": ["हैकिंग", "hacking", "साइबर अपराध", "cyber crime"],
                "PHISING": ["फिशिंग", "phishing", "धोखाधड़ी", "online fraud"],
                "DIGITAL ARREST": ["डिजिटल गिरफ्तारी", "digital arrest", "ऑनलाइन धोखाधड़ी"],
                "MONEY FRAUD": ["पैसे की धोखाधड़ी", "money fraud", "वित्तीय धोखाधड़ी"],
                "EXPLICIT CONTENT": ["अश्लील सामग्री", "explicit content", "पोर्न", "obscene"],
                "DIGITAL RANSOMI": ["डिजिटल फिरौती", "digital ransom", "रैंसमवेयर"]
            },
            "HATE SPEECH": {
                "AGAINST RELIGION": ["धर्म विरोधी", "against religion", "धार्मिक घृणा", "religious hate"],
                "AGAINST CASTEISM": ["जाति विरोधी", "against caste", "जातीय घृणा", "caste hate"],
                "POLITICALLY MOTIVATED": ["राजनीतिक प्रेरित", "politically motivated", "राजनीतिक हिंसा"]
            },
            "VIRAL & FACT CHECK": {
                "VIRAL": ["वायरल", "viral", "सोशल मीडिया", "social media"],
                "FAKE NEWS": ["फेक न्यूज", "fake news", "झूठी खबर", "false news"],
                "RUMOURS": ["अफवाह", "rumours", "गलत जानकारी", "misinformation"]
            },
            "ELECTION": {
                "BOOTH CAPTURING": ["बूथ कैप्चरिंग", "booth capturing", "मतदान केंद्र पर कब्जा"],
                "MCC VIOLATIONS": ["आचार संहिता उल्लंघन", "MCC violation", "election code violation"],
                "FAKE VOTING": ["फर्जी मतदान", "fake voting", "बोगस वोटिंग"],
                "BOOTH FACILITIES RELATED ISSUE": ["मतदान सुविधा", "voting facility", "बूथ सुविधा"],
                "BOOTH MACHINE RELATED ISSUES": ["ईवीएम", "EVM", "मशीन खराब", "machine problem"],
                "COMPLAIN AGAINST BOOTH OFFICIALS": ["बूथ अधिकारी शिकायत", "booth official complaint"],
                "HINDRANCE IN ELECTION SERVICES": ["चुनाव में बाधा", "election hindrance", "मतदान में रुकावट"],
                "ELECTORAL DISPUTES": ["चुनावी विवाद", "electoral dispute", "मतदान विवाद"]
            },
            "LAW & ORDER": {
                "PROTEST": ["प्रदर्शन", "protest", "धरना", "demonstration", "आंदोलन"],
                "MOVEMENTS": ["आंदोलन", "movement", "अभियान", "campaign"],
                "CROWD SUMMON": ["भीड़ एकत्रित", "crowd gathering", "जमावड़ा", "assembly"],
                "ANTI NATIONAL ACTIVITIES": ["राष्ट्र विरोधी", "anti national", "देशद्रोह", "sedition"],
                "TERRORIST RELATED": ["आतंकवाद", "terrorism", "आतंकी", "terrorist"],
                "DISASTER RELATED": ["आपदा", "disaster", "प्राकृतिक आपदा", "natural disaster"]
            },
            "ANTI NARCOTICS": {
                "ANTI NARCOTICS": ["नशा निवारण", "anti narcotics", "ड्रग्स", "drugs", "नशीले पदार्थ"]
            },
            "FESTIVALS": {
                "HINDU RELATED": ["हिंदू त्योहार", "hindu festival", "दिवाली", "होली", "दशहरा"],
                "MUSLIM RELATED": ["मुस्लिम त्योहार", "muslim festival", "ईद", "रमजान", "मुहर्रम"],
                "GOVERNMENT INITIATIVES": ["सरकारी पहल", "government initiative", "योजना", "scheme"]
            }
        }

        # Enhanced JSON schema with MULTIPLE category fields
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
            "contextual_understanding": "",
            "incident_location_analysis": {
                "incident_districts": [],
                "related_districts": [],
                "incident_thanas": [],
                "related_thanas": [],
                "primary_location": {
                    "district": "",
                    "thana": "",
                    "specific_location": ""
                }
            },
            "temporal_info": {
                "incident_date": "",  # YYYY-MM-DD format
                "incident_time": "",  # Time if mentioned
                "temporal_phrase": "",  # Original phrase
                "temporal_type": "",  # "absolute" or "relative"
                "confidence": 0.0,
                "days_ago": None  # Days from current date
            },
            "advanced_sentiment": {
                "overall_stance": "neutral",  # pro/against/neutral
                "overall_confidence": 0.0,
                "pro_towards": {
                    "castes": [],
                    "religions": [],
                    "organisations": [],
                    "political_parties": [],
                    "other_aspects": []
                },
                "against_towards": {
                    "castes": [],
                    "religions": [],
                    "organisations": [],
                    "political_parties": [],
                    "other_aspects": []
                },
                "neutral_towards": {
                    "castes": [],
                    "religions": [],
                    "organisations": [],
                    "political_parties": [],
                    "other_aspects": []
                },
                "analysis_method": "",
                "reasoning": ""
            }
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
                            max_model_len=32768,
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
        """
        Enhanced keyword finding with strict handling for acronyms SC/ST/OBC.

        - Matches SC, ST, OBC only as standalone tokens (optionally prefixed by '#'),
          surrounded by non-letters. This prevents false positives like 'IST' -> 'ST'.
        - For all other keywords, uses case-insensitive whole-word matching with Unicode support.
        """
        found = set()
        text_lower = text.lower()

        # Special-case acronyms: treat them as strict tokens (avoid matching inside words like 'IST')
        ACRONYMS = {"SC", "ST", "OBC"}

        for keyword in vocabulary:
            # --- Strict handling for SC/ST/OBC ---
            if keyword.upper() in ACRONYMS:
                token = keyword.upper()
                # Optional leading '#', and MUST NOT be surrounded by letters (A-Z only) to avoid 'IST', 'CASTE', etc.
                # Examples matched: "SC", "#SC", "(SC)", "SC/ST", "OBC,", "—ST", " ST "
                # Examples NOT matched: "IST", "CASTE", "obc123x" (letter on either side blocks it)
                pattern = rf'(?<![A-Za-z])#?{token}(?![A-Za-z])'
                if re.search(pattern, text):
                    # return canonical acronym casing
                    found.add(token)
                continue

            # --- General keywords (Hindi/English phrases, full words) ---
            kw = keyword
            kw_lower = kw.lower()

            # First try fast contains check (kept from your original) but still enforce word boundaries
            if kw in text or kw_lower in text_lower:
                # Use Unicode-safe word boundary: (?<!\w) and (?!\w) with IGNORECASE
                if re.search(rf'(?<!\w){re.escape(kw)}(?!\w)', text, re.IGNORECASE):
                    found.add(kw)
                    continue

            # If not caught above, try explicit lower-case boundary match
            if re.search(rf'(?<!\w){re.escape(kw_lower)}(?!\w)', text_lower, re.IGNORECASE):
                found.add(kw)

        return self._dedupe(list(found))

    def _has_devanagari(self, text: str) -> bool:
        """Check if the text contains Devanagari characters."""
        devanagari_pattern = re.compile(r'[\u0900-\u097F]')  # Unicode range for Devanagari characters
        return bool(devanagari_pattern.search(text))

    def _extract_temporal_info(self, text: str) -> Dict[str, Any]:
        """Extract temporal information from text"""

        result = {
            "incident_date": "",
            "incident_time": "",
            "temporal_phrase": "",
            "temporal_type": "",
            "confidence": 0.0,
            "days_ago": None
        }

        text_lower = text.lower()

        # Relative dates
        relative_dates = {
            'आज': 0, 'today': 0,
            'कल': 1, 'yesterday': 1, 'कल रात': 1,
            'परसों': 2, 'day before yesterday': 2,
            'बीते': 1, 'last': 1, 'गत': 1, 'पिछले': 1,
        }

        # Check for relative dates
        for phrase, days_ago in relative_dates.items():
            if phrase in text_lower:
                result['temporal_phrase'] = phrase
                result['temporal_type'] = 'relative'
                result['days_ago'] = days_ago
                incident_date = datetime.now() - timedelta(days=days_ago)
                result['incident_date'] = incident_date.strftime('%Y-%m-%d')
                result['confidence'] = 0.8
                break

        # Check for absolute dates if no relative date found
        if not result['incident_date']:
            date_patterns = [
                r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
                r'दिनांक\s*(\d{1,2}[./]\d{1,2}[./]\d{2,4})',
            ]

            for pattern in date_patterns:
                match = re.search(pattern, text)
                if match:
                    date_str = match.group(1) if match.lastindex else match.group(0)
                    result['temporal_phrase'] = date_str
                    result['temporal_type'] = 'absolute'

                    # Parse date
                    for fmt in ['%d-%m-%Y', '%d/%m/%Y', '%d.%m.%Y', '%d-%m-%y', '%d/%m/%y']:
                        try:
                            parsed_date = datetime.strptime(date_str.replace('.', '/').replace('-', '/'), fmt)
                            result['incident_date'] = parsed_date.strftime('%Y-%m-%d')
                            result['days_ago'] = (datetime.now() - parsed_date).days
                            result['confidence'] = 0.9
                            break
                        except:
                            continue
                    break

        # ✅ MODIFIED: If no date found, return EMPTY instead of assuming
        if not result['incident_date']:
            result['temporal_phrase'] = ""  # ❌ NOT "हाल ही में"
            result['temporal_type'] = "not_provided"  # ✅ Clear indicator
            result['days_ago'] = None  # ❌ NOT 3
            result['incident_date'] = ""  # ❌ NOT fake date
            result['confidence'] = 0.0

        return result

    def _classify_multiple_categories(self, text: str, extracted_entities: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced MULTI-category classification with intersectional analysis"""
        text_lower = text.lower()

        # Store all potential matches with scores
        all_matches = []

        # Extract caste-related indicators for intersectional analysis
        caste_indicators = extracted_entities.get("caste_names", []) or self._find_keywords(text, self.castes)
        is_caste_related = len(caste_indicators) > 0

        # Extract gender indicators
        gender_indicators = ["महिला", "लड़की", "औरत", "woman", "girl", "female", "बेटी", "पत्नी", "wife"]
        is_gender_related = any(indicator in text_lower for indicator in gender_indicators)

        # Extract minor indicators
        minor_indicators = ["नाबालिग", "minor", "बच्चा", "बच्ची", "child", "छोटा", "छोटी", "किशोर", "किशोरी"]
        is_minor_related = any(indicator in text_lower for indicator in minor_indicators)

        for broad_category, subcategories in self.category_keywords.items():
            for sub_category, keywords in subcategories.items():
                matched_keywords = []
                base_score = 0

                # Calculate base score from keyword matching
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    if keyword in text or keyword_lower in text_lower:
                        matched_keywords.append(keyword)
                        base_score += 2
                    elif re.search(r'\b' + re.escape(keyword_lower) + r'\b', text_lower):
                        matched_keywords.append(keyword)
                        base_score += 1

                # Apply intersectional bonuses
                intersectional_score = base_score
                reasoning_parts = []

                # If base keywords match, check for intersectional aspects
                if base_score > 0:
                    reasoning_parts.append(f"Direct keyword matches: {len(matched_keywords)}")

                    # INTERSECTIONAL ANALYSIS
                    if sub_category == "AGAINST MINORS" and is_minor_related:
                        intersectional_score += 3
                        reasoning_parts.append("Minor-related indicators found")

                    if sub_category == "AGAINST WOMEN" and is_gender_related:
                        intersectional_score += 3
                        reasoning_parts.append("Gender-related indicators found")

                    if sub_category == "CASTEISM" and is_caste_related:
                        intersectional_score += 3
                        reasoning_parts.append("Caste-related indicators found")

                    # Cross-category intersections
                    if (sub_category in ["AGAINST MINORS", "AGAINST WOMEN", "CASTEISM"] and
                            is_minor_related and is_gender_related and is_caste_related):
                        intersectional_score += 2  # Triple intersection bonus
                        reasoning_parts.append("Triple intersection: caste + gender + minor")

                    elif ((sub_category == "AGAINST MINORS" and is_gender_related) or
                          (sub_category == "AGAINST WOMEN" and is_minor_related)):
                        intersectional_score += 1  # Gender-minor intersection
                        reasoning_parts.append("Gender-minor intersection")

                    elif ((sub_category == "CASTEISM" and is_gender_related) or
                          (sub_category == "AGAINST WOMEN" and is_caste_related)):
                        intersectional_score += 1  # Caste-gender intersection
                        reasoning_parts.append("Caste-gender intersection")

                    elif ((sub_category == "CASTEISM" and is_minor_related) or
                          (sub_category == "AGAINST MINORS" and is_caste_related)):
                        intersectional_score += 1  # Caste-minor intersection
                        reasoning_parts.append("Caste-minor intersection")

                # Boost for specific incident matches
                incidents_text = " ".join(extracted_entities.get("incidents", []))
                if "पोक्सो" in incidents_text.lower() or "pocso" in incidents_text.lower():
                    if sub_category == "AGAINST MINORS":
                        intersectional_score += 4
                        reasoning_parts.append("POCSO Act mentioned")

                if "दुष्कर्म" in text_lower or "rape" in text_lower:
                    if sub_category in ["AGAINST WOMEN", "AGAINST MINORS"]:
                        intersectional_score += 3
                        reasoning_parts.append("Sexual assault indicators")

                # Store match if it has any score
                if intersectional_score > 0:
                    confidence = min(0.95, intersectional_score / (len(keywords) + 5))  # Normalize confidence

                    match_data = {
                        "broad_category": broad_category,
                        "sub_category": sub_category,
                        "confidence": confidence,
                        "matched_keywords": matched_keywords[:5],
                        "reasoning": " | ".join(
                            reasoning_parts) if reasoning_parts else f"Keywords matched for {sub_category}",
                        "score": intersectional_score
                    }
                    all_matches.append(match_data)

        # Sort by score (descending) and filter meaningful matches
        all_matches.sort(key=lambda x: x["score"], reverse=True)

        # Define thresholds
        MIN_SCORE_THRESHOLD = 1  # Minimum score to be considered
        CONFIDENCE_THRESHOLD = 0.1  # Minimum confidence to include

        # Filter and prepare final results
        valid_matches = [
            match for match in all_matches
            if match["score"] >= MIN_SCORE_THRESHOLD and match["confidence"] >= CONFIDENCE_THRESHOLD
        ]

        # Remove score from final results (internal use only)
        for match in valid_matches:
            del match["score"]

        # Determine primary classification (highest scoring)
        primary_classification = valid_matches[0] if valid_matches else {
            "broad_category": "UNCLASSIFIED",
            "sub_category": "GENERAL",
            "confidence": 0.0
        }

        # Limit to top 5 classifications to avoid noise
        final_classifications = valid_matches[:5] if len(valid_matches) > 5 else valid_matches

        return {
            "category_classifications": final_classifications,
            "primary_classification": {
                "broad_category": primary_classification["broad_category"],
                "sub_category": primary_classification["sub_category"],
                "confidence": primary_classification["confidence"]
            }
        }

    def _analyze_location_context(self, text: str, districts: List[str], thanas: List[str]) -> Dict[str, Any]:
        """Enhanced location analysis to separate incident vs related locations"""

        # Keywords that indicate incident location
        incident_indicators = [
            "में हुई", "में घटना", "में मामला", "पर दर्ज", "में पंजीकृत", "में गिरफ्तार",
            "incident in", "case registered in", "arrested in", "happened in", "occurred in"
        ]

        # Keywords that indicate related/mentioned locations
        related_indicators = [
            "निवासी", "रहने वाला", "का रहने वाला", "से आया", "से संबंधित", "का मूल निवासी",
            "resident of", "belongs to", "native of", "from", "originally from"
        ]

        result = {
            "incident_districts": [],
            "related_districts": [],
            "incident_thanas": [],
            "related_thanas": [],
            "primary_location": {
                "district": "",
                "thana": "",
                "specific_location": ""
            }
        }

        text_lower = text.lower()

        # Analyze districts
        for district in districts:
            is_incident_location = False
            is_related_location = False

            # Find district mentions in context
            district_pattern = re.escape(district.lower())

            # Check for incident indicators before/after district name
            for indicator in incident_indicators:
                if re.search(f"{indicator}.*?{district_pattern}|{district_pattern}.*?{indicator}", text_lower):
                    is_incident_location = True
                    break

            # Check for related indicators
            if not is_incident_location:
                for indicator in related_indicators:
                    if re.search(f"{indicator}.*?{district_pattern}|{district_pattern}.*?{indicator}", text_lower):
                        is_related_location = True
                        break

            # Classify based on analysis
            if is_incident_location:
                result["incident_districts"].append(district)
                if not result["primary_location"]["district"]:
                    result["primary_location"]["district"] = district
            elif is_related_location:
                result["related_districts"].append(district)
            else:
                # Default: if mentioned early in text, likely incident location
                district_position = text_lower.find(district.lower())
                text_length = len(text)
                if district_position != -1 and district_position < (text_length * 0.3):
                    result["incident_districts"].append(district)
                    if not result["primary_location"]["district"]:
                        result["primary_location"]["district"] = district
                else:
                    result["related_districts"].append(district)

        # Analyze thanas similarly
        for thana in thanas:
            is_incident_location = False
            is_related_location = False

            thana_pattern = re.escape(thana.lower())

            # Thana-specific incident indicators
            thana_incident_indicators = incident_indicators + [
                "थाना", "कोतवाली", "पुलिस स्टेशन", "police station", "PS"
            ]

            for indicator in thana_incident_indicators:
                if re.search(f"{indicator}.*?{thana_pattern}|{thana_pattern}.*?{indicator}", text_lower):
                    is_incident_location = True
                    break

            if not is_incident_location:
                for indicator in related_indicators:
                    if re.search(f"{indicator}.*?{thana_pattern}|{thana_pattern}.*?{indicator}", text_lower):
                        is_related_location = True
                        break

            if is_incident_location:
                result["incident_thanas"].append(thana)
                if not result["primary_location"]["thana"]:
                    result["primary_location"]["thana"] = thana
            elif is_related_location:
                result["related_thanas"].append(thana)
            else:
                # Default classification
                thana_position = text_lower.find(thana.lower())
                if thana_position != -1 and thana_position < (len(text) * 0.3):
                    result["incident_thanas"].append(thana)
                    if not result["primary_location"]["thana"]:
                        result["primary_location"]["thana"] = thana
                else:
                    result["related_thanas"].append(thana)

        # Extract specific location (village, area, etc.)
        location_patterns = [
            r"ग्राम\s+([A-Za-z\u0900-\u097F]+)",
            r"village\s+([A-Za-z]+)",
            r"मोहल्ला\s+([A-Za-z\u0900-\u097F]+)",
            r"colony\s+([A-Za-z\s]+)",
            r"नगर\s+([A-Za-z\u0900-\u097F]+)"
        ]

        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not result["primary_location"]["specific_location"]:
                result["primary_location"]["specific_location"] = match.group(1).strip()
                break

        return result

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
5. Analyze location context to separate incident vs related locations

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

temporal_info:
- incident_date: Extract date in YYYY-MM-DD format (e.g., "2025-01-15")
- incident_time: Extract time if mentioned (e.g., "10:30 PM", "रात 9 बजे")
- temporal_phrase: Original temporal phrase from text (e.g., "कल रात", "08.09.2025", "yesterday")
- temporal_type: "absolute" (specific date) or "relative" (yesterday, last week)
- confidence: 0.0-1.0 based on clarity of temporal information
- days_ago: Number of days from current date (null if future/unclear)

TEMPORAL EXTRACTION RULES:
- "आज" / "today" → days_ago: 0
- "कल" / "yesterday" → days_ago: 1
- "परसों" → days_ago: 2
- "दिनांक 08.09.2025" → Parse to YYYY-MM-DD format
- If no temporal info found, set temporal_type: "NOT PROVIDED", days_ago: 0

TEMPORAL EXAMPLES:
✅ "कल रात घटना हुई" → {{"temporal_phrase": "कल रात", "temporal_type": "relative", "days_ago": 1, "incident_date": "{(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')}"}}
✅ "दिनांक 08.09.2025 को" → {{"temporal_phrase": "08.09.2025", "temporal_type": "absolute", "incident_date": "2025-09-08", "days_ago": ...}}
✅ "बीती रात" → {{"temporal_phrase": "बीती रात", "temporal_type": "relative", "days_ago": 1}}

advanced_sentiment:
- overall_stance: "pro", "against", or "neutral"
- overall_confidence: 0.0-1.0
- pro_towards: {{"castes": [], "religions": [], "organisations": [], "political_parties": [], "other_aspects": []}}
- against_towards: {{"castes": [], "religions": [], "organisations": [], "political_parties": [], "other_aspects": []}}
- reasoning: Brief explanation of sentiment analysis

SENTIMENT ANALYSIS RULES:
- Identify if text is supportive (PRO), opposed (AGAINST), or neutral towards entities
- Look for keywords: समर्थन, पक्ष (pro) vs विरोध, खिलाफ (against)
- Check context around caste/religion/organization mentions
- Political parties: BJP, Congress, SP, BSP, etc.

SENTIMENT EXAMPLES:
✅ "दलितों के साथ अन्याय हुआ" → against_towards.castes: ["दलित"]
✅ "मुस्लिम समुदाय का समर्थन" → pro_towards.religions: ["मुस्लिम"]
✅ "सरकार की आलोचना" → against_towards.other_aspects: ["सरकार"]

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

        # NEW: Handle multiple category classifications
        # llm_classifications = llm_json.get("category_classifications", [])
        # llm_primary = llm_json.get("primary_classification", {})
        #
        # if isinstance(llm_classifications, list) and llm_classifications:
        #     # Use LLM classifications if available
        #     result["category_classifications"] = llm_classifications
        #     result["primary_classification"] = llm_primary if llm_primary else llm_classifications[0]
        # else:
        #     # Fallback: use rule-based multi-classification
        #     multi_classification = self._classify_multiple_categories(
        #         regex_extractions.get("original_text", ""), result
        #     )
        #     result["category_classifications"] = multi_classification["category_classifications"]
        #     result["primary_classification"] = multi_classification["primary_classification"]

        # ✅ ADD THIS AFTER THE COMMENTED SECTION:
        # Add empty placeholders for categories (will be filled by keyword classifier later)
        result["category_classifications"] = []
        result["primary_classification"] = {
            "broad_category": "",
            "sub_category": "",
            "confidence": 0.0
        }

        # NEW: Handle location analysis
        llm_location = llm_json.get("incident_location_analysis", {})
        if isinstance(llm_location, dict) and (
                llm_location.get("incident_districts") or llm_location.get("incident_thanas")):
            result["incident_location_analysis"] = llm_location
        else:
            # Fallback: use rule-based location analysis
            result["incident_location_analysis"] = self._analyze_location_context(
                regex_extractions.get("original_text", ""),
                result["district_names"],
                result["thana_names"]
            )

        return result

    """
    Fixed NER Extractor - extract() and extract_batch() now consistent
    Both methods include:
    - Temporal extraction
    - Advanced sentiment analysis
    """

    def extract(self, text: str, max_tokens: int = 1500, temperature: float = 0.1) -> Dict[str, Any]:
        """Enhanced extraction with better error handling"""
        if not text or not text.strip():
            logger.warning("Empty text provided for extraction")
            return dict(self.json_schema)

        start_time = time.time()

        try:
            # ========== STEP 1: Regex Extractions ==========
            regex_extractions = {
                "hashtags": self._find_hashtags(text),
                "mention_ids": self._find_mentions(text),
                "district_names": self._find_districts(text),
                "thana_names": self._find_thana(text),
                "caste_names": self._find_keywords(text, self.castes),
                "religion_names": self._find_keywords(text, self.religions),
                "original_text": text
            }

            logger.debug("🔍 STEP 1 COMPLETED - Regex extractions done")

            # ========== STEP 2: LLM Extraction ==========
            llm_json = {}

            if self._model_loaded and self.model is not None:
                prompt = self._build_enhanced_instruction_prompt(text)

                try:
                    raw_response = ""

                    # vLLM
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

                    # MLX
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

                    # Transformers
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
                            logger.info("✅ STEP 2 COMPLETED - LLM extraction successful")
                        else:
                            logger.warning("LLM returned empty or invalid JSON")
                    else:
                        logger.info("No LLM response available")

                except Exception as e:
                    logger.error(f"LLM generation failed: {e}")

            # ========== STEP 3: Merge Results ==========
            final_result = self._merge_results(llm_json, regex_extractions)
            logger.debug("✅ STEP 3 COMPLETED - Results merged")

            # ========== STEP 4: Temporal Extraction ==========
            if 'temporal_info' not in final_result or not final_result.get('temporal_info', {}).get('incident_date'):
                temporal_info = self._extract_temporal_info(text)
                final_result['temporal_info'] = temporal_info

            logger.info("✅ STEP 4 COMPLETED - Temporal extraction done")

            # ========== STEP 5: Advanced Sentiment Analysis ==========
            from sentiment_analyzer import AdvancedSentimentAnalyzer

            sentiment_analyzer = AdvancedSentimentAnalyzer(
                llm_model=self.model,
                llm_tokenizer=self.tokenizer,
                loading_method=self._loading_method
            )

            advanced_sentiment = sentiment_analyzer.analyze_advanced_sentiment(text, final_result)
            final_result['advanced_sentiment'] = advanced_sentiment

            logger.info("✅ STEP 5 COMPLETED - Advanced sentiment analysis done")

            processing_time = (time.time() - start_time) * 1000
            logger.info(f"✅ EXTRACT COMPLETE in {processing_time:.2f}ms")

            return final_result

        except Exception as e:
            logger.exception("NER extraction failed")
            logger.error(f"NER extraction failed: {e}")
            return dict(self.json_schema)

    def extract_batch(self, texts: List[str], max_tokens: int = 1500, temperature: float = 0.1) -> List[Dict[str, Any]]:
        """
        ✅ FIXED: Extract entities from multiple texts with FULL pipeline
        Now includes temporal extraction and sentiment analysis for each text
        """
        if not self._model_loaded or self.model is None:
            logger.warning("Model not available for batch processing")
            return [dict(self.json_schema) for _ in texts]

        results = []
        total_start = time.time()

        # Import sentiment analyzer once
        from sentiment_analyzer import AdvancedSentimentAnalyzer

        sentiment_analyzer = AdvancedSentimentAnalyzer(
            llm_model=self.model,
            llm_tokenizer=self.tokenizer,
            loading_method=self._loading_method
        )

        # ========== vLLM BATCH PROCESSING ==========
        if self._loading_method == 'vllm' and VLLM_AVAILABLE:
            try:
                logger.info(f"🚀 Starting vLLM batch processing for {len(texts)} texts")

                # STEP 1: Build prompts
                prompts = [self._build_enhanced_instruction_prompt(text) for text in texts]

                # STEP 2: Batch LLM generation
                sampling_params = SamplingParams(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=[self.tokenizer.eos_token] if hasattr(self.tokenizer, 'eos_token') else None,
                    top_p=0.95,
                    frequency_penalty=0.1,
                    presence_penalty=0.1
                )

                outputs = self.model.generate(prompts, sampling_params)
                logger.info("✅ vLLM batch generation complete")

                # STEP 3: Process each output with FULL pipeline
                for i, output in enumerate(outputs):
                    try:
                        text = texts[i]
                        raw_response = output.outputs[0].text.strip()

                        # 3.1: Regex extractions
                        regex_extractions = {
                            "hashtags": self._find_hashtags(text),
                            "mention_ids": self._find_mentions(text),
                            "district_names": self._find_districts(text),
                            "thana_names": self._find_thana(text),
                            "caste_names": self._find_keywords(text, self.castes),
                            "religion_names": self._find_keywords(text, self.religions),
                        }

                        # 3.2: Parse LLM response
                        llm_json = self._safe_json_parse(raw_response) if raw_response else {}

                        # 3.3: Merge results
                        final_result = self._merge_results(llm_json, regex_extractions)

                        # ✅ 3.4: Temporal extraction (ADDED)
                        if 'temporal_info' not in final_result or not final_result.get('temporal_info', {}).get(
                                'incident_date'):
                            temporal_info = self._extract_temporal_info(text)
                            final_result['temporal_info'] = temporal_info

                        # ✅ 3.5: Advanced sentiment analysis (ADDED)
                        advanced_sentiment = sentiment_analyzer.analyze_advanced_sentiment(text, final_result)
                        final_result['advanced_sentiment'] = advanced_sentiment

                        results.append(final_result)

                        if (i + 1) % 10 == 0:
                            logger.info(f"  Processed {i + 1}/{len(texts)} texts")

                    except Exception as e:
                        logger.error(f"Failed to process batch item {i}: {e}")
                        results.append(dict(self.json_schema))

                total_time = time.time() - total_start
                logger.info(f"✅ vLLM batch extraction completed: {len(texts)} texts in {total_time:.2f}s")
                return results

            except Exception as e:
                logger.error(f"vLLM batch processing failed: {e}")
                # Fall through to sequential processing

        # ========== SEQUENTIAL PROCESSING (MLX/Transformers) ==========
        logger.info(f"🚀 Starting sequential batch processing for {len(texts)} texts")

        for i, text in enumerate(texts):
            try:
                # Call extract() which has the full pipeline
                result = self.extract(text, max_tokens, temperature)
                results.append(result)

                if (i + 1) % 10 == 0:
                    elapsed = time.time() - total_start
                    avg_time = elapsed / (i + 1)
                    eta = avg_time * (len(texts) - i - 1)
                    logger.info(f"  Processed {i + 1}/{len(texts)} texts in {elapsed:.2f}s (ETA: {eta:.2f}s)")

            except Exception as e:
                logger.error(f"Failed to process text {i + 1}: {e}")
                results.append(dict(self.json_schema))

        total_time = time.time() - total_start
        logger.info(f"✅ Sequential batch extraction completed: {len(texts)} texts in {total_time:.2f}s")
        return results

    # ... (rest of the methods remain the same but can be enhanced similarly)

    # def extract_batch(self, texts: List[str], max_tokens: int = 1500, temperature: float = 0.1) -> List[Dict[str, Any]]:
    #     """Extract entities from multiple texts efficiently"""
    #     if not self._model_loaded or self.model is None:
    #         logger.warning("Model not available for batch processing")
    #         return [dict(self.json_schema) for _ in texts]
    #
    #     results = []
    #     total_start = time.time()
    #
    #     # vLLM batch processing
    #     if self._loading_method == 'vllm' and VLLM_AVAILABLE:
    #         try:
    #             prompts = [self._build_enhanced_instruction_prompt(text) for text in texts]
    #
    #             sampling_params = SamplingParams(
    #                 temperature=temperature,
    #                 max_tokens=max_tokens,
    #                 stop=[self.tokenizer.eos_token] if hasattr(self.tokenizer, 'eos_token') else None,
    #                 top_p=0.95,
    #                 frequency_penalty=0.1,
    #                 presence_penalty=0.1
    #             )
    #
    #             outputs = self.model.generate(prompts, sampling_params)
    #
    #             for i, output in enumerate(outputs):
    #                 try:
    #                     text = texts[i]
    #                     raw_response = output.outputs[0].text.strip()
    #
    #                     # Regex extractions
    #                     regex_extractions = {
    #                         "hashtags": self._find_hashtags(text),
    #                         "mention_ids": self._find_mentions(text),
    #                         "district_names": self._find_districts(text),
    #                         "thana_names": self._find_thana(text),
    #                         "caste_names": self._find_keywords(text, self.castes),
    #                         "religion_names": self._find_keywords(text, self.religions),
    #                     }
    #
    #                     # Parse LLM response
    #                     llm_json = self._safe_json_parse(raw_response) if raw_response else {}
    #
    #                     # Merge and add result
    #                     final_result = self._merge_results(llm_json, regex_extractions)
    #                     results.append(final_result)
    #
    #                 except Exception as e:
    #                     logger.error(f"Failed to process batch item {i}: {e}")
    #                     results.append(dict(self.json_schema))
    #
    #             total_time = time.time() - total_start
    #             logger.info(f"vLLM batch extraction completed: {len(texts)} texts in {total_time:.2f}s")
    #             return results
    #
    #         except Exception as e:
    #             logger.error(f"vLLM batch processing failed: {e}")
    #
    #     # Sequential processing for MLX and Transformers
    #     for i, text in enumerate(texts):
    #         try:
    #             result = self.extract(text, max_tokens, temperature)
    #             results.append(result)
    #
    #             if (i + 1) % 10 == 0:
    #                 elapsed = time.time() - total_start
    #                 avg_time = elapsed / (i + 1)
    #                 eta = avg_time * (len(texts) - i - 1)
    #                 logger.info(f"Processed {i + 1}/{len(texts)} texts in {elapsed:.2f}s (ETA: {eta:.2f}s)")
    #
    #         except Exception as e:
    #             logger.error(f"Failed to process text {i + 1}: {e}")
    #             results.append(dict(self.json_schema))
    #
    #     total_time = time.time() - total_start
    #     logger.info(f"Sequential batch extraction completed: {len(texts)} texts in {total_time:.2f}s")
    #     return results

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
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "total_broad_categories": len(self.category_keywords),
            "total_subcategories": sum(len(subs) for subs in self.category_keywords.values()),
            "intersectional_analysis": True
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


def test_multi_classification(self, test_cases: List[str] = None) -> Dict[str, Any]:
    """Test the multi-classification system with various intersectional cases"""
    if test_cases is None:
        test_cases = [
            "अनुसूचित जाति की नाबालिग लड़की से दुष्कर्म का मामला दर्ज",
            "दलित बच्चे की हत्या, पोक्सो एक्ट लगाया गया",
            "मुस्लिम महिला पर जातीय हमला",
            "ट्रैफिक जाम के कारण रास्ता बंद",
            "पुलिस अधिकारी पर भ्रष्टाचार का आरोप"
        ]

    results = {}

    for i, test_text in enumerate(test_cases):
        print(f"\n--- Test Case {i + 1} ---")
        print(f"Text: {test_text}")

        result = self.extract(test_text)

        classifications = result.get("category_classifications", [])
        primary = result.get("primary_classification", {})

        print(
            f"Primary: {primary.get('broad_category', 'N/A')} > {primary.get('sub_category', 'N/A')} (confidence: {primary.get('confidence', 0):.2f})")

        print("All Classifications:")
        for j, cls in enumerate(classifications):
            print(f"  {j + 1}. {cls.get('broad_category', 'N/A')} > {cls.get('sub_category', 'N/A')} "
                  f"(confidence: {cls.get('confidence', 0):.2f})")
            print(f"     Keywords: {cls.get('matched_keywords', [])}")
            print(f"     Reasoning: {cls.get('reasoning', 'N/A')}")

        results[f"test_{i + 1}"] = {
            "text": test_text,
            "classifications": classifications,
            "primary": primary
        }

    return results


def quick_test_multi_classification():
    """Quick test to verify multi-classification works"""
    extractor = MistralNERExtractor(
        model_path="/Users/pankajkumar/.cache/huggingface/hub/models--mlx-community--Dolphin-Mistral-24B-Venice-Edition-4bit/snapshots/7674b37fe24022cf79e77d204fac5b9582b0dc40",
        model_id="mlx-community/Dolphin-Mistral-24B-Venice-Edition-4bit"
    )

    test_cases = [
        "अनुसूचित जाति की नाबालिग लड़की से दुष्कर्म",
        "दलित बच्चे की हत्या",
        "ट्रैफिक जाम के कारण देरी"
    ]

    for text in test_cases:
        result = extractor.extract(text)
        classifications = result.get("category_classifications", [])
        primary = result.get("primary_classification", {})

        print(f"\nText: {text}")
        print(f"Primary: {primary.get('broad_category')} > {primary.get('sub_category')}")
        print(f"Total Classifications: {len(classifications)}")
        for cls in classifications:
            print(f"  - {cls.get('broad_category')} > {cls.get('sub_category')} ({cls.get('confidence'):.2f})")


# Uncomment to run quick test
# quick_test_multi_classification()

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

        # Test multi-classification specifically
        print("\n8. Testing Multi-Classification System...")
        print("-" * 30)

        # Test intersectional case
        intersectional_text = "अनुसूचित जाति की नाबालिग लड़की से दुष्कर्म का मामला, अमरोहा थाना रजबपुर"
        intersectional_result = extractor.extract(intersectional_text)

        print("Intersectional Test Case:")
        print(f"Text: {intersectional_text}")
        print("\nClassifications:")
        for i, cls in enumerate(intersectional_result.get("category_classifications", [])):
            print(f"  {i + 1}. {cls.get('broad_category')} > {cls.get('sub_category')} "
                  f"(confidence: {cls.get('confidence', 0):.2f})")
            print(f"     Keywords: {cls.get('matched_keywords', [])}")
            print(f"     Reasoning: {cls.get('reasoning', '')}")

        print(f"\nPrimary Classification: {intersectional_result.get('primary_classification', {})}")

        print("\nLocation Analysis:")
        location_analysis = intersectional_result.get("incident_location_analysis", {})
        print(f"  Incident Districts: {location_analysis.get('incident_districts', [])}")
        print(f"  Incident Thanas: {location_analysis.get('incident_thanas', [])}")
        print(f"  Primary Location: {location_analysis.get('primary_location', {})}")

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
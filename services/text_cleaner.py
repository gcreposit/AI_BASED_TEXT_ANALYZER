"""
AI-Powered Intelligent Text Cleaning Service
Uses Mistral 24B to intelligently clean scraped text
"""

import re
import logging
import json
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class AITextCleaner:
    """
    AI-powered text cleaning using Mistral 24B
    Intelligently removes clutter while preserving content
    """

    def __init__(self, ner_extractor):
        """
        Initialize with NER extractor (which has the loaded Mistral model)

        Args:
            ner_extractor: MistralNERExtractor instance with loaded model
        """
        self.ner_extractor = ner_extractor
        self.model = ner_extractor.model
        self.tokenizer = ner_extractor.tokenizer
        self._loading_method = ner_extractor._loading_method

        # Fallback regex patterns (used if AI fails)
        self.critical_noise_patterns = [
            r'Copyright\s*©.*',
            r'All\s*Rights\s*Reserved.*',
        ]

        # Min/max text length
        self.min_text_length = 20
        self.max_text_length = 5000

        # Cache for similar cleaning patterns (optional optimization)
        self._cleaning_cache = {}
        self._cache_max_size = 100

    def clean_text(self,
                   text: str,
                   source_type: str = "unknown",
                   use_ai: bool = True) -> Dict[str, Any]:
        """
        Main cleaning function with AI-powered intelligence

        Args:
            text: Raw input text
            source_type: 'social_media', 'news', 'whatsapp', etc.
            use_ai: If False, use basic regex fallback

        Returns:
            Dictionary with cleaned text and metadata
        """
        if not text or not text.strip():
            return {
                "cleaned_text": "",
                "original_length": 0,
                "cleaned_length": 0,
                "cleaning_method": "none",
                "cleaning_applied": False,
                "removed_noise": []
            }

        original_text = text
        original_length = len(text)

        # Quick pre-filter for extremely short texts
        if original_length < self.min_text_length:
            return {
                "cleaned_text": text.strip(),
                "original_length": original_length,
                "cleaned_length": len(text.strip()),
                "cleaning_method": "basic_trim",
                "cleaning_applied": False
            }

        # Check cache (optional optimization)
        cache_key = f"{hash(text[:200])}_{source_type}"
        if cache_key in self._cleaning_cache and use_ai:
            logger.debug("Using cached cleaning result")
            return self._cleaning_cache[cache_key]

        # Try AI-powered cleaning
        if use_ai and self._model_available():
            try:
                cleaned_text = self._ai_clean_text(text, source_type)
                cleaning_method = "ai_powered"

                # Validate AI output
                if not cleaned_text or len(cleaned_text) < self.min_text_length:
                    logger.warning("AI cleaning produced invalid output, using fallback")
                    cleaned_text = self._fallback_clean(text, source_type)
                    cleaning_method = "regex_fallback"

            except Exception as e:
                logger.error(f"AI cleaning failed: {e}, using fallback")
                cleaned_text = self._fallback_clean(text, source_type)
                cleaning_method = "regex_fallback"
        else:
            # Use regex fallback
            cleaned_text = self._fallback_clean(text, source_type)
            cleaning_method = "regex_basic"

        # Apply length constraints
        if len(cleaned_text) > self.max_text_length:
            cleaned_text = self._truncate_intelligently(cleaned_text)

        cleaned_length = len(cleaned_text)
        reduction_pct = ((original_length - cleaned_length) / original_length * 100) if original_length > 0 else 0

        result = {
            "cleaned_text": cleaned_text,
            "original_length": original_length,
            "cleaned_length": cleaned_length,
            "reduction_percentage": round(reduction_pct, 1),
            "cleaning_method": cleaning_method,
            "cleaning_applied": reduction_pct > 5,
            "removed_noise": []
        }

        # Cache result
        if use_ai and len(self._cleaning_cache) < self._cache_max_size:
            self._cleaning_cache[cache_key] = result

        return result

    def _model_available(self) -> bool:
        """Check if AI model is available"""
        return (self.ner_extractor._model_loaded and
                self.model is not None and
                self.tokenizer is not None)

    def _ai_clean_text(self, text: str, source_type: str) -> str:
        """
        Use Mistral 24B to intelligently clean text
        """
        # Build intelligent prompt
        prompt = self._build_cleaning_prompt(text, source_type)

        # Generate with appropriate method
        if self._loading_method == 'vllm':
            cleaned_text = self._generate_vllm(prompt)
        elif self._loading_method == 'mlx':
            cleaned_text = self._generate_mlx(prompt)
        elif self._loading_method == 'transformers':
            cleaned_text = self._generate_transformers(prompt)
        else:
            raise ValueError(f"Unknown loading method: {self._loading_method}")

        # Parse and validate
        cleaned_text = self._parse_ai_response(cleaned_text, text)

        return cleaned_text

    def _build_cleaning_prompt(self, text: str, source_type: str) -> str:
        """
        Build an intelligent prompt for text cleaning that adapts to various content sources:
        - News websites
        - Twitter (main tweets)
        - WhatsApp messages
        - Facebook posts
        - Instagram captions
        - YouTube video descriptions
        """
        # Truncate for prompt safety
        text_sample = text if len(text) <= 4000 else text[:4000] + "..."

        st = (source_type or "unknown").lower()

        # High-level source context (fixed typos; no accidental string concat)
        source_instructions = {
            "news": (
                "This is a news article scraped from a website. Remove headers, menus, ads, "
                "copyright lines, timestamps, subscription buttons, 'Read More' prompts, and social media widgets. "
                "Preserve the main story, quotes, facts, and contextual paragraphs. Maintain paragraph flow and Hindi/English mix."
            ),
            "twitter": (
                "This is a main tweet authored by the user. Preserve the tweet content verbatim. "
                "Only remove platform UI chrome (e.g., Like/Retweet/Reply counts, button labels) if present around the text."
            ),
            "whatsapp": (
                "This is a main WhatsApp message sent by the user. Preserve the message body verbatim. "
                "Only remove forwarding headers or surrounding app UI if present."
            ),
            "facebook": (
                "This is a main Facebook post authored by the user. Preserve the post content verbatim. "
                "Only remove page/app UI chrome (e.g., Like/Share prompts or counters) if present."
            ),
            "instagram": (
                "This is a main Instagram caption authored by the user. Preserve the caption verbatim. "
                "Only remove surrounding app UI (e.g., Like/Comment prompts) if present."
            ),
            "youtube": (
                "This is a YouTube video description. Remove subscribe/join/follow prompts, timestamps, chapter markers, links to unrelated videos, "
                "and promotional blurbs. Keep the key video summary, main topic, and informational parts."
            ),
            "unknown": (
                "This is text from an unknown mixed source. Remove navigation elements, advertisements, repeated UI labels, "
                "and unrelated URLs, while preserving all meaningful sentences and context."
            ),
        }

        # Platform-specific rules that complement the global guidelines
        platform_rules = {
            "twitter": (
                "TWITTER (MAIN TWEET):\n"
                "- KEEP the entire tweet body including hashtags, mentions, and emojis.\n"
                "- REMOVE only platform UI chrome (e.g., Like/Retweet/Reply counts, button labels) or tracking wrappers.\n"
                "- Do not trim hashtags/mentions if they are part of the user's message."
            ),
            "facebook": (
                "FACEBOOK (MAIN POST):\n"
                "- KEEP the full post text including hashtags/mentions/emojis.\n"
                "- REMOVE only surrounding UI prompts (e.g., Like/Share/Comment buttons, counters).\n"
                "- Do not trim hashtags/mentions that are part of the message."
            ),
            "instagram": (
                "INSTAGRAM (CAPTION):\n"
                "- KEEP the full caption including hashtags/mentions/emojis.\n"
                "- REMOVE only surrounding app UI (button labels, counters).\n"
                "- Do not trim hashtags/mentions that are part of the caption."
            ),
            "whatsapp": (
                "WHATSAPP (MESSAGE):\n"
                "- KEEP the full message as written by the user.\n"
                "- REMOVE only 'Forwarded' headers or app UI wrappers if present.\n"
                "- Do not delete emojis or lines unless they are clearly app UI."
            ),
            "news": (
                "NEWS (ARTICLE BODY):\n"
                "- KEEP article paragraphs, headlines, bylines, quotes, facts, and context.\n"
                "- REMOVE menus, ads, subscription prompts, 'Read More' widgets, social buttons, and copyright lines."
            ),
            "youtube": (
                "YOUTUBE (DESCRIPTION):\n"
                "- KEEP the summary/description conveying the video's topic and details.\n"
                "- REMOVE subscribe/join/follow prompts, chapter timestamps, unrelated links, and promotional blurbs."
            ),
            "unknown": (
                "UNKNOWN SOURCE:\n"
                "- KEEP meaningful sentences and context.\n"
                "- REMOVE obvious navigation, ads, repetitive UI labels, and unrelated links."
            ),
        }

        instruction = source_instructions.get(st, source_instructions["unknown"])
        platform_note = platform_rules.get(st, platform_rules["unknown"])

        # Base guidelines are consistent; platform_note clarifies exceptions for main posts
        prompt = f"""
    You are an advanced multilingual (Hindi/English/Hinglish) text-cleaning system
    trained to intelligently remove clutter and preserve meaningful information.

    TASK:
    Analyze and clean the following text by removing only noise or irrelevant UI elements,
    while keeping all actual content, meaning, names, facts, and context intact.

    SOURCE TYPE: {source_type.upper()}
    CONTEXT: {instruction}

    GUIDELINES (GLOBAL):
    1) KEEP
       - The main story/message/opinion and all real-world entities (people, places, dates, events).
       - Meaningful hashtags and emojis when they are part of the user's message.
       - Original language (do not translate) and paragraph/line breaks for readability.
       - Any Date or Time Related information shall not be removed
    2) REMOVE
       - Navigation/menu words (Home, Video, Search, होम, वीडियो, सर्च) and other website/app UI chrome.
       - Ads, promotions, sponsorships, subscription prompts.
       - Auto-generated lines ( © notices, “Read More”, “Follow us”).
       - URLs unrelated to the main content.
    3) LENGTH
       - If the text is very short (<30 chars), return it as-is.
       - If the text is long, remove redundancy but preserve coherence and flow.

    PLATFORM-SPECIFIC RULES:
    {platform_note}

    OUTPUT:
    - Return ONLY the cleaned text, nothing else (no explanations, no metadata).

    TEXT TO CLEAN:
    {text_sample}

    CLEANED TEXT (return only the cleaned version below):
    """.strip()

        return prompt

    def _generate_vllm(self, prompt: str) -> str:
        """Generate using vLLM"""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.1,  # Low temperature for consistent cleaning
            max_tokens=16384,
            top_p=0.9,
            frequency_penalty=0.1
        )

        outputs = self.model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def _generate_mlx(self, prompt: str) -> str:
        """Generate using MLX"""
        from mlx_lm import generate as mlx_generate

        return mlx_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=2048,
            temp=0.1,
            verbose=False
        )

    def _generate_transformers(self, prompt: str) -> str:
        """Generate using Transformers"""
        import torch

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )

        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.1,
                do_sample=False,  # Deterministic for consistency
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_response[len(prompt):].strip()

    def _parse_ai_response(self, ai_response: str, original_text: str) -> str:
        """
        Parse and validate AI response
        """
        cleaned = ai_response.strip()

        # Remove common AI response artifacts
        cleaned = re.sub(r'^(?:Here is the cleaned text|Cleaned text)[\s:]*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^```\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)

        # Validate: cleaned text should be reasonably related to original
        # If AI hallucinates or removes too much, use fallback
        if len(cleaned) < len(original_text) * 0.2:  # Lost more than 80%
            logger.warning("AI removed too much content, using fallback")
            return self._fallback_clean(original_text, "unknown")

        return cleaned.strip()

    def _fallback_clean(self, text: str, source_type: str) -> str:
        """
        Regex-based fallback cleaning (simple but safe)
        """
        cleaned = text

        # Remove critical noise patterns
        for pattern in self.critical_noise_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        # Basic normalization
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Max 2 newlines

        # Source-specific basic cleaning
        if source_type == "news":
            # Remove common news site patterns
            cleaned = re.sub(r'^(?:Breaking News|ब्रेकिंग न्यूज)[\s:]+', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'(?:और पढ़ें|Read More)[\s.]*$', '', cleaned, flags=re.IGNORECASE)

        elif source_type == "social_media":
            # Keep hashtags but clean excessive ones
            hashtags = re.findall(r'#\w+', cleaned)
            if len(hashtags) > 5:
                # Keep only first 3 hashtags
                for hashtag in hashtags[3:]:
                    cleaned = cleaned.replace(hashtag, '', 1)

        return cleaned.strip()

    def _truncate_intelligently(self, text: str) -> str:
        """
        Truncate text while trying to preserve complete sentences
        """
        if len(text) <= self.max_text_length:
            return text

        truncated = text[:self.max_text_length]

        # Try to end at sentence boundary
        for delimiter in ['।', '.', '!', '?']:
            last_delim = truncated.rfind(delimiter)
            if last_delim > self.max_text_length * 0.85:  # At least 85% preserved
                return truncated[:last_delim + 1].strip()

        # Fallback: end at word boundary
        last_space = truncated.rfind(' ')
        if last_space > 0:
            return truncated[:last_space].strip()

        return truncated.strip()

    def clear_cache(self):
        """Clear the cleaning cache"""
        self._cleaning_cache.clear()
        logger.info("Cleaning cache cleared")
import json
import logging
import re
import time
from typing import Dict, Any, List, Optional, Tuple

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class NewsExtractor:
    """
    News extraction service using Mistral 24B model to clean and extract
    proper news titles and content from scraped text data
    """

    def __init__(self, ner_extractor):
        """
        Initialize with existing NER extractor to leverage the loaded Mistral model
        """
        self.ner_extractor = ner_extractor
        self.model = ner_extractor.model
        self.tokenizer = ner_extractor.tokenizer
        self._loading_method = ner_extractor._loading_method

    def _build_news_extraction_prompt(self, raw_text: str) -> str:
        """
        Build specialized prompt for news extraction and cleaning
        """
        is_instruct_model = self.ner_extractor._is_instruction_tuned_model()

        instructions = f"""You are an expert news content extraction and cleaning assistant. Your task is to analyze scraped news text and extract clean, properly formatted news articles.

STRICT REQUIREMENTS:
1. Return ONLY valid JSON - no code blocks, explanations, or additional text
2. Extract proper news titles and content, removing ads, navigation, footers, etc.
3. If multiple news articles are present, return all of them
4. Maintain original language (Hindi/English/Hinglish)
5. Ensure titles are concise and descriptive
6. Content should be complete news articles only

JSON SCHEMA (EXACT FORMAT):
{{
  "articles": [
    {{
      "title": "Clean, descriptive news headline",
      "content": "Complete news article content without ads or irrelevant text",
      "language": "hindi/english/hinglish",
      "confidence": 0.0-1.0
    }}
  ],
  "total_articles": 0,
  "extraction_quality": "high/medium/low",
  "processing_notes": "Brief note about extraction quality"
}}

EXTRACTION RULES:

INCLUDE:
- News headlines and titles
- Complete news article content
- Direct quotes from officials/sources
- Key facts, dates, locations, people mentioned
- Relevant context and background information

EXCLUDE:
- Website navigation menus
- Advertisement content
- Social media sharing buttons
- Footer information
- Cookie notices
- Related articles links
- Author bio sections (unless part of the article)
- Comment sections
- Website branding/logos
- Subscription prompts
- Newsletter signup forms

CLEANING GUIDELINES:
- Remove HTML tags, excessive whitespace, and formatting artifacts
- Fix broken sentences caused by web scraping
- Maintain paragraph structure for readability
- Preserve quotes and factual information
- Remove duplicate content if present
- Combine fragmented sentences that belong together

TITLE EXTRACTION:
- Should be clear, concise, and descriptive (5-15 words typically)
- Remove website names, dates, or categories from titles
- Ensure title accurately represents the main news story
- For Hindi content, maintain proper Devanagari script

CONTENT REQUIREMENTS:
- Must be substantial (minimum 50 words for a valid article)
- Should tell a complete story with who, what, when, where, why
- Maintain chronological flow if present
- Include important details and context
- Remove any promotional or advertising content

LANGUAGE DETECTION:
- "hindi": Primarily Hindi/Devanagari script
- "english": Primarily English text
- "hinglish": Mixed Hindi-English content

CONFIDENCE SCORING:
- 1.0: Perfect extraction, clear single article
- 0.8-0.9: Good extraction, minor cleaning needed
- 0.6-0.7: Moderate extraction, some uncertainty
- 0.4-0.5: Low confidence, fragmented or unclear content
- 0.0-0.3: Very poor extraction, mostly noise

RAW SCRAPED TEXT TO PROCESS:
{raw_text.strip()}

IMPORTANT: Return ONLY the JSON object. No explanations or additional text."""

        if is_instruct_model:
            return f"<s>[INST]{instructions.strip()}[/INST]"
        else:
            return f"{instructions.strip()}\n\nJSON:"

    # def _safe_json_parse(self, response_text: str) -> Dict[str, Any]:
    #     """
    #     Enhanced JSON parsing for news extraction responses
    #     """
    #     text = response_text.strip()
    #
    #     # Remove code fences and markdown
    #     text = re.sub(r'^```(?:json)?\n?', '', text, flags=re.MULTILINE)
    #     text = re.sub(r'\n?```$', '', text, flags=re.MULTILINE)
    #     text = text.strip()
    #
    #     # Try to find JSON object
    #     json_match = re.search(r'\{.*\}', text, re.DOTALL)
    #     if json_match:
    #         text = json_match.group(0)
    #
    #     try:
    #         parsed_json = json.loads(text)
    #         return parsed_json
    #     except json.JSONDecodeError as e:
    #         logger.warning(f"JSON parsing failed: {e}")
    #
    #         # Try to fix common JSON issues
    #         try:
    #             # Fix trailing commas
    #             text = re.sub(r',(\s*[}\]])', r'\1', text)
    #             # Fix unescaped quotes in strings
    #             text = re.sub(r'(?<!\\)"(?![,\s}:\[\]])', '\\"', text)
    #             parsed_json = json.loads(text)
    #             return parsed_json
    #         except json.JSONDecodeError:
    #             logger.error("Could not fix JSON parsing issues")
    #             return self._get_fallback_response()

    def _safe_json_parse(self, response_text: str) -> Dict[str, Any]:
        """
        Enhanced JSON parsing for news extraction responses
        """
        text = response_text.strip()

        # Remove code fences and markdown
        text = re.sub(r'^```(?:json)?\n?', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n?```$', '', text, flags=re.MULTILINE)
        text = text.strip()

        # Try to find JSON object - use non-greedy matching and look for the FIRST complete object
        json_match = re.search(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        # First attempt - try parsing as-is
        try:
            parsed_json = json.loads(text)
            return parsed_json
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            logger.debug(f"Failed JSON text: {text[:500]}...")

            # Try to fix common JSON issues
            try:
                # Fix trailing commas before closing brackets
                text = re.sub(r',(\s*[}\]])', r'\1', text)

                # Fix missing commas between array/object elements
                text = re.sub(r'"\s*\n\s*"', '",\n"', text)
                text = re.sub(r'}\s*\n\s*{', '},\n{', text)

                # Fix truncated strings (common when model hits token limit)
                # If the JSON ends mid-string, try to close it
                if text.rstrip().endswith('"') and text.count('"') % 2 != 0:
                    text = text.rstrip() + '"'

                # Try to ensure proper closing
                open_braces = text.count('{') - text.count('}')
                open_brackets = text.count('[') - text.count(']')

                if open_braces > 0:
                    text += '}' * open_braces
                if open_brackets > 0:
                    text += ']' * open_brackets

                parsed_json = json.loads(text)
                logger.info("Successfully repaired malformed JSON")
                return parsed_json

            except json.JSONDecodeError as e2:
                logger.error(f"Could not fix JSON parsing issues: {e2}")
                logger.error(f"Attempted to parse: {text[:1000]}...")
                return self._get_fallback_response()

    def _preprocess_scraped_text(self, raw_text: str) -> str:
        """
        Clean and preprocess scraped text to improve extraction quality
        """
        # Remove excessive navigation/menu repetition
        lines = raw_text.split('\n')

        # Remove duplicate consecutive lines (common in navigation menus)
        cleaned_lines = []
        prev_line = None
        consecutive_count = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip if same line repeated more than 2 times
            if line == prev_line:
                consecutive_count += 1
                if consecutive_count > 2:
                    continue
            else:
                consecutive_count = 0

            cleaned_lines.append(line)
            prev_line = line

        cleaned_text = '\n'.join(cleaned_lines)

        # Remove common noise patterns
        noise_patterns = [
            r'कॉपी लिंक\s*शेयर',
            r'(?:होम|वीडियो|सर्च|ई-पेपर)\s*(?:होम|वीडियो|सर्च|ई-पेपर)*',
            r'Copyright ©.*',
            r'Advertise with Us.*',
            r'This website follows.*'
        ]

        for pattern in noise_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)

        # Limit length while trying to keep complete sentences
        if len(cleaned_text) > 25000:
            cleaned_text = cleaned_text[:25000]
            # Try to end at a sentence
            last_period = cleaned_text.rfind('।')  # Hindi period
            if last_period < 0:
                last_period = cleaned_text.rfind('.')
            if last_period > 20000:  # Make sure we don't cut too much
                cleaned_text = cleaned_text[:last_period + 1]

        return cleaned_text

    def _get_fallback_response(self) -> Dict[str, Any]:
        """
        Return fallback response when extraction fails
        """
        return {
            "articles": [],
            "total_articles": 0,
            "extraction_quality": "low",
            "processing_notes": "Extraction failed, could not parse content"
        }

    def _validate_and_clean_response(self, parsed_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean the parsed response
        """
        if not isinstance(parsed_json, dict):
            return self._get_fallback_response()

        # Ensure required fields exist
        articles = parsed_json.get("articles", [])
        if not isinstance(articles, list):
            articles = []

        # Validate each article
        cleaned_articles = []
        for article in articles:
            if not isinstance(article, dict):
                continue

            title = str(article.get("title", "")).strip()
            content = str(article.get("content", "")).strip()

            # Skip articles with insufficient content
            if len(title) < 5 or len(content) < 50:
                logger.warning(f"Skipping article - title length: {len(title)}, content length: {len(content)}")
                logger.debug(f"Rejected title: {title[:100]}")
                logger.debug(f"Rejected content: {content[:200]}")
                continue

            # Clean and validate fields
            # Clean and validate fields - handle non-string types
            raw_confidence = article.get("confidence", 0.5)
            try:
                confidence_value = float(raw_confidence) if raw_confidence is not None else 0.5
            except (ValueError, TypeError):
                confidence_value = 0.5

            cleaned_article = {
                "title": str(title)[:200],  # Convert to string and limit title length
                "content": str(content),  # Convert to string
                "language": str(article.get("language", "unknown")),
                "confidence": max(0.0, min(1.0, confidence_value))
            }

            cleaned_articles.append(cleaned_article)

        return {
            "articles": cleaned_articles,
            "total_articles": len(cleaned_articles),
            "extraction_quality": parsed_json.get("extraction_quality", "medium"),
            "processing_notes": parsed_json.get("processing_notes", "Processing completed")
        }

    def extract_news(self, raw_text: str, max_tokens: int = 32000, temperature: float = 0.1) -> Dict[str, Any]:
        """
        Extract clean news articles from raw scraped text
        """
        if not raw_text or not raw_text.strip():
            logger.warning("Empty text provided for news extraction")
            return self._get_fallback_response()

        # Preprocess the input
        raw_text = self._preprocess_scraped_text(raw_text)
        logger.info(f"Preprocessed text length: {len(raw_text)} characters")

        # Truncate input if too long (keep within model limits)
        if len(raw_text) > 30000:
            logger.info(f"Truncating input text from {len(raw_text)} to 30000 characters")
            raw_text = raw_text[:30000]

        start_time = time.time()

        try:
            if not self.ner_extractor._model_loaded or not self.model:
                logger.error("Model not loaded for news extraction")
                return self._get_fallback_response()

            # Build specialized prompt for news extraction
            prompt = self._build_news_extraction_prompt(raw_text)

            # Debug: Log prompt statistics
            logger.info(f"Prompt length: {len(prompt)} characters")
            logger.info(f"Prompt tokens (estimated): {len(prompt) // 4}")
            logger.info(f"Max new tokens allowed: {max_tokens}")

            raw_response = ""

            # Generate response based on loading method
            if self._loading_method == 'vllm':
                try:
                    from vllm import SamplingParams

                    # Get proper stop tokens
                    stop_tokens = []
                    if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
                        stop_tokens.append(self.tokenizer.eos_token)

                    logger.debug(f"Stop tokens configured: {stop_tokens}")

                    sampling_params = SamplingParams(
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=0.95,
                        frequency_penalty=0.1,
                        presence_penalty=0.1,
                        stop=stop_tokens if stop_tokens else None
                    )

                    outputs = self.model.generate([prompt], sampling_params)
                    raw_response = outputs[0].outputs[0].text.strip()

                    # Log finish reason to understand why generation stopped
                    finish_reason = outputs[0].outputs[0].finish_reason
                    logger.info(f"✅ Generation finished with reason: {finish_reason}")

                    # Log token usage
                    prompt_tokens = len(outputs[0].prompt_token_ids)
                    output_tokens = len(outputs[0].outputs[0].token_ids)
                    logger.info(
                        f"Token usage - Prompt: {prompt_tokens}, Output: {output_tokens}, Total: {prompt_tokens + output_tokens}")

                    logger.debug("Used vLLM for news extraction")

                except Exception as e:
                    logger.error(f"vLLM generation failed: {e}", exc_info=True)
                    return self._get_fallback_response()

            elif self._loading_method == 'mlx':
                try:
                    from mlx_lm import generate as mlx_generate

                    raw_response = mlx_generate(
                        self.model,
                        self.tokenizer,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temp=temperature,
                        verbose=False
                    )
                    logger.debug("Used MLX for news extraction")
                    logger.info("✅ Generation completed (MLX)")

                except Exception as e:
                    logger.error(f"MLX generation failed: {e}", exc_info=True)
                    return self._get_fallback_response()

            elif self._loading_method == 'transformers':
                try:
                    import torch

                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=8192,
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
                    logger.debug("Used Transformers for news extraction")
                    logger.info("✅ Generation completed (Transformers)")

                except Exception as e:
                    logger.error(f"Transformers generation failed: {e}", exc_info=True)
                    return self._get_fallback_response()

            if not raw_response:
                logger.error("❌ No response generated from model")
                return self._get_fallback_response()

            logger.info(f"Raw model response length: {len(raw_response)} characters")
            logger.debug(f"Raw model response (first 4000 chars): {raw_response[:4000]}")
            logger.debug(f"Raw model response (last 500 chars): {raw_response[-500:]}")

            # Parse and validate the response
            parsed_json = self._safe_json_parse(raw_response)

            # Log parsing results
            if isinstance(parsed_json, dict) and 'articles' in parsed_json:
                logger.info(f"Parsed JSON contains {len(parsed_json.get('articles', []))} articles")
            else:
                logger.warning("Parsed JSON is malformed or missing articles")

            result = self._validate_and_clean_response(parsed_json)

            processing_time = (time.time() - start_time) * 1000
            logger.info(f"News extraction completed in {processing_time:.2f}ms, "
                        f"extracted {result['total_articles']} articles")

            return result

        except Exception as e:
            logger.error(f"News extraction failed: {e}", exc_info=True)
            return self._get_fallback_response()

    def extract_news_batch(self, raw_texts: List[str], max_tokens: int = 32000, temperature: float = 0.1) -> List[
        Dict[str, Any]]:
        """
        Extract news from multiple raw text inputs efficiently
        """
        if not raw_texts:
            return []

        results = []
        total_start = time.time()

        # vLLM batch processing
        if self._loading_method == 'vllm':
            try:
                from vllm import SamplingParams

                # Truncate long texts and build prompts
                processed_texts = []
                prompts = []

                for text in raw_texts:
                    if len(text) > 30000:
                        text = text[:30000]
                    processed_texts.append(text)
                    prompts.append(self._build_news_extraction_prompt(text))

                # Get proper stop tokens
                stop_tokens = []
                if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
                    stop_tokens.append(self.tokenizer.eos_token)

                sampling_params = SamplingParams(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.95,
                    frequency_penalty=0.1,
                    presence_penalty=0.1,
                    stop=stop_tokens if stop_tokens else None
                )

                outputs = self.model.generate(prompts, sampling_params)

                for i, output in enumerate(outputs):
                    try:
                        raw_response = output.outputs[0].text.strip()
                        finish_reason = output.outputs[0].finish_reason
                        logger.debug(f"Batch item {i} finished with reason: {finish_reason}")

                        parsed_json = self._safe_json_parse(raw_response)
                        result = self._validate_and_clean_response(parsed_json)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to process batch item {i}: {e}")
                        results.append(self._get_fallback_response())

                total_time = time.time() - total_start
                logger.info(f"vLLM batch news extraction: {len(raw_texts)} texts in {total_time:.2f}s")
                return results

            except Exception as e:
                logger.error(f"vLLM batch processing failed: {e}", exc_info=True)

        # Sequential processing for MLX and Transformers
        for i, text in enumerate(raw_texts):
            try:
                result = self.extract_news(text, max_tokens, temperature)
                results.append(result)

                if (i + 1) % 5 == 0:
                    elapsed = time.time() - total_start
                    avg_time = elapsed / (i + 1)
                    eta = avg_time * (len(raw_texts) - i - 1)
                    logger.info(f"Processed {i + 1}/{len(raw_texts)} texts in {elapsed:.2f}s (ETA: {eta:.2f}s)")

            except Exception as e:
                logger.error(f"Failed to process text {i + 1}: {e}")
                results.append(self._get_fallback_response())

        total_time = time.time() - total_start
        logger.info(f"Sequential batch news extraction: {len(raw_texts)} texts in {total_time:.2f}s")
        return results


    def get_extraction_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the news extractor
        """
        return {
            "model_loaded": self.ner_extractor._model_loaded,
            "loading_method": self._loading_method,
            "max_input_length": 30000,
            "supported_languages": ["hindi", "english", "hinglish"],
            "model_info": self.ner_extractor.get_model_info()
        }
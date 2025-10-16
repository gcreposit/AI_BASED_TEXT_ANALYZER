"""
Universal Advanced Sentiment Analysis Service
Analyzes Pro/Against/Neutral stance without predefined biases
Supports Hindi, English, and Hinglish

Primary: LLM-based full context analysis
Fallback: Minimal rule-based analysis
"""

import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class SentimentStance(str, Enum):
    """Sentiment stance categories"""
    PRO = "pro"
    AGAINST = "against"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class AdvancedSentimentAnalyzer:
    """
    Universal sentiment analyzer with LLM-first approach
    No predefined biases or keyword lists
    Analyzes full text context
    """

    def __init__(self, llm_model=None, llm_tokenizer=None, loading_method=None):
        """
        Initialize with LLM (primary) and minimal rule-based (fallback)

        Args:
            llm_model: Loaded LLM model
            llm_tokenizer: LLM tokenizer
            loading_method: 'vllm', 'mlx', or 'transformers'
        """
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.loading_method = loading_method

        # Minimal universal sentiment indicators (language-agnostic patterns)
        self.positive_patterns = [
            r'\b(समर्थन|support|favor|help|प्रशंसा|praise|अच्छा|good|बेहतर|better)\b',
            r'\b(स्वागत|welcome|सराहना|appreciate|धन्यवाद|thank)\b'
        ]

        self.negative_patterns = [
            r'\b(विरोध|against|oppose|खिलाफ|नफरत|hate)\b',
            r'\b(आरोप|accuse|दोष|blame|निंदा|condemn)\b',
            r'\b(शिकायत|complaint|आक्रोश|outrage|असंतोष|dissatisfaction)\b'
        ]

    def analyze_advanced_sentiment(self,
                                   text: str,
                                   ner_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main function: LLM analyzes full text, rule-based minimal fallback

        Args:
            text: Full original text (not truncated)
            ner_data: Extracted NER data

        Returns:
            Dictionary with Pro/Against/Neutral analysis
        """
        result = {
            'overall_stance': SentimentStance.NEUTRAL,
            'overall_confidence': 0.0,
            'pro_towards': {
                'castes': [],
                'religions': [],
                'organisations': [],
                'political_parties': [],
                'other_aspects': []
            },
            'against_towards': {
                'castes': [],
                'religions': [],
                'organisations': [],
                'political_parties': [],
                'other_aspects': []
            },
            'neutral_towards': {
                'castes': [],
                'religions': [],
                'organisations': [],
                'political_parties': [],
                'other_aspects': []
            },
            'analysis_method': 'rule_based',
            'reasoning': ''
        }

        # PRIMARY: LLM analysis with FULL text context
        if self.llm_model and self.llm_tokenizer:
            try:
                logger.info("Performing LLM-based sentiment analysis on full text")
                llm_result = self._analyze_with_llm_full_context(text, ner_data)

                if llm_result and llm_result.get('success', False):
                    result = self._convert_llm_to_result_format(llm_result, ner_data)
                    result['analysis_method'] = 'llm_primary'

                    # Only enhance with rule-based if LLM confidence is very low
                    if result['overall_confidence'] < 0.5:
                        logger.info("LLM confidence low, adding minimal rule-based support")
                        rule_result = self._minimal_rule_based_analysis(text, ner_data)
                        result = self._merge_sentiment_results(result, rule_result)
                        result['analysis_method'] = 'hybrid_llm_rule'

                    return result

            except Exception as e:
                logger.error(f"LLM sentiment analysis failed: {e}", exc_info=True)

        # FALLBACK: Minimal rule-based analysis
        logger.warning("Using minimal rule-based sentiment analysis (fallback)")
        result = self._minimal_rule_based_analysis(text, ner_data)
        result['analysis_method'] = 'rule_based_fallback'

        return result

    def _analyze_with_llm_full_context(self,
                                       text: str,
                                       ner_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        LLM analysis with FULL text context (no truncation)
        Universal approach without predefined biases

        Args:
            text: Full original text
            ner_data: NER extracted data

        Returns:
            LLM sentiment result
        """
        # Prepare entity lists from NER
        castes = ner_data.get('caste_names', [])
        religions = ner_data.get('religion_names', [])
        organisations = ner_data.get('organisation_names', [])
        political_parties = ner_data.get('political_party_names', [])
        persons = ner_data.get('person_names', [])
        locations = ner_data.get('location_names', [])
        incidents = ner_data.get('incidents', [])

        # Build entity context
        entity_context = []
        if castes:
            entity_context.append(f"- Castes mentioned: {', '.join(castes)}")
        if religions:
            entity_context.append(f"- Religions mentioned: {', '.join(religions)}")
        if organisations:
            entity_context.append(f"- Organizations mentioned: {', '.join(organisations)}")
        if political_parties:
            entity_context.append(f"- Political parties mentioned: {', '.join(political_parties)}")
        if persons:
            entity_context.append(f"- Persons mentioned: {', '.join(persons[:5])}")
        if locations:
            entity_context.append(f"- Locations mentioned: {', '.join(locations[:5])}")
        if incidents:
            entity_context.append(f"- Incidents mentioned: {', '.join(incidents[:3])}")

        entity_info = "\n".join(entity_context) if entity_context else "- No specific entities detected"

        prompt = f"""You are an expert multilingual sentiment analyzer. Analyze the COMPLETE text below to determine the stance (Pro/Against/Neutral) towards different entities, groups, organizations, and aspects mentioned.

FULL TEXT TO ANALYZE:
{text}

DETECTED ENTITIES (for reference):
{entity_info}

INSTRUCTIONS:
1. Read and understand the ENTIRE text context
2. Identify ALL entities, groups, organizations, institutions, and aspects mentioned
3. For EACH identified entity/aspect, determine the stance based ONLY on the text:
   - PRO: Text expresses support, praise, defense, or positive sentiment
   - AGAINST: Text expresses opposition, criticism, complaints, accusations, or negative sentiment
   - NEUTRAL: Text mentions without clear positive or negative stance

4. Look for contextual indicators:
   - Accusations, complaints, misconduct allegations → likely AGAINST
   - Praise, support, defense, appreciation → likely PRO
   - Factual reporting without judgment → likely NEUTRAL

5. Consider:
   - Words expressing dissatisfaction, outrage, complaints
   - Words expressing support, appreciation, praise
   - Overall tone and context around each entity

6. DO NOT use predefined biases - analyze based purely on the text provided

RETURN ONLY VALID JSON (no markdown, no extra text):
{{
  "entities_analyzed": [
    {{
      "entity_name": "name of entity/aspect",
      "entity_type": "caste/religion/organisation/political_party/person/aspect/other",
      "stance": "pro/against/neutral",
      "confidence": 0.85,
      "reasoning": "brief explanation based on text context"
    }}
  ],
  "overall_stance": "pro/against/neutral",
  "overall_confidence": 0.85,
  "overall_reasoning": "brief explanation of overall sentiment in text"
}}

JSON:"""

        try:
            response = self._call_llm(prompt)

            # Clean response
            response = response.strip()
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].split('```')[0].strip()

            # Extract JSON
            if '{' in response:
                response = response[response.index('{'):]
            if '}' in response:
                response = response[:response.rindex('}')+1]

            llm_data = json.loads(response)
            llm_data['success'] = True

            logger.info(f"LLM full context analysis successful: {llm_data.get('overall_stance', 'unknown')}")
            return llm_data

        except json.JSONDecodeError as e:
            logger.error(f"LLM JSON parsing failed: {e}. Response preview: {response[:300]}")
            return {'success': False}
        except Exception as e:
            logger.error(f"LLM analysis error: {e}", exc_info=True)
            return {'success': False}

    def _call_llm(self, prompt: str) -> str:
        """Call LLM based on loading method"""
        try:
            if self.loading_method == 'vllm':
                from vllm import SamplingParams
                sampling_params = SamplingParams(
                    temperature=0.1,
                    max_tokens=1500,
                    top_p=0.95
                )
                outputs = self.llm_model.generate([prompt], sampling_params)
                return outputs[0].outputs[0].text.strip()

            elif self.loading_method == 'mlx':
                from mlx_lm import generate as mlx_generate
                return mlx_generate(
                    self.llm_model,
                    self.llm_tokenizer,
                    prompt=prompt,
                    max_tokens=1500,
                    temp=0.1
                )

            else:  # transformers
                import torch
                inputs = self.llm_tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096  # Increased for full context
                )

                if hasattr(self.llm_model, 'device'):
                    inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.llm_model.generate(
                        **inputs,
                        max_new_tokens=1500,
                        temperature=0.1,
                        do_sample=True
                    )

                response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response[len(prompt):].strip()

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _convert_llm_to_result_format(self,
                                     llm_data: Dict[str, Any],
                                     ner_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert LLM output to standard result format"""
        result = {
            'overall_stance': llm_data.get('overall_stance', SentimentStance.NEUTRAL),
            'overall_confidence': float(llm_data.get('overall_confidence', 0.7)),
            'pro_towards': {
                'castes': [],
                'religions': [],
                'organisations': [],
                'political_parties': [],
                'other_aspects': []
            },
            'against_towards': {
                'castes': [],
                'religions': [],
                'organisations': [],
                'political_parties': [],
                'other_aspects': []
            },
            'neutral_towards': {
                'castes': [],
                'religions': [],
                'organisations': [],
                'political_parties': [],
                'other_aspects': []
            },
            'analysis_method': 'llm',
            'reasoning': llm_data.get('overall_reasoning', '')
        }

        # Process entities analyzed by LLM
        entities_analyzed = llm_data.get('entities_analyzed', [])

        for entity_data in entities_analyzed:
            entity_name = entity_data.get('entity_name', '')
            entity_type = entity_data.get('entity_type', 'other')
            stance = entity_data.get('stance', 'neutral')
            confidence = entity_data.get('confidence', 0.7)
            reasoning = entity_data.get('reasoning', '')

            if not entity_name:
                continue

            # Map entity type to result categories
            category_mapping = {
                'caste': 'castes',
                'religion': 'religions',
                'organisation': 'organisations',
                'organization': 'organisations',
                'political_party': 'political_parties',
                'party': 'political_parties',
                'person': 'other_aspects',
                'aspect': 'other_aspects',
                'other': 'other_aspects'
            }

            category = category_mapping.get(entity_type.lower(), 'other_aspects')

            entity_info = {
                'name': entity_name,
                'stance': stance,
                'confidence': float(confidence),
                'reasoning': reasoning,
                'source': 'llm'
            }

            # Add to appropriate stance category
            if stance == 'pro':
                result['pro_towards'][category].append(entity_info)
            elif stance == 'against':
                result['against_towards'][category].append(entity_info)
            else:
                result['neutral_towards'][category].append(entity_info)

        return result

    def _minimal_rule_based_analysis(self,
                                    text: str,
                                    ner_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Minimal rule-based analysis without predefined biases
        Only uses basic pattern matching on entities from NER
        """
        result = {
            'overall_stance': SentimentStance.NEUTRAL,
            'overall_confidence': 0.3,  # Low confidence for rule-based
            'pro_towards': {
                'castes': [],
                'religions': [],
                'organisations': [],
                'political_parties': [],
                'other_aspects': []
            },
            'against_towards': {
                'castes': [],
                'religions': [],
                'organisations': [],
                'political_parties': [],
                'other_aspects': []
            },
            'neutral_towards': {
                'castes': [],
                'religions': [],
                'organisations': [],
                'political_parties': [],
                'other_aspects': []
            },
            'analysis_method': 'rule_based',
            'reasoning': 'Minimal rule-based analysis - LLM unavailable'
        }

        text_lower = text.lower()

        # Extract entities from NER
        entity_groups = {
            'castes': ner_data.get('caste_names', []),
            'religions': ner_data.get('religion_names', []),
            'organisations': ner_data.get('organisation_names', []),
            'political_parties': ner_data.get('political_party_names', [])
        }

        # Analyze each entity group
        for category, entities in entity_groups.items():
            for entity in entities:
                if not entity:
                    continue

                # Extract context around entity
                context = self._extract_entity_context(text, entity, window=150)
                context_lower = context.lower()

                # Count positive and negative patterns
                positive_count = sum(
                    len(re.findall(pattern, context_lower, re.IGNORECASE))
                    for pattern in self.positive_patterns
                )

                negative_count = sum(
                    len(re.findall(pattern, context_lower, re.IGNORECASE))
                    for pattern in self.negative_patterns
                )

                # Determine stance
                if negative_count > positive_count and negative_count >= 1:
                    stance = SentimentStance.AGAINST
                    confidence = min(0.6, 0.3 + (negative_count * 0.1))
                    result['against_towards'][category].append({
                        'name': entity,
                        'stance': stance,
                        'confidence': confidence,
                        'source': 'rule_based'
                    })
                elif positive_count > negative_count and positive_count >= 1:
                    stance = SentimentStance.PRO
                    confidence = min(0.6, 0.3 + (positive_count * 0.1))
                    result['pro_towards'][category].append({
                        'name': entity,
                        'stance': stance,
                        'confidence': confidence,
                        'source': 'rule_based'
                    })
                else:
                    result['neutral_towards'][category].append({
                        'name': entity,
                        'stance': SentimentStance.NEUTRAL,
                        'confidence': 0.5,
                        'source': 'rule_based'
                    })

        # Determine overall stance
        result['overall_stance'], result['overall_confidence'] = self._determine_overall_stance(result)
        result['reasoning'] = self._generate_reasoning(result)

        return result

    def _extract_entity_context(self, text: str, entity: str, window: int = 150) -> str:
        """Extract context around entity mention"""
        try:
            pos = text.lower().find(entity.lower())
            if pos == -1:
                return text[:window * 2]
            start = max(0, pos - window)
            end = min(len(text), pos + len(entity) + window)
            return text[start:end]
        except:
            return text[:window * 2]

    def _determine_overall_stance(self, result: Dict[str, Any]) -> Tuple[str, float]:
        """Determine overall stance from individual stances"""
        pro_count = sum(len(result['pro_towards'][k]) for k in result['pro_towards'])
        against_count = sum(len(result['against_towards'][k]) for k in result['against_towards'])
        neutral_count = sum(len(result['neutral_towards'][k]) for k in result['neutral_towards'])

        total = pro_count + against_count + neutral_count
        if total == 0:
            return SentimentStance.NEUTRAL, 0.5

        if against_count > pro_count * 1.2:
            return SentimentStance.AGAINST, min(0.9, 0.5 + (against_count / total) * 0.4)
        elif pro_count > against_count * 1.2:
            return SentimentStance.PRO, min(0.9, 0.5 + (pro_count / total) * 0.4)
        else:
            return SentimentStance.NEUTRAL, 0.6

    def _generate_reasoning(self, result: Dict[str, Any]) -> str:
        """Generate human-readable reasoning"""
        parts = []

        for category, items in result['pro_towards'].items():
            if items:
                names = [item.get('name', '')[:30] for item in items[:3] if item.get('name')]
                if names:
                    parts.append(f"Pro-{category}: {', '.join(names)}")

        for category, items in result['against_towards'].items():
            if items:
                names = [item.get('name', '')[:30] for item in items[:3] if item.get('name')]
                if names:
                    parts.append(f"Against-{category}: {', '.join(names)}")

        return " | ".join(parts) if parts else "No clear stance detected"

    def _merge_sentiment_results(self,
                                 primary: Dict[str, Any],
                                 secondary: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two sentiment results (primary takes precedence)"""

        if primary['overall_confidence'] >= secondary['overall_confidence']:
            base = primary.copy()

            # Add missing entities from secondary
            for stance in ['pro_towards', 'against_towards', 'neutral_towards']:
                for category in base[stance]:
                    existing_names = {item.get('name', '').lower() for item in base[stance][category]}

                    for item in secondary[stance].get(category, []):
                        item_name = item.get('name', '').lower()
                        if item_name and item_name not in existing_names:
                            base[stance][category].append(item)

            return base
        else:
            return self._merge_sentiment_results(secondary, primary)
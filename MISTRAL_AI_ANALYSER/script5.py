#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hindi Text Extractor using Claude 3.7 Sonnet Reasoning Gemma3 12B on Apple Silicon (MLX)
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import textwrap
from typing import Dict, Any, List
import re as stdre

from huggingface_hub import snapshot_download
from mlx_lm import load
from script import DISTRICT_VARIANTS

# Use the 'regex' package for robust Unicode handling (e.g., Devanagari hashtags)
try:
    import regex as re
except Exception:
    re = stdre

# --------------------------- Heuristics & Dictionaries ---------------------------

RELIGIONS = [
    "हिन्दू", "हिंदू", "मुस्लिम", "इस्लाम", "ईसाई", "क्रिश्चियन", "सिख", "बौद्ध", "जैन",
    "Hindu", "Muslim", "Islam", "Christian", "Sikh", "Buddhist", "Jain"
]

CASTES = [
    "SC", "ST", "OBC", "ब्राह्मण", "ठाकुर", "राजपूत", "यादव", "दलित", "कुर्मी"
]

UP_DISTRICTS = [
    "आगरा", "अलीगढ़", "अयोध्या", "बांदा", "बरेली", "लखनऊ", "वाराणसी"
]

THANA_PATTERNS = [
    r"(?:थाना|कोतवाली|Kotwali|Thana|PS)\s+([^\s,।:-]+(?:\s+[^\s,।:-]+)?)"
]

# --------------------------- LLM Prompting (Claude 3.7 Sonnet Reasoning) ---------------------------

JSON_SCHEMA = {
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

def dedupe(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if not x:
            continue
        k = x.strip()
        if k and k not in seen:
            out.append(k)
            seen.add(k)
    return out


def find_hashtags(text: str) -> List[str]:
    pat = re.compile(r"#([A-Za-z0-9_\.\u0900-\u097F]+)")
    return dedupe([m.group(0) for m in pat.finditer(text)])


def find_mentions(text: str) -> List[str]:
    pat = re.compile(r"@([A-Za-z0-9_\.]+)")
    return dedupe([m.group(0) for m in pat.finditer(text)])


def find_districts(text: str) -> List[str]:
    hits = []
    t = text
    for k, v in DISTRICT_VARIANTS.items():
        if k in t:
            hits.append(v)
    for d in UP_DISTRICTS:
        if d in t:
            hits.append(d)
    return dedupe(hits)


def find_thana(text: str) -> List[str]:
    hits = []
    for p in THANA_PATTERNS:
        for m in re.finditer(p, text, flags=re.IGNORECASE):
            name = m.group(1).strip()
            name = name.rstrip(".,:;|/\\-—–")
            hits.append(name)
    return dedupe(hits)


def find_keywords(text: str, vocab: List[str]) -> List[str]:
    hits = []
    for w in vocab:
        if w in text:
            hits.append(w)
    return dedupe(hits)


def build_inst_prompt(text: str) -> str:
    instructions = f"""
आप एक डेटा-एक्सट्रैक्शन सहायक हैं। नीचे दिए गए हिन्दी टेक्स्ट से पार्स कर के केवल वैध JSON लौटाइए। 
JSON की कुंजियाँ और फ़ॉर्मेट बिल्कुल इस स्कीमा जैसा होना चाहिए:
{json.dumps(JSON_SCHEMA, ensure_ascii=False, indent=2)}

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


def safe_json_parse(s: str) -> Dict[str, Any]:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = stdre.sub(r"^json", "", s, flags=stdre.IGNORECASE).strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            idx = s.rfind("}")
            if idx != -1:
                return json.loads(s[:idx + 1])
        except Exception:
            pass
    return {}


def merge_results(llm_json: Dict[str, Any], regex_boost: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(JSON_SCHEMA)
    list_keys = [
        "person_names", "organisation_names", "location_names", "district_names",
        "thana_names", "incidents", "caste_names", "religion_names", "hashtags",
        "mention_ids", "events"
    ]
    for k in list_keys:
        vals = []
        v1 = llm_json.get(k, []) if isinstance(llm_json.get(k, []), list) else []
        v2 = regex_boost.get(k, []) if isinstance(regex_boost.get(k, []), list) else []
        vals.extend([str(x).strip() for x in v1 if x])
        vals.extend([str(x).strip() for x in v2 if x])
        out[k] = dedupe(vals)

    s = llm_json.get("sentiment", {})
    if isinstance(s, dict) and "label" in s and "confidence" in s:
        out["sentiment"] = {"label": str(s.get("label")), "confidence": float(s.get("confidence"))}
    else:
        out["sentiment"] = {"label": "neutral", "confidence": 0.5}

    ctx = llm_json.get("contextual_understanding", "")
    out["contextual_understanding"] = str(ctx).strip()

    return out


def run_claude_infer(model_id: str, prompt: str, max_tokens: int = 1024, temp: float = 0.2) -> str:
    local_path = snapshot_download(repo_id=model_id, local_files_only=False)
    model, tokenizer = load(local_path)
    out = model.generate(inputs=prompt, max_length=max_tokens, temperature=temp)
    return out


def extract(text: str, model_id: str) -> Dict[str, Any]:
    regex_boost = {
        "hashtags": find_hashtags(text),
        "mention_ids": find_mentions(text),
        "district_names": find_districts(text),
        "thana_names": find_thana(text),
        "caste_names": find_keywords(text, CASTES),
        "religion_names": find_keywords(text, RELIGIONS),
    }

    prompt = build_inst_prompt(text)
    raw = run_claude_infer(model_id, prompt)
    llm_json = safe_json_parse(raw)

    final = merge_results(llm_json, regex_boost)
    return final


def main():
    parser = argparse.ArgumentParser(
        description="Hindi Text -> Structured Extraction via Claude 3.7 Sonnet Reasoning Gemma3 12B")
    parser.add_argument("--text", type=str, help="Direct Hindi text to analyze")
    parser.add_argument("--file", type=str, help="Path to a UTF-8 text file to analyze")
    parser.add_argument("--out", type=str, default="", help="Write result JSON to this path")
    parser.add_argument("--model", type=str, default="reedmayhew/claude-3.7-sonnet-reasoning-gemma3-12B",
                        help="HF repo id of Claude 3.7 Sonnet Reasoning Gemma3 12B model")
    parser.add_argument("--max_tokens", type=int, default=768, help="Max new tokens for the LLM response")
    parser.add_argument("--temp", type=float, default=0.2, help="Sampling temperature")

    args = parser.parse_args()

    if not args.text and not args.file:
        print("Please provide --text or --file", file=sys.stderr)
        sys.exit(2)

    if args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}", file=sys.stderr)
            sys.exit(2)
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text

    text = stdre.sub(r"[ \t]+", " ", text).strip()

    try:
        result = extract(text, args.model)
    except KeyboardInterrupt:
        print("Cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Inference failed: {e}", file=sys.stderr)
        sys.exit(1)

    js = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(js)
        print(f"Saved: {args.out}")
    else:
        print(js)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hindi Text Extractor using Mistral-7B (Instruct) on Apple Silicon (MLX)

Tested on macOS (Apple Silicon). Designed for offline inference after the first download.

Dependencies (install once):
    python3 -m pip install mlx-lm huggingface_hub regex

Model (default; quantized for Apple Silicon):
    mlx-community/Mistral-7B-Instruct-v0.3-4bit

Usage examples:
    # Analyze direct text
    python hindi_mistral_extractor.py --text "अयोध्या ज़िले में थाना कैंट क्षेत्र में विवाद हुआ..." --out result.json

    # Analyze text from a file
    python hindi_mistral_extractor.py --file input.txt --out result.json

    # Change model (any MLX-compatible Mistral Instruct)
    python hindi_mistral_extractor.py --text "..." --model mlx-community/Mistral-7B-Instruct-v0.3-8bit

Notes:
    • First run downloads the model from Hugging Face and caches it locally.
    • Subsequent runs work fully offline (no network needed).
    • On Apple Silicon with 36 GB RAM, 4-bit is recommended for comfort.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import textwrap
from typing import Dict, Any, List
import re as stdre

# Use the 'regex' package for robust Unicode handling (e.g., Devanagari hashtags)
try:
    import regex as re
except Exception:
    re = stdre

# Hugging Face download (first run only, then fully offline via cache)
from huggingface_hub import snapshot_download

# Apple MLX LLM loader/generator
from mlx_lm import load, generate


# --------------------------- Heuristics & Dictionaries ---------------------------

# Common religion names (Hindi + English variants)
RELIGIONS = [
    "हिन्दू", "हिंदू", "मुस्लिम", "इस्लाम", "ईसाई", "क्रिश्चियन", "सिख", "बौद्ध", "जैन",
    "पारसी", "ज़रथुस्त्र", "यहूदी",
    "Hindu", "Muslim", "Islam", "Christian", "Sikh", "Buddhist", "Jain", "Parsi", "Zoroastrian", "Jew", "Jewish"
]

# Common caste/community names (Indic + English spellings; non-exhaustive)
CASTES = [
    "SC", "ST", "OBC", "General", "अनुसूचित", "जनजाति", "पिछड़ा", "सामान्य",
    "ब्राह्मण", "क्षत्रिय", "ठाकुर", "राजपूत", "यादव", "जाटव", "दलित", "कुर्मी", "लोधी", "कुशवाहा",
    "पटेल", "गुप्ता", "बनिया", "वैश्य", "कायस्थ", "मौर्य", "कश्यप", "निषाद", "पासी", "कोरी",
    "वर्मा", "मीना", "सैनी", "राठौर", "धोबी", "धनुक", "कुम्हार", "माली", "लोहा", "सोनी", "बाल्मीकि",
    "भाट", "खटीक", "भुर्जी", "तांती", "लुहार", "पाल", "यादव/अहीर"
]

# Uttar Pradesh districts (75); add more states if needed
UP_DISTRICTS = [
    "आगरा","अलीगढ़","अम्बेडकर नगर","अमेठी","अमरोहा","औरैया","अयोध्या","आजमगढ़","बहराइच","बरेली",
    "बस्ती","बागपत","बलिया","बांदा","बुलंदशहर","बाराबंकी","बिजनौर","चंदौली","चित्रकूट","देवरिया",
    "एटा","इटावा","फ़र्रुख़ाबाद","फतेहपुर","फ़िरोज़ाबाद","गाजियाबाद","गाजीपुर","गोंडा","गोरखपुर","हमीरपुर",
    "हापुड़","हरदोई","हाथरस","जौनपुर","झांसी","कन्नौज","कानपुर देहात","कानपुर नगर","कौशाम्बी","कुशीनगर",
    "लखीमपुर खीरी","ललितपुर","लखनऊ","महाराजगंज","महामाया नगर","मऊ","मैनपुरी","मेरठ","मिर्ज़ापुर","मुरादाबाद",
    "मुज़फ़्फरनगर","पीलीभीत","प्रतापगढ़","प्रयागराज","रामपुर","रायबरेली","सहारनपुर","संतकबीर नगर","संत रविदास नगर",
    "शाहजहाँपुर","शामली","श्रावस्ती","सिद्धार्थनगर","सीतापुर","सोनभद्र","सुल्तानपुर","उन्नाव","वाराणसी","बदायूं",
    "बलरामपुर","कासगंज","अयोध्या (फ़ैज़ाबाद)"
]

DISTRICT_VARIANTS = {
    "इलाहाबाद": "प्रयागराज",
    "फ़ैज़ाबाद": "अयोध्या",
    "Faizabad": "अयोध्या",
    "Allahabad": "प्रयागराज",
    "Gautam Buddha Nagar": "गौतम बुद्ध नगर",
    "Ghaziabad": "गाजियाबाद",
}

THANA_PATTERNS = [
    r"(?:थाना|कोतवाली|Kotwali|Thana|PS)\s+([^\s,।:-]+(?:\s+[^\s,।:-]+)?)"
]


# --------------------------- LLM Prompting (Mistral) ---------------------------

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
    # captures Latin, digits, underscore and Devanagari
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
            # sanitize trailing punctuation
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
    """Build an [INST]...[/INST] prompt for Mistral Instruct models.
    We enforce strict JSON output in the schema above (no prose)."""
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
{ text.strip() }
"""
    # Mistral Instruct chat template
    return f"<s>[INST]{instructions.strip()}[/INST]"


def safe_json_parse(s: str) -> Dict[str, Any]:
    s = s.strip()
    # Strip surrounding fences if any
    if s.startswith("```"):
        s = s.strip("`")
        # Sometimes it's like ```json ... ```
        s = stdre.sub(r"^json", "", s, flags=stdre.IGNORECASE).strip()
    # Attempt parse; fallback to empty schema
    try:
        return json.loads(s)
    except Exception:
        # Try to cut before stray tokens
        try:
            idx = s.rfind("}")
            if idx != -1:
                return json.loads(s[:idx+1])
        except Exception:
            pass
    return {}


def merge_results(llm_json: Dict[str, Any], regex_boost: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(JSON_SCHEMA)  # shallow template
    # Merge list fields
    list_keys = [
        "person_names","organisation_names","location_names","district_names",
        "thana_names","incidents","caste_names","religion_names","hashtags",
        "mention_ids","events"
    ]
    for k in list_keys:
        vals = []
        v1 = llm_json.get(k, []) if isinstance(llm_json.get(k, []), list) else []
        v2 = regex_boost.get(k, []) if isinstance(regex_boost.get(k, []), list) else []
        vals.extend([str(x).strip() for x in v1 if x])
        vals.extend([str(x).strip() for x in v2 if x])
        out[k] = dedupe(vals)

    # Sentiment
    s = llm_json.get("sentiment", {})
    if isinstance(s, dict) and "label" in s and "confidence" in s:
        out["sentiment"] = {"label": str(s.get("label")), "confidence": float(s.get("confidence"))}
    else:
        out["sentiment"] = {"label": "neutral", "confidence": 0.5}

    # Context
    ctx = llm_json.get("contextual_understanding", "")
    out["contextual_understanding"] = str(ctx).strip()

    return out


def run_mistral_infer(model_id: str, prompt: str, max_tokens: int = 1024, temp: float = 0.2) -> str:
    """
    Loads the MLX model (downloading once if needed) and generates output for the prompt.
    Returns raw text.
    """
    # Ensure local snapshot exists (first run will download; later runs are offline)
    local_path = snapshot_download(repo_id=model_id, local_files_only=False)
    # Load from local snapshot
    model, tokenizer = load(local_path)
    # Generate deterministically-ish for extraction
    out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    return out


def extract(text: str, model_id: str) -> Dict[str, Any]:
    # Regex/dictionary boosts (deterministic helpers)
    regex_boost = {
        "hashtags": find_hashtags(text),
        "mention_ids": find_mentions(text),
        "district_names": find_districts(text),
        "thana_names": find_thana(text),
        "caste_names": find_keywords(text, CASTES),
        "religion_names": find_keywords(text, RELIGIONS),
    }

    # LLM parse
    prompt = build_inst_prompt(text)
    raw = run_mistral_infer(model_id, prompt)
    llm_json = safe_json_parse(raw)

    # Merge + return
    final = merge_results(llm_json, regex_boost)
    return final


def main():
    parser = argparse.ArgumentParser(description="Hindi Text -> Structured Extraction via Mistral-7B (MLX)")
    parser.add_argument("--text", type=str, help="Direct Hindi text to analyze")
    parser.add_argument("--file", type=str, help="Path to a UTF-8 text file to analyze")
    parser.add_argument("--out", type=str, default="", help="Write result JSON to this path")
    parser.add_argument("--model", type=str, default="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                        help="HF repo id of an MLX-compatible Mistral Instruct model")
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

    # Normalize whitespace
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
# python script.py --text "मीरजापुर पुलिस* प्रेस नोट दिनांक 04.09.2025 थाना चिल्ह साइबर क्राइम टीम द्वारा ऑनलाइन ठगी की घटना से सम्बन्धित सम्पूर्ण धनराशि ₹ 18000/- को पीड़ित के खाते में कराया गया वापस— आवेदक दिनेश कुमार द्वारा थाना चिल्ह पर शिकायत सं. 23108250126760 के सन्दर्भ में शिकायती प्रार्थना पत्र दिया गया कि सोलर उर्जा लगाने के नाम पर पैसो की ठगी की गयी । जिसपर थाना चिल्ह साइबर क्राइम टीम द्वारा तत्काल सम्बन्धित बैंक को मेल किया गया । उक्त के सम्बन्ध में थाना चिल्ह की साइबर सेल टीम द्वारा जाँच प्रारम्भ की गयी । सोमेन बर्मा"वरिष्ठ पुलिस अधीक्षक मीरजापुर के निर्देशन में अपर पुलिस अधीक्षक नगर तथा क्षेत्राधिकारी सदर के नेतृत्व में थाना चिल्ह की साइबर पुलिस टीम द्वारा आवेदक के बैंक खाते में कुल ₹ 18000/- वापस कराए गये । खाते में पैसे वापस आने पर आवेदक द्वारा थाने पर उपस्थित होकर मीरजापुर पुलिस के पुलिस उच्चाधिकारीगण एवं साइबर क्राइम टीम थाना अहरौरा पुलिस टीम की भूरि-भूरि प्रशंसा करते हुए धन्यवाद दिया गया । आवेदक को साइबर जागरुकता अभियान के तहत साइबर सम्बन्धी होने वाली ऑनलाइन ठगी/धोखाधड़ी की घटना के सम्बन्ध में जागरुक किया गया । साइबर क्राइम टीम थाना चिल्ह— साइबर प्रभारी  उ0नि0मनोज सिंह व महिला आरक्षी सोनाली पाण्डेय ।" --out result.json

# python script.py --text "जनपदीय एसओजी व थाना कबरई की संयुक्त पुलिस टीम ने फर्जी रायल्टी प्रकरण से सम्बन्धित 02 नफर वांछित अभियुक्तों को किया गिरफ्तार, जिनके कब्जे से फर्जी रायल्टी पेपर किये गये बरामद...।  * पुलिस अधीक्षक महोबा, श्री प्रबल प्रताप सिंह के निर्देशन में जनपद महोबा में अपराध की रोकथाम हेतु आपराधिक घटनाओं में संलिप्त वांछित अभियुक्तों की चेकिंग एवं उनकी गिरफ्तारी/बरामदगी हेतु चलाये जा रहे अभियान के अनुपालन में अपर पुलिस अधीक्षक, श्रीमती वन्दना सिंह व क्षेत्राधिकारी नगर श्री दीपक दुबे के निकट पर्यवेक्षण में फर्जी रायल्टी प्रकरण के सम्बन्ध में पंजीकृत अभियोग में वांछित चल रहे अभियुक्तों की गिरफ्तारी हेतु टीमें गठित की गई थी । * जिसके क्रम में आज दिनांक 04.09.2025 को थानाध्यक्ष कबरई श्री सत्यवेन्द्र सिंह भदौरिया व जनपदीय एसओजी की संयुक्त टीम के द्वारा थाना स्थानीय पर पंजीकृत मु0अ0सं0 246/25 धारा 305/318(4)/338/336(3)/340(2)/317(2)/3(5) बीएनएस से सम्बन्धित 02 नफर वांछित अभियुक्तगण क्रमशः 1.अभिषेक शिवहरे पुत्र हरिकिशन उम्र करीब 25 वर्ष निवासी मु0 बैबेथोक कस्बा व थाना मटौंध जनपद बांदा 2.राजाबाबू पुत्र घासीराम अहिरवार उम्र करीब 21 वर्ष निवासी ग्राम लिलवाही थाना कबरई जनपद महोबा को पहरा रोड रेलवे क्रासिंग के पास से नियमानुसार गिरफ्तार किया गया। जिनके कब्जे से 09 अदद फर्जी रायल्टी पेपर व 02 अदद मोबाइल एंड्राइड बरामद हुआ। बाद आवश्यक कार्यवाही अभियुक्तगण उपरोक्त को मा0 न्यायालय के समक्ष पेशी हेतु भेजा गया।   गिरफ्तार अभियुक्तगण-          1.अभिषेक शिवहरे पुत्र हरिकिशन उम्र करीब 25 वर्ष निवासी मु0 बैबेथोक कस्बा व थाना मटौंध जनपद बांदा         2.राजाबाबू पुत्र घासीराम अहिरवार उम्र करीब 21 वर्ष निवासी ग्राम लिलवाही थाना कबरई जनपद महोबा  गिरफ्तार करने वाली संयुक्त पुलिस टीम का विवरण- जनपदीय एसओजी टीम- 1. उ0नि0 विवेक यादव  2. कां0 दीपक वर्मा 3. कां0 आशीष बघेल 4. कां0 कुलदीप यादव  5. कां0 सत्यम सिंह जादौन 6. कां0 निर्भय 7. कां0 अभिषेक दुबे  थाना कबरई पुलिस टीम- 1. थानाध्यक्ष श्री सत्यवेन्द्र सिंह भदौरिया 2. व0उ0नि0 मलखान सिंह 3. कां0 धर्मेन्द्र" --out result.json


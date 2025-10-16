"""
District Normalization Service
Ensures ALL districts are stored and matched in canonical English form
Handles Hindi/English/variant mappings
"""

import logging
from typing import Optional, List, Dict
from difflib import get_close_matches

logger = logging.getLogger(__name__)


class DistrictNormalizer:
    """
    Canonical district name normalization
    Converts all variants (Hindi/English/typos) to standard English names
    """

    # Master mapping: ALL variants → One English canonical form
    DISTRICT_MAPPINGS = {
        # Lucknow
        "लखनऊ": "Lucknow",
        "lucknow": "Lucknow",
        "LUCKNOW": "Lucknow",
        "लखनउ": "Lucknow",
        "lakhnau": "Lucknow",

        # Moradabad
        "मुरादाबाद": "Moradabad",
        "moradabad": "Moradabad",
        "MORADABAD": "Moradabad",
        "मोरादाबाद": "Moradabad",
        "muradabad": "Moradabad",

        # Agra
        "आगरा": "Agra",
        "agra": "Agra",
        "AGRA": "Agra",
        "आगर": "Agra",

        # Varanasi
        "वाराणसी": "Varanasi",
        "varanasi": "Varanasi",
        "VARANASI": "Varanasi",
        "बनारस": "Varanasi",
        "banaras": "Varanasi",
        "benares": "Varanasi",

        # Kanpur
        "कानपुर": "Kanpur",
        "kanpur": "Kanpur",
        "KANPUR": "Kanpur",

        # Prayagraj
        "प्रयागराज": "Prayagraj",
        "prayagraj": "Prayagraj",
        "PRAYAGRAJ": "Prayagraj",
        "इलाहाबाद": "Prayagraj",
        "allahabad": "Prayagraj",
        "illahabad": "Prayagraj",

        # Ghaziabad
        "गाजियाबाद": "Ghaziabad",
        "ghaziabad": "Ghaziabad",
        "GHAZIABAD": "Ghaziabad",
        "गाज़ियाबाद": "Ghaziabad",

        # Meerut
        "मेरठ": "Meerut",
        "meerut": "Meerut",
        "MEERUT": "Meerut",

        # Gorakhpur
        "गोरखपुर": "Gorakhpur",
        "gorakhpur": "Gorakhpur",
        "GORAKHPUR": "Gorakhpur",

        # Bareilly
        "बरेली": "Bareilly",
        "bareilly": "Bareilly",
        "BAREILLY": "Bareilly",

        # Aligarh
        "अलीगढ़": "Aligarh",
        "aligarh": "Aligarh",
        "ALIGARH": "Aligarh",

        # Saharanpur
        "सहारनपुर": "Saharanpur",
        "saharanpur": "Saharanpur",
        "SAHARANPUR": "Saharanpur",

        # Firozabad
        "फिरोजाबाद": "Firozabad",
        "firozabad": "Firozabad",
        "FIROZABAD": "Firozabad",

        # Jhansi
        "झांसी": "Jhansi",
        "jhansi": "Jhansi",
        "JHANSI": "Jhansi",

        # Muzaffarnagar
        "मुजफ्फरनगर": "Muzaffarnagar",
        "muzaffarnagar": "Muzaffarnagar",
        "MUZAFFARNAGAR": "Muzaffarnagar",

        # Mathura
        "मथुरा": "Mathura",
        "mathura": "Mathura",
        "MATHURA": "Mathura",

        # Budaun
        "बदायूं": "Budaun",
        "budaun": "Budaun",
        "BUDAUN": "Budaun",
        "badaun": "Budaun",

        # Rampur
        "रामपुर": "Rampur",
        "rampur": "Rampur",
        "RAMPUR": "Rampur",

        # Shahjahanpur
        "शाहजहांपुर": "Shahjahanpur",
        "shahjahanpur": "Shahjahanpur",
        "SHAHJAHANPUR": "Shahjahanpur",

        # Farrukhabad
        "फर्रुखाबाद": "Farrukhabad",
        "farrukhabad": "Farrukhabad",
        "FARRUKHABAD": "Farrukhabad",

        # Etawah
        "इटावा": "Etawah",
        "etawah": "Etawah",
        "ETAWAH": "Etawah",

        # Sitapur
        "सीतापुर": "Sitapur",
        "sitapur": "Sitapur",
        "SITAPUR": "Sitapur",

        # Hardoi
        "हरदोई": "Hardoi",
        "hardoi": "Hardoi",
        "HARDOI": "Hardoi",

        # Unnao
        "उन्नाव": "Unnao",
        "unnao": "Unnao",
        "UNNAO": "Unnao",

        # Rae Bareli
        "रायबरेली": "Raebareli",
        "raebareli": "Raebareli",
        "RAEBARELI": "Raebareli",
        "rae bareli": "Raebareli",

        # Faizabad (now Ayodhya)
        "फैजाबाद": "Ayodhya",
        "faizabad": "Ayodhya",
        "FAIZABAD": "Ayodhya",
        "अयोध्या": "Ayodhya",
        "ayodhya": "Ayodhya",
        "AYODHYA": "Ayodhya",

        # Ambedkar Nagar
        "अम्बेडकर नगर": "Ambedkar Nagar",
        "ambedkar nagar": "Ambedkar Nagar",
        "AMBEDKAR NAGAR": "Ambedkar Nagar",

        # Sultanpur
        "सुल्तानपुर": "Sultanpur",
        "sultanpur": "Sultanpur",
        "SULTANPUR": "Sultanpur",

        # Bahraich
        "बहराइच": "Bahraich",
        "bahraich": "Bahraich",
        "BAHRAICH": "Bahraich",

        # Shravasti
        "श्रावस्ती": "Shravasti",
        "shravasti": "Shravasti",
        "SHRAVASTI": "Shravasti",

        # Balrampur
        "बलरामपुर": "Balrampur",
        "balrampur": "Balrampur",
        "BALRAMPUR": "Balrampur",

        # Gonda
        "गोंडा": "Gonda",
        "gonda": "Gonda",
        "GONDA": "Gonda",

        # Siddharthnagar
        "सिद्धार्थनगर": "Siddharthnagar",
        "siddharthnagar": "Siddharthnagar",
        "SIDDHARTHNAGAR": "Siddharthnagar",

        # Basti
        "बस्ती": "Basti",
        "basti": "Basti",
        "BASTI": "Basti",

        # Sant Kabir Nagar
        "संत कबीर नगर": "Sant Kabir Nagar",
        "sant kabir nagar": "Sant Kabir Nagar",
        "SANT KABIR NAGAR": "Sant Kabir Nagar",

        # Maharajganj
        "महराजगंज": "Maharajganj",
        "maharajganj": "Maharajganj",
        "MAHARAJGANJ": "Maharajganj",

        # Kushinagar
        "कुशीनगर": "Kushinagar",
        "kushinagar": "Kushinagar",
        "KUSHINAGAR": "Kushinagar",

        # Deoria
        "देवरिया": "Deoria",
        "deoria": "Deoria",
        "DEORIA": "Deoria",

        # Azamgarh
        "आजमगढ़": "Azamgarh",
        "azamgarh": "Azamgarh",
        "AZAMGARH": "Azamgarh",

        # Mau
        "मऊ": "Mau",
        "mau": "Mau",
        "MAU": "Mau",

        # Ballia
        "बलिया": "Ballia",
        "ballia": "Ballia",
        "BALLIA": "Ballia",

        # Jaunpur
        "जौनपुर": "Jaunpur",
        "jaunpur": "Jaunpur",
        "JAUNPUR": "Jaunpur",

        # Ghazipur
        "गाजीपुर": "Ghazipur",
        "ghazipur": "Ghazipur",
        "GHAZIPUR": "Ghazipur",

        # Chandauli
        "चंदौली": "Chandauli",
        "chandauli": "Chandauli",
        "CHANDAULI": "Chandauli",

        # Bhadohi
        "भदोही": "Bhadohi",
        "bhadohi": "Bhadohi",
        "BHADOHI": "Bhadohi",
        "sant ravidas nagar": "Bhadohi",

        # Mirzapur
        "मिर्जापुर": "Mirzapur",
        "mirzapur": "Mirzapur",
        "MIRZAPUR": "Mirzapur",

        # Sonbhadra
        "सोनभद्र": "Sonbhadra",
        "sonbhadra": "Sonbhadra",
        "SONBHADRA": "Sonbhadra",

        # Pratapgarh
        "प्रतापगढ़": "Pratapgarh",
        "pratapgarh": "Pratapgarh",
        "PRATAPGARH": "Pratapgarh",

        # Kaushambi
        "कौशाम्बी": "Kaushambi",
        "kaushambi": "Kaushambi",
        "KAUSHAMBI": "Kaushambi",

        # Fatehpur
        "फतेहपुर": "Fatehpur",
        "fatehpur": "Fatehpur",
        "FATEHPUR": "Fatehpur",

        # Banda
        "बांदा": "Banda",
        "banda": "Banda",
        "BANDA": "Banda",

        # Chitrakoot
        "चित्रकूट": "Chitrakoot",
        "chitrakoot": "Chitrakoot",
        "CHITRAKOOT": "Chitrakoot",

        # Hamirpur
        "हमीरपुर": "Hamirpur",
        "hamirpur": "Hamirpur",
        "HAMIRPUR": "Hamirpur",

        # Mahoba
        "महोबा": "Mahoba",
        "mahoba": "Mahoba",
        "MAHOBA": "Mahoba",

        # Lalitpur
        "ललितपुर": "Lalitpur",
        "lalitpur": "Lalitpur",
        "LALITPUR": "Lalitpur",

        # Jalaun
        "जालौन": "Jalaun",
        "jalaun": "Jalaun",
        "JALAUN": "Jalaun",

        # Kannauj
        "कन्नौज": "Kannauj",
        "kannauj": "Kannauj",
        "KANNAUJ": "Kannauj",

        # Etah
        "एटा": "Etah",
        "etah": "Etah",
        "ETAH": "Etah",

        # Mainpuri
        "मैनपुरी": "Mainpuri",
        "mainpuri": "Mainpuri",
        "MAINPURI": "Mainpuri",

        # Auraiya
        "औरैया": "Auraiya",
        "auraiya": "Auraiya",
        "AURAIYA": "Auraiya",

        # Kanpur Dehat
        "कानपुर देहात": "Kanpur Dehat",
        "kanpur dehat": "Kanpur Dehat",
        "KANPUR DEHAT": "Kanpur Dehat",

        # Kasganj
        "कासगंज": "Kasganj",
        "kasganj": "Kasganj",
        "KASGANJ": "Kasganj",

        # Hathras
        "हाथरस": "Hathras",
        "hathras": "Hathras",
        "HATHRAS": "Hathras",

        # Pilibhit
        "पीलीभीत": "Pilibhit",
        "pilibhit": "Pilibhit",
        "PILIBHIT": "Pilibhit",

        # Lakhimpur Kheri
        "लखीमपुर खीरी": "Lakhimpur Kheri",
        "lakhimpur kheri": "Lakhimpur Kheri",
        "LAKHIMPUR KHERI": "Lakhimpur Kheri",
        "kheri": "Lakhimpur Kheri",

        # Bijnor
        "बिजनौर": "Bijnor",
        "bijnor": "Bijnor",
        "BIJNOR": "Bijnor",

        # Amroha
        "अमरोहा": "Amroha",
        "amroha": "Amroha",
        "AMROHA": "Amroha",

        # Sambhal
        "संभल": "Sambhal",
        "sambhal": "Sambhal",
        "SAMBHAL": "Sambhal",

        # Bulandshahr
        "बुलंदशहर": "Bulandshahr",
        "bulandshahr": "Bulandshahr",
        "BULANDSHAHR": "Bulandshahr",

        # Hapur
        "हापुड़": "Hapur",
        "hapur": "Hapur",
        "HAPUR": "Hapur",

        # Gautam Buddha Nagar (Noida)
        "गौतम बुद्ध नगर": "Gautam Buddha Nagar",
        "gautam buddha nagar": "Gautam Buddha Nagar",
        "GAUTAM BUDDHA NAGAR": "Gautam Buddha Nagar",
        "noida": "Gautam Buddha Nagar",
        "NOIDA": "Gautam Buddha Nagar",
        "नोएडा": "Gautam Buddha Nagar",

        # Bagpat
        "बागपत": "Baghpat",
        "bagpat": "Baghpat",
        "BAGPAT": "Baghpat",
        "Baghpat": "Baghpat",

        # Shamli
        "शामली": "Shamli",
        "shamli": "Shamli",
        "SHAMLI": "Shamli",

        # Amethi
        "अमेठी": "Amethi",
        "amethi": "Amethi",
        "AMETHI": "Amethi",

        #Barabanki
        "Barabanki": "बाराबंकी",
    }

    # Reverse mapping for display purposes (English → Hindi)
    CANONICAL_TO_HINDI = {
                            "Agra": "आगरा",
                            "Aligarh": "अलीगढ़",
                            "Ambedkar Nagar": "अम्बेडकर नगर",
                            "Amethi": "अमेठी",
                            "Amroha": "अमरोहा",
                            "Auraiya": "औरैया",
                            "Ayodhya": "अयोध्या",
                            "Azamgarh": "आजमगढ़",
                            "Baghpat": "बागपत",
                            "Bahraich": "बहराइच",
                            "Ballia": "बलिया",
                            "Balrampur": "बलरामपुर",
                            "Banda": "बांदा",
                            "Barabanki": "बाराबंकी",
                            "Bareilly": "बरेली",
                            "Basti": "बस्ती",
                            "Bhadohi": "भदोही",
                            "Bijnor": "बिजनौर",
                            "Budaun": "बदायूं",
                            "Bulandshahr": "बुलंदशहर",
                            "Chandauli": "चंदौली",
                            "Chitrakoot": "चित्रकूट",
                            "Deoria": "देवरिया",
                            "Etah": "एटा",
                            "Etawah": "इटावा",
                            "Farrukhabad": "फर्रुखाबाद",
                            "Fatehpur": "फतेहपुर",
                            "Firozabad": "फिरोजाबाद",
                            "Gautam Buddha Nagar": "गौतम बुद्ध नगर",
                            "Ghaziabad": "गाजियाबाद",
                            "Ghazipur": "गाजीपुर",
                            "Gonda": "गोंडा",
                            "Gorakhpur": "गोरखपुर",
                            "Hamirpur": "हमीरपुर",
                            "Hapur": "हापुड़",
                            "Hardoi": "हरदोई",
                            "Hathras": "हाथरस",
                            "Jalaun": "जालौन",
                            "Jaunpur": "जौनपुर",
                            "Jhansi": "झांसी",
                            "Kannauj": "कन्नौज",
                            "Kanpur Dehat": "कानपुर देहात",
                            "Kanpur": "कानपुर",
                            "Kasganj": "कासगंज",
                            "Kaushambi": "कौशाम्बी",
                            "Kushinagar": "कुशीनगर",
                            "Lakhimpur Kheri": "लखीमपुर खीरी",
                            "Lalitpur": "ललितपुर",
                            "Lucknow": "लखनऊ",
                            "Mahoba": "महोबा",
                            "Maharajganj": "महराजगंज",
                            "Mainpuri": "मैनपुरी",
                            "Mathura": "मथुरा",
                            "Mau": "मऊ",
                            "Meerut": "मेरठ",
                            "Mirzapur": "मिर्जापुर",
                            "Moradabad": "मुरादाबाद",
                            "Muzaffarnagar": "मुजफ्फरनगर",
                            "Pilibhit": "पीलीभीत",
                            "Pratapgarh": "प्रतापगढ़",
                            "Prayagraj": "प्रयागराज",
                            "Raebareli": "रायबरेली",
                            "Rampur": "रामपुर",
                            "Saharanpur": "सहारनपुर",
                            "Sambhal": "संभल",
                            "Sant Kabir Nagar": "संत कबीर नगर",
                            "Shahjahanpur": "शाहजहांपुर",
                            "Shamli": "शामली",
                            "Shravasti": "श्रावस्ती",
                            "Siddharthnagar": "सिद्धार्थनगर",
                            "Sitapur": "सीतापुर",
                            "Sonbhadra": "सोनभद्र",
                            "Sultanpur": "सुल्तानपुर",
                            "Unnao": "उन्नाव",
                            "Varanasi": "वाराणसी"
                        }


    @classmethod
    def normalize(cls, district_name: str) -> Optional[str]:
        """
        Convert ANY district variant to canonical English form

        Examples:
            normalize("लखनऊ") → "Lucknow"
            normalize("lucknow") → "Lucknow"
            normalize("MORADABAD") → "Moradabad"

        Args:
            district_name: Input district name in any form

        Returns:
            Canonical English district name or None if not found
        """
        if not district_name:
            return None

        district_clean = district_name.strip()

        # Direct lookup (fastest)
        canonical = cls.DISTRICT_MAPPINGS.get(district_clean)
        if canonical:
            return canonical

        # Case-insensitive fallback
        district_lower = district_clean.lower()
        for variant, canonical in cls.DISTRICT_MAPPINGS.items():
            if variant.lower() == district_lower:
                return canonical

        # Fuzzy matching for typos (last resort)
        canonical_list = list(set(cls.DISTRICT_MAPPINGS.values()))
        matches = get_close_matches(district_clean, canonical_list, n=1, cutoff=0.85)

        if matches:
            logger.info(f"Fuzzy matched '{district_clean}' to '{matches[0]}'")
            return matches[0]

        logger.warning(f"Could not normalize district: '{district_clean}'")
        return None

    @classmethod
    def normalize_list(cls, districts: List[str]) -> List[str]:
        """
        Normalize a list of districts, removing duplicates

        Args:
            districts: List of district names in various forms

        Returns:
            List of canonical English district names (deduplicated)
        """
        normalized = []
        seen = set()

        for district in districts:
            canonical = cls.normalize(district)
            if canonical and canonical not in seen:
                normalized.append(canonical)
                seen.add(canonical)

        return normalized

    @classmethod
    def get_hindi_name(cls, canonical_name: str) -> str:
        """
        Get Hindi display name for canonical English name

        Args:
            canonical_name: English canonical district name

        Returns:
            Hindi name or original if no mapping exists
        """
        return cls.CANONICAL_TO_HINDI.get(canonical_name, canonical_name)

    @classmethod
    def get_all_canonical_districts(cls) -> List[str]:
        """Get list of all canonical district names"""
        return sorted(set(cls.DISTRICT_MAPPINGS.values()))

    @classmethod
    def is_valid_district(cls, district_name: str) -> bool:
        """Check if district name is valid (can be normalized)"""
        return cls.normalize(district_name) is not None
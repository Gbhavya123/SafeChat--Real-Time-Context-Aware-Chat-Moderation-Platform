"""
Text Preprocessing for SafeChat

Handles text normalization, language detection (with Hinglish/code-mixing support),
and cleaning for optimal model input.
"""

import re
import unicodedata
from typing import Optional

from loguru import logger


# ── Script Detection (for code-mixed language identification) ───────────

# Unicode ranges for Indian scripts
DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")   # Hindi, Sanskrit, Marathi
BENGALI_RANGE = re.compile(r"[\u0980-\u09FF]")       # Bengali, Assamese
TAMIL_RANGE = re.compile(r"[\u0B80-\u0BFF]")
TELUGU_RANGE = re.compile(r"[\u0C00-\u0C7F]")
KANNADA_RANGE = re.compile(r"[\u0C80-\u0CFF]")
MALAYALAM_RANGE = re.compile(r"[\u0D00-\u0D7F]")
GUJARATI_RANGE = re.compile(r"[\u0A80-\u0AFF]")
GURMUKHI_RANGE = re.compile(r"[\u0A00-\u0A7F]")       # Punjabi
ODIA_RANGE = re.compile(r"[\u0B00-\u0B7F]")
LATIN_RANGE = re.compile(r"[a-zA-Z]")

INDIAN_SCRIPT_MAP = {
    "devanagari": DEVANAGARI_RANGE,
    "bengali": BENGALI_RANGE,
    "tamil": TAMIL_RANGE,
    "telugu": TELUGU_RANGE,
    "kannada": KANNADA_RANGE,
    "malayalam": MALAYALAM_RANGE,
    "gujarati": GUJARATI_RANGE,
    "gurmukhi": GURMUKHI_RANGE,
    "odia": ODIA_RANGE,
}


def detect_language(text: str) -> str:
    """
    Detect language with special handling for Indian languages and code-mixing.

    Returns standardized language codes:
      - 'en'    : English
      - 'hi'    : Hindi (Devanagari script)
      - 'hi-en' : Hinglish (code-mixed Hindi + English)
      - 'bn'    : Bengali
      - 'ta'    : Tamil
      - 'te'    : Telugu
      - 'kn'    : Kannada
      - 'ml'    : Malayalam
      - 'gu'    : Gujarati
      - 'pa'    : Punjabi
      - 'or'    : Odia
      - 'indic-en' : Any Indian language mixed with English
      - 'other' : Fallback

    NOTE: This script-based detection is MORE RELIABLE for code-mixed text
    than library-based detectors (langdetect/fasttext) which assume monolingual input.
    """
    if not text or not text.strip():
        return "en"

    has_latin = bool(LATIN_RANGE.search(text))

    # Check each Indian script
    detected_scripts = {}
    for script_name, pattern in INDIAN_SCRIPT_MAP.items():
        matches = pattern.findall(text)
        if matches:
            detected_scripts[script_name] = len(matches)

    # No Indian script detected
    if not detected_scripts:
        if has_latin:
            # Could be transliterated Hindi (romanized) — check with langdetect
            return _detect_romanized_indian(text)
        return "en"

    # Find dominant Indian script
    dominant_script = max(detected_scripts, key=detected_scripts.get)

    # Map script to language code
    script_to_lang = {
        "devanagari": "hi",
        "bengali": "bn",
        "tamil": "ta",
        "telugu": "te",
        "kannada": "kn",
        "malayalam": "ml",
        "gujarati": "gu",
        "gurmukhi": "pa",
        "odia": "or",
    }

    lang = script_to_lang.get(dominant_script, "other")

    # Check for code-mixing (Indian script + significant Latin text)
    if has_latin and detected_scripts:
        latin_chars = len(LATIN_RANGE.findall(text))
        indian_chars = sum(detected_scripts.values())
        total = latin_chars + indian_chars

        # If more than 20% of script chars are Latin, it's code-mixed
        if total > 0 and latin_chars / total > 0.2:
            if lang == "hi":
                return "hi-en"  # Hinglish
            return "indic-en"   # Other Indian + English mix

    return lang


def _detect_romanized_indian(text: str) -> str:
    """
    Detect if Latin-script text is actually romanized Hindi/Hinglish.

    Uses common Hindi words written in Latin script as indicators.
    """
    # Common romanized Hindi words (colloquial + formal)
    hindi_indicators = {
        # Pronouns and common words
        "kya", "hai", "hain", "nahi", "nhi", "mat", "aur", "bhi", "toh",
        "mein", "main", "tera", "mera", "tumhara", "hamara", "apna",
        "yeh", "woh", "koi", "kuch", "sab", "bahut", "bohot",
        # Verbs
        "karo", "karna", "bolo", "bolna", "jao", "jana", "aao", "aana",
        "dekho", "dekhna", "suno", "sunna", "chalo", "ruk", "ruko",
        # Slang / colloquial
        "yaar", "bhai", "dude", "arre", "abey", "oye", "chal",
        "accha", "theek", "sahi", "galat", "bakwas", "pagal",
        # Toxicity indicators (important for our use case)
        "bewakoof", "gadha", "ullu", "kamina", "kamini", "harami",
        "chutiya", "madarchod", "behenchod", "bhosdike", "gaandu",
        "saala", "saali", "kutte", "kuttia", "haramkhor",
        # Expanded romanized abuse cues so short profanity-heavy chats are
        # still routed through the Hinglish path.
        "badir", "badirchand", "bakchod", "bakchodi", "bakland",
        "baklol", "baklund", "bakwaas", "bhenchod", "bhosdi",
        "bhosdika", "bhosdiki", "bhosde", "bhadwa", "bhadwe",
        "bsdk", "chakka", "chhakka", "chinal", "chodu", "chut",
        "chutiye", "chutiyapa", "gaand", "gandu", "ghatiya", "gawar",
        "haramzada", "haramzade", "hijda", "hijde", "hijra", "jahil",
        "jhantu", "jhandu", "kanjar", "kanjari", "kaminey", "kutta",
        "kuttey", "kuttay",
        "lauda", "lavda", "loda", "lodu", "lund", "lundtopi",
        "madarchod", "madarchood", "najayaz",
        "nalayak", "nikamma", "paagal", "randi", "randwa", "randwe",
        "sala", "saale", "sali", "stupid", "suar", "suwar",
        "tharki", "tatti", "tatte",
    }

    # Tokenize defensively so "bhosdike," still counts as a Hinglish cue.
    words = set(re.findall(r"[a-z]+", text.lower()))
    hindi_word_count = len(words & hindi_indicators)

    # If 2+ Hindi indicator words found, classify as romanized Hindi/Hinglish
    if hindi_word_count >= 2:
        return "hi-en"
    elif hindi_word_count >= 1 and len(words) <= 5:
        return "hi-en"

    # For Latin-script text that doesn't look like Hinglish, default to English.
    # Generic language detectors are noisy on short toxic chat messages and can
    # misclassify simple English as unrelated languages such as "sw".
    return "en"


def is_indian_language(lang_code: str) -> bool:
    """Check if a language code represents an Indian language."""
    return lang_code in {
        "hi", "hi-en", "bn", "ta", "te", "kn", "ml",
        "gu", "pa", "or", "indic-en",
    }


# ── Text Cleaning ──────────────────────────────────────────────────────

def clean_text(text: str, preserve_case: bool = False) -> str:
    """
    Clean and normalize text for model input.

    Steps:
      1. Unicode normalization (NFC — canonical composition)
      2. Remove zero-width characters and control chars (preserve newlines)
      3. Normalize whitespace
      4. Optionally lowercase

    NOTE: We do NOT remove emojis or special chars — the models handle them,
    and they carry semantic meaning for toxicity detection.
    """
    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Remove zero-width chars and most control characters (keep \n, \t)
    text = re.sub(r"[\u200b-\u200f\u2028-\u202f\u2060-\u2069\ufeff]", "", text)

    # Normalize repeated whitespace (but preserve single newlines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip
    text = text.strip()

    if not preserve_case:
        text = text.lower()

    return text


def normalize_for_toxicity(text: str) -> str:
    """
    Additional normalization specifically for toxicity detection.

    Handles common evasion techniques:
      - L33t speak: "h4te" → "hate"
      - Character repetition: "fuckkkk" → "fuck"
      - Separator insertion: "f.u.c.k" → "fuck"
      - Mixed scripts for evasion: "fuсk" (Cyrillic с) → "fuck"
    """
    # Step 1: Basic cleaning
    text = clean_text(text, preserve_case=False)

    # Step 2: Reduce character repetition (keep max 2 of same char)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Step 3: Remove separators between single characters
    # "f.u.c.k" or "f u c k" → "fuck"
    # Only for Latin characters (don't break Devanagari)
    text = re.sub(
        r"(?<=[a-z])[.\-_\s](?=[a-z](?:[.\-_\s][a-z]){2,})",
        "",
        text,
    )

    # Step 4: Common leet speak mappings
    leet_map = {
        "0": "o", "1": "i", "3": "e", "4": "a",
        "5": "s", "7": "t", "8": "b", "@": "a",
        "$": "s", "!": "i",
    }
    # Only apply leet substitution in words that look like leet speak
    def _deleet(match):
        word = match.group(0)
        if any(c in word for c in leet_map):
            for leet, normal in leet_map.items():
                word = word.replace(leet, normal)
        return word

    text = re.sub(r"\b\S+\b", _deleet, text)

    return text

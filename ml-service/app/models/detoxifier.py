"""
SafeChat — Text Detoxifier (IndicBART Generation)

Converts toxic Hindi/Hinglish/English sentences to polite versions.
Uses the original downloaded IndicBART model with prompt-based generation.
"""

import re
from typing import Dict, List, Optional
from loguru import logger
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.config import settings
from app.utils.preprocessing import detect_language

LANGUAGE_TAGS = {
    "en": "<2en>",
    "hi": "<2hi>",
    "hi-en": "<2en>",
    "indic-en": "<2en>",
}

POLITE_FALLBACKS = {
    "en": "Could you please say that more politely and respectfully?",
    "hi": "कृपया यही बात थोड़े विनम्र और सम्मानजनक तरीके से कहें।",
    "hi-en": "Please isi baat ko thoda politely aur respectfully bolo.",
    "indic-en": "Please say this more politely and respectfully.",
}


class Detoxifier:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._model_loaded = False
        logger.info("Detoxifier initialized (IndicBART mode).")

    def load_model(self) -> None:
        if not settings.USE_MODEL_DETOX:
            return

        try:
            logger.info(f"Loading IndicBART detox model: {settings.DETOX_MODEL}...")
            self._tokenizer = AutoTokenizer.from_pretrained(settings.DETOX_MODEL, use_fast=False)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(settings.DETOX_MODEL)
            self._model.to(settings.DEVICE)
            self._model.eval()
            self._model_loaded = True
            logger.success(f"IndicBART Detox model loaded successfully on {settings.DEVICE}.")
        except Exception as e:
            logger.warning(f"Failed to load IndicBART: {e}")
            self._model_loaded = False

    def detoxify(
        self,
        text: str,
        toxicity_categories: Optional[Dict[str, float]] = None,
        target_language: Optional[str] = None,
        preserve_intent: bool = True,
    ) -> Dict:
        lang = self._normalize_language(target_language or detect_language(text))
        dominant_category = self._dominant_category(toxicity_categories)

        if toxicity_categories and max(toxicity_categories.values(), default=0.0) < self._safe_threshold_for_language(lang):
            return {
                "original": text,
                "detoxified": text,
                "suggestions": [text],
                "method": "passthrough",
                "language": lang,
                "confidence": 1.0,
            }

        generated = None
        if self._model_loaded and settings.USE_MODEL_DETOX:
            generated = self._model_detoxify(text, lang)

        suggestions = self._build_suggestions(
            original_text=text,
            lang=lang,
            dominant_category=dominant_category,
            generated=generated,
            preserve_intent=preserve_intent,
        )
        primary = suggestions[0] if suggestions else self._fallback_rewrite(lang)

        return {
            "original": text,
            "detoxified": primary,
            "suggestions": suggestions,
            "method": "indic_bart" if generated else "template",
            "language": lang,
            "confidence": 0.85 if generated else 0.60,
        }

    def _model_detoxify(self, text: str, lang: str) -> Optional[str]:
        if not self._model or not self._tokenizer:
            return None

        try:
            prompt = self._build_prompt(text, lang)

            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                max_length=settings.MAX_SEQ_LENGTH,
                truncation=True,
            )
            inputs = {k: v.to(settings.DEVICE) for k, v in inputs.items()}

            forced_bos_token_id = self._forced_bos_token_id(lang)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=settings.DETOX_MAX_LENGTH,
                    num_beams=settings.DETOX_NUM_BEAMS,
                    min_new_tokens=8,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    length_penalty=1.0,
                    early_stopping=True,
                    forced_bos_token_id=forced_bos_token_id,
                )

            result = self._tokenizer.decode(outputs[0], skip_special_tokens=False)
            cleaned = self._clean_generation(result, prompt, text)
            return cleaned

        except Exception as e:
            logger.error(f"IndicBART detoxification failed: {e}")
            return None

    def get_info(self) -> Dict:
        return {
            "mode": "indic_bart" if self._model_loaded else "fallback",
            "model_name": settings.DETOX_MODEL if self._model_loaded else None,
        }

    @staticmethod
    def _normalize_language(lang: str) -> str:
        if lang in {"hi", "hi-en", "indic-en", "en"}:
            return lang
        return "en"

    @staticmethod
    def _safe_threshold_for_language(lang: str) -> float:
        if lang == "hi-en":
            return settings.THRESHOLD_SAFE_HINGLISH
        return settings.THRESHOLD_SAFE

    def _forced_bos_token_id(self, lang: str) -> Optional[int]:
        if not self._tokenizer:
            return None
        token = LANGUAGE_TAGS.get(lang, "<2en>")
        token_id = self._tokenizer.convert_tokens_to_ids(token)
        return token_id if isinstance(token_id, int) and token_id >= 0 else None

    @staticmethod
    def _build_prompt(text: str, lang: str) -> str:
        if lang == "hi":
            return (
                "इस अपमानजनक वाक्य को विनम्र और सम्मानजनक हिंदी में दोबारा लिखो.\n"
                f"अपमानजनक: {text}\n"
                "विनम्र:"
            )
        if lang == "hi-en":
            return (
                "Is toxic text ko polite aur respectful Hinglish mein Roman script me rewrite karo.\n"
                f"Toxic: {text}\n"
                "Polite Hinglish:"
            )
        return (
            "Rewrite the following toxic text into polite and respectful English.\n"
            f"Toxic: {text}\n"
            "Polite:"
        )

    @staticmethod
    def _fallback_rewrite(lang: str) -> str:
        return POLITE_FALLBACKS.get(lang, POLITE_FALLBACKS["en"])

    @staticmethod
    def _dominant_category(toxicity_categories: Optional[Dict[str, float]]) -> str:
        if not toxicity_categories:
            return "toxic"
        for label in ("identity_hate", "threat", "obscene", "insult"):
            if toxicity_categories.get(label, 0.0) >= 0.30:
                return label
        return max(toxicity_categories, key=toxicity_categories.get)

    def _build_suggestions(
        self,
        original_text: str,
        lang: str,
        dominant_category: str,
        generated: Optional[str],
        preserve_intent: bool,
    ) -> List[str]:
        suggestions: List[str] = []
        if generated:
            suggestions.append(generated)

        suggestions.extend(self._template_suggestions(lang, dominant_category, preserve_intent))
        if not suggestions:
            suggestions.append(self._fallback_rewrite(lang))

        deduped: List[str] = []
        seen = set()
        original_normalized = original_text.strip().lower()
        for suggestion in suggestions:
            cleaned = suggestion.strip()
            if not cleaned:
                continue
            normalized = cleaned.lower()
            if normalized == original_normalized:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(cleaned)
            if len(deduped) == 3:
                break

        return deduped or [self._fallback_rewrite(lang)]

    def _template_suggestions(self, lang: str, dominant_category: str, preserve_intent: bool) -> List[str]:
        category = dominant_category if dominant_category in {"insult", "obscene", "threat", "identity_hate"} else "toxic"

        if lang == "hi":
            templates = {
                "toxic": [
                    "कृपया यही बात थोड़े विनम्र और सम्मानजनक तरीके से कहें।",
                    "अपनी बात बिना अपमानजनक भाषा के भी कही जा सकती है।",
                    "कृपया बातचीत में शिष्ट और सम्मानजनक भाषा रखें।",
                ],
                "insult": [
                    "मैं असहमत हूँ, लेकिन कृपया सम्मानजनक भाषा रखें।",
                    "नाराज़गी जताइए, पर व्यक्तिगत अपमान मत कीजिए।",
                    "कृपया मुद्दे पर बात करें, व्यक्ति पर नहीं।",
                ],
                "obscene": [
                    "कृपया अश्लील शब्दों के बिना अपनी बात कहें।",
                    "आप नाराज़ हो सकते हैं, लेकिन भाषा मर्यादित रखें।",
                    "अपना संदेश साफ और सम्मानजनक तरीके से रखें।",
                ],
                "threat": [
                    "धमकी देने के बजाय शांत और स्पष्ट तरीके से अपनी बात रखें।",
                    "मैं बहुत नाराज़ हूँ, लेकिन धमकी नहीं देना चाहता।",
                    "हिंसा या डराने वाली भाषा से बचते हुए बात करें।",
                ],
                "identity_hate": [
                    "कृपया किसी की पहचान या समुदाय को निशाना बनाए बिना अपनी बात कहें।",
                    "असहमति व्यक्त की जा सकती है, लेकिन घृणित भाषा ठीक नहीं है।",
                    "हर व्यक्ति और समुदाय के प्रति सम्मानजनक भाषा रखें।",
                ],
            }
        elif lang == "hi-en":
            templates = {
                "toxic": [
                    "Please isi baat ko thoda politely aur respectfully bolo.",
                    "Apni baat bina gaali diye bhi clear tarah se boli ja sakti hai.",
                    "Chalo baat ko thoda calm aur respectful tareeke se rakhte hain.",
                ],
                "insult": [
                    "Main disagree karta hoon, lekin personal insult ke bina bolunga.",
                    "Please gusse mein bhi respectfully baat karo.",
                    "Issue par bolo, insaan ko target mat karo.",
                ],
                "obscene": [
                    "Please abusive words hata kar apni baat bolo.",
                    "Frustration dikhani hai to bhi language clean rakho.",
                    "Same point ko thoda decent aur respectful Hinglish mein bolo.",
                ],
                "threat": [
                    "Main upset hoon, lekin dhamki nahi dunga.",
                    "Please threatening language ke bina apni baat bolo.",
                    "Chalo isko calmly resolve karte hain, darane wali language ke bina.",
                ],
                "identity_hate": [
                    "Please kisi identity ya community ko target kiye bina baat karo.",
                    "Disagree karna theek hai, hate speech nahi.",
                    "Respectful Hinglish mein point rakho, identity-based abuse ke bina.",
                ],
            }
        else:
            templates = {
                "toxic": [
                    "Could you please say that more politely and respectfully?",
                    "You can make the same point without abusive language.",
                    "Let's keep the conversation respectful.",
                ],
                "insult": [
                    "I disagree, but I want to say it respectfully.",
                    "Please speak respectfully even if you're upset.",
                    "Let's discuss the issue without personal insults.",
                ],
                "obscene": [
                    "Please say this without profanity.",
                    "You can express frustration without abusive words.",
                    "Let's keep the wording clean and respectful.",
                ],
                "threat": [
                    "I am upset, but I will not use threats.",
                    "Please address this calmly without threatening language.",
                    "Let's resolve this without violence or intimidation.",
                ],
                "identity_hate": [
                    "Please make your point without targeting anyone's identity.",
                    "Disagreement is fine, but hateful language is not.",
                    "Let's keep this respectful toward every person and community.",
                ],
            }

        selected = list(templates.get(category, templates["toxic"]))
        if not preserve_intent:
            selected[0] = self._fallback_rewrite(lang)
        return selected

    @staticmethod
    def _clean_generation(result: str, prompt: str, original_text: str) -> Optional[str]:
        cleaned = result
        cleaned = re.sub(r"</?s>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<2[a-z]+>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace("[CLS]", " ").replace("[SEP]", " ").replace("<pad>", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        for marker in ("Polite Hinglish:", "Polite:", "विनम्र:", "अपमानजनक:", "Toxic:"):
            if marker in cleaned:
                cleaned = cleaned.split(marker)[-1].strip()

        if not cleaned:
            return None

        # Reject degenerate outputs that mostly echo the prompt or collapse into punctuation.
        lowered = cleaned.lower()
        if len(cleaned) < 4:
            return None
        if re.fullmatch(r"[\W_]+", cleaned):
            return None
        if any(
            token in lowered
            for token in (
                "rewrite the following",
                "polite hinglish",
                "toxic:",
                "tox",
                "polite",
                "rewrite",
                "अपमानजनक",
            )
        ):
            return None
        if lowered == original_text.lower().strip():
            return None
        if re.search(r"\b(\w+)(?:\s+\1){2,}\b", lowered):
            return None

        return cleaned

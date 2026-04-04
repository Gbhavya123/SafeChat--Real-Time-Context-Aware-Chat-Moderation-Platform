"""
SafeChat — Toxicity Classifier (Single MuRIL Model)

Architecture:
  ┌─────────┐
  │  Input  │ (Hindi, English, Hinglish)
  └────┬────┘
       │
       ▼
  ┌───────────────────────┐
  │ google/muril-base     │ (SequenceClassifier)
  └───────┬───────────────┘
          │ Output Logits
          ▼
  ┌───────────────────────┐
  │ Sigmoid Activation    │ 
  └───────┬───────────────┘
          ▼
  ┌───────────────────────┐
  │ Labels & Severity     │
  └───────────────────────┘
"""

import time
import re
from typing import Dict, Optional
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
)
from loguru import logger

from app.config import settings
from app.utils.preprocessing import clean_text, detect_language, normalize_for_toxicity

# Labels as defined in our training data
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

LOW_INSULT_FLOORS = {"toxic": 0.24, "insult": 0.21}
MEDIUM_INSULT_FLOORS = {"toxic": 0.38, "insult": 0.34}
STRONG_INSULT_FLOORS = {"toxic": 0.44, "insult": 0.38}
OBSCENE_FLOORS = {"toxic": 0.68, "obscene": 0.62, "insult": 0.56}
SEVERE_OBSCENE_FLOORS = {"toxic": 0.74, "obscene": 0.68, "insult": 0.60}
IDENTITY_SLUR_FLOORS = {"toxic": 0.44, "insult": 0.38, "identity_hate": 0.34}

HINGLISH_LOW_INSULT_WORDS = (
    "badir",
    "badirchand",
    "baklol",
    "baklund",
    "bakwas",
    "bakwaas",
    "fatu",
    "fattu",
    "ghatiya",
    "gawar",
    "jahil",
    "jhandu",
    "jhant",
    "jhantu",
    "jhaantu",
    "nalayak",
    "nikamma",
    "pagal",
    "paagal",
    "sala",
    "saala",
    "saale",
    "sali",
    "saali",
    "ullu",
)

HINGLISH_MEDIUM_INSULT_WORDS = (
    "bastard",
    "bewakoof",
    "bewaqoof",
    "bewakuf",
    "bevakuf",
    "chinal",
    "gadha",
    "gadhe",
    "gadhi",
    "idiot",
    "jerk",
    "kamina",
    "kamine",
    "kaminey",
    "kamini",
    "kamino",
    "kanjar",
    "kanjari",
    "kanjaron",
    "kutta",
    "kutte",
    "kuttey",
    "kuttay",
    "kutti",
    "kuttia",
    "kuttiya",
    "kutiya",
    "loser",
    "moron",
    "rascal",
    "stupid",
    "suar",
    "suvar",
    "suwar",
    "sooar",
    "tharki",
)

HINGLISH_STRONG_INSULT_WORDS = (
    "bhadva",
    "bhadve",
    "bhadwa",
    "bhadwe",
    "bhadwi",
    "harami",
    "haraami",
    "haramkhor",
    "haramzada",
    "haramzade",
    "haramzaadi",
    "haramzadi",
    "najayaz",
    "najayz",
    "randi",
    "randwa",
    "randwe",
)

HINGLISH_IDENTITY_SLUR_WORDS = (
    "chakka",
    "chakke",
    "chhakka",
    "chhakke",
    "hijda",
    "hijde",
    "hijra",
    "hijre",
)

HINGLISH_OBSCENE_WORDS = (
    "asshole",
    "bakchod",
    "bakchodi",
    "bsdk",
    "chodra",
    "chodu",
    "choot",
    "chosi",
    "chot",
    "chuth",
    "chut",
    "chutiya",
    "chutiye",
    "chutiyagiri",
    "chutiyapa",
    "chutiyon",
    "chutia",
    "gaand",
    "gaandu",
    "gand",
    "gandu",
    "lawda",
    "lauda",
    "laude",
    "lavda",
    "loda",
    "lode",
    "lodu",
    "lund",
    "lundi",
    "lundtopi",
    "tatta",
    "tatte",
    "tattey",
    "tatti",
)

HINGLISH_SEVERE_OBSCENE_WORDS = (
    "allahchodi",
    "allachodi",
    "bahanchod",
    "bahenchod",
    "banchod",
    "behenchod",
    "behenchhod",
    "benchod",
    "bhencho",
    "bhenchod",
    "bhenchhod",
    "bhnchod",
    "bhosad",
    "bhosada",
    "bhosadi",
    "bhosadike",
    "bhosadika",
    "bhosadiwala",
    "bhosadiwale",
    "bhosde",
    "bhosday",
    "bhosdi",
    "bhosdika",
    "bhosdiki",
    "bhosdike",
    "bhosdiwala",
    "bhosdiwale",
    "bhosdiwaloon",
    "bhosri",
    "bhosrik",
    "bhosriwala",
    "bhosriwale",
    "bitch",
    "cunt",
    "dickhead",
    "dipshit",
    "fucker",
    "fucking",
    "madarchod",
    "madarchood",
    "madarjaat",
    "madarjat",
    "motherfucker",
    "motherfucking",
    "scumbag",
    "slut",
    "whore",
)

HINDI_LOW_INSULT_WORDS = (
    "अक्लहीन",
    "बकवास",
    "बेकार",
    "जाहिल",
    "घटिया",
    "पागल",
    "नालायक",
    "निकम्मा",
    "निकम्मे",
)

HINDI_MEDIUM_INSULT_WORDS = (
    "बेवकूफ",
    "गधा",
    "गधे",
    "गधी",
    "कमीना",
    "कमीने",
    "कमीनी",
    "कुत्ता",
    "कुत्ते",
    "कुतिया",
    "मूर्ख",
    "गंवार",
    "बदतमीज",
)

HINDI_STRONG_INSULT_WORDS = (
    "हरामी",
    "हरामखोर",
    "हरामज़ादा",
    "हरामजादा",
    "हरामज़ादी",
    "हरामजादी",
    "रंडी",
    "नीच",
)

HINDI_IDENTITY_SLUR_WORDS = (
    "छक्का",
    "छक्के",
    "हिजड़ा",
    "हिजड़े",
    "हिजरा",
    "हिजरे",
)

HINDI_OBSCENE_WORDS = (
    "चूत",
    "चूतिया",
    "चूतिये",
    "चूतियापा",
    "गांड",
    "गाण्ड",
    "गांडू",
    "गाण्डू",
    "लंड",
    "लौड़ा",
    "लौड़े",
    "लवड़ा",
    "टट्टी",
    "टट्टे",
)

HINDI_SEVERE_OBSCENE_WORDS = (
    "बहनचोद",
    "भोसड़ी",
    "भोसडी",
    "भोसड़ीके",
    "भोसडीके",
    "मादरचोद",
    "मदरचोद",
    "मादरजात",
)

ENGLISH_TOKEN_FLOORS = {
    "asshole": {"toxic": 0.68, "obscene": 0.62, "insult": 0.56},
    "bastard": {"toxic": 0.38, "insult": 0.34},
    "bitch": {"toxic": 0.68, "obscene": 0.62, "insult": 0.56},
    "bullshit": {"toxic": 0.44, "obscene": 0.38, "insult": 0.34},
    "crap": {"toxic": 0.34, "insult": 0.31},
    "dickhead": {"toxic": 0.68, "obscene": 0.62, "insult": 0.56},
    "dipshit": {"toxic": 0.68, "obscene": 0.62, "insult": 0.56},
    "dumbass": {"toxic": 0.44, "insult": 0.38},
    "fucker": {"toxic": 0.74, "obscene": 0.68, "insult": 0.60},
    "fucking": {"toxic": 0.68, "obscene": 0.62, "insult": 0.56},
    "idiot": {"toxic": 0.38, "insult": 0.34},
    "jackass": {"toxic": 0.44, "insult": 0.38},
    "jerk": {"toxic": 0.34, "insult": 0.31},
    "loser": {"toxic": 0.38, "insult": 0.34},
    "moron": {"toxic": 0.38, "insult": 0.34},
    "prick": {"toxic": 0.44, "insult": 0.38},
    "retard": {"toxic": 0.44, "insult": 0.38, "identity_hate": 0.34},
    "scumbag": {"toxic": 0.44, "insult": 0.38},
    "shithead": {"toxic": 0.68, "obscene": 0.62, "insult": 0.56},
    "slut": {"toxic": 0.68, "obscene": 0.62, "insult": 0.56},
    "stupid": {"toxic": 0.34, "insult": 0.31},
    "trash": {"toxic": 0.34, "insult": 0.31},
    "whore": {"toxic": 0.68, "obscene": 0.62, "insult": 0.56},
}

ENGLISH_PHRASE_FLOORS = {
    "drop dead": {"toxic": 0.44, "threat": 0.38},
    "go kill yourself": {"toxic": 0.56, "threat": 0.52},
    "go to hell": {"toxic": 0.38, "insult": 0.34},
    "i will kill you": {"toxic": 0.56, "threat": 0.52},
    "i'll kill you": {"toxic": 0.56, "threat": 0.52},
    "piece of shit": {"toxic": 0.68, "obscene": 0.62, "insult": 0.56},
    "shut the fuck up": {"toxic": 0.74, "obscene": 0.68, "insult": 0.60},
    "shut up": {"toxic": 0.34, "insult": 0.31},
    "son of a bitch": {"toxic": 0.68, "obscene": 0.62, "insult": 0.56},
}


def _build_token_floors(grouped_words: tuple[tuple[tuple[str, ...], Dict[str, float]], ...]) -> Dict[str, Dict[str, float]]:
    floors: Dict[str, Dict[str, float]] = {}
    for words, labels in grouped_words:
        for word in words:
            floors[word] = dict(labels)
    return floors


HINGLISH_TOKEN_FLOORS = _build_token_floors(
    (
        (HINGLISH_LOW_INSULT_WORDS, LOW_INSULT_FLOORS),
        (HINGLISH_MEDIUM_INSULT_WORDS, MEDIUM_INSULT_FLOORS),
        (HINGLISH_STRONG_INSULT_WORDS, STRONG_INSULT_FLOORS),
        (HINGLISH_IDENTITY_SLUR_WORDS, IDENTITY_SLUR_FLOORS),
        (HINGLISH_OBSCENE_WORDS, OBSCENE_FLOORS),
        (HINGLISH_SEVERE_OBSCENE_WORDS, SEVERE_OBSCENE_FLOORS),
    )
)

HINGLISH_PHRASE_FLOORS = {
    "behen chod": {"toxic": 0.74, "obscene": 0.68, "insult": 0.60},
    "bhen chod": {"toxic": 0.74, "obscene": 0.68, "insult": 0.60},
    "gaand mar": {"toxic": 0.68, "obscene": 0.62, "insult": 0.56},
    "jaan se maar": {"toxic": 0.56, "threat": 0.52},
    "kill kar dunga": {"toxic": 0.56, "threat": 0.52},
    "kutte ki aulad": {"toxic": 0.44, "insult": 0.38},
    "maar dunga": {"toxic": 0.52, "threat": 0.48},
    "madar chod": {"toxic": 0.74, "obscene": 0.68, "insult": 0.60},
    "mar dunga": {"toxic": 0.52, "threat": 0.48},
    "mother fucker": {"toxic": 0.74, "obscene": 0.68, "insult": 0.60},
    "piece of shit": {"toxic": 0.68, "obscene": 0.62, "insult": 0.56},
    "son of a bitch": {"toxic": 0.68, "obscene": 0.62, "insult": 0.56},
    "suwar ki aulad": {"toxic": 0.44, "insult": 0.38},
    "teri behen": {"toxic": 0.44, "insult": 0.38},
    "teri ma": {"toxic": 0.44, "insult": 0.38},
    "teri maa": {"toxic": 0.44, "insult": 0.38},
    "tu mar ja": {"toxic": 0.38, "threat": 0.34},
    "ullu ka pattha": {"toxic": 0.38, "insult": 0.34},
}

HINDI_TOKEN_FLOORS = _build_token_floors(
    (
        (HINDI_LOW_INSULT_WORDS, LOW_INSULT_FLOORS),
        (HINDI_MEDIUM_INSULT_WORDS, MEDIUM_INSULT_FLOORS),
        (HINDI_STRONG_INSULT_WORDS, STRONG_INSULT_FLOORS),
        (HINDI_IDENTITY_SLUR_WORDS, IDENTITY_SLUR_FLOORS),
        (HINDI_OBSCENE_WORDS, OBSCENE_FLOORS),
        (HINDI_SEVERE_OBSCENE_WORDS, SEVERE_OBSCENE_FLOORS),
    )
)

HINDI_PHRASE_FLOORS = {
    "जा मर": {"toxic": 0.38, "threat": 0.34},
    "जान से मार दूंगा": {"toxic": 0.56, "threat": 0.52},
    "जान से मार दूँगा": {"toxic": 0.56, "threat": 0.52},
    "तेरी बहन": {"toxic": 0.44, "insult": 0.38},
    "तेरी मां": {"toxic": 0.44, "insult": 0.38},
    "तेरी माँ": {"toxic": 0.44, "insult": 0.38},
    "मार दूंगा": {"toxic": 0.52, "threat": 0.48},
    "मार दूँगा": {"toxic": 0.52, "threat": 0.48},
}

class ToxicityClassifier:
    """
    Toxicity classifier powered by fine-tuned MuRIL base.
    """

    def __init__(self, model_name: str = settings.CLASSIFIER_MODEL, device: str = settings.DEVICE):
        self.device = device
        self._loaded = False
        self._model_name = model_name

        self.tokenizer = None
        self.model = None
        self.model_version = settings.APP_VERSION

    def load(self) -> None:
        """Load MuRIL model into memory."""
        logger.info(f"Loading tokenizer and model ({self._model_name}) on {self.device}...")
        try:
            # MuRIL checkpoints on lightweight containers can fail through AutoTokenizer even when
            # the underlying vocab is present, so we fall back to the explicit BERT tokenizer.
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self._model_name, use_fast=False)
            except Exception as tokenizer_error:
                logger.warning(
                    f"AutoTokenizer failed for {self._model_name}: {tokenizer_error}. "
                    "Falling back to BertTokenizer."
                )
                self.tokenizer = BertTokenizer.from_pretrained(self._model_name, do_lower_case=False)
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self._model_name,
                    num_labels=len(LABELS),
                    problem_type="multi_label_classification"
                )
            except Exception as model_error:
                logger.warning(
                    f"AutoModelForSequenceClassification failed for {self._model_name}: {model_error}. "
                    "Falling back to BertForSequenceClassification."
                )
                self.model = BertForSequenceClassification.from_pretrained(
                    self._model_name,
                    num_labels=len(LABELS),
                    problem_type="multi_label_classification"
                )
            # If the model hasn't been fine-tuned yet, it will throw a warning about randomly initialized weights. 
            # We explicitly ignore here for first-time startup.
            self.model.to(self.device)
            self.model.eval()

            self._loaded = True
            logger.success("MuRIL toxicity model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load MuRIL model: {e}")
            raise e

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, text: str, context: Optional[list[str]] = None) -> Dict:
        """
        Classify text for toxicity using MuRIL.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call classifier.load() first.")

        start_time = time.perf_counter()

        # Step 1: Preprocess
        normalized = normalize_for_toxicity(text)
        lang = detect_language(text)  # "en", "hi", "hi-en"
        model_input = normalized

        # Incorporate up to 4 past messages for context if provided
        if context and len(context) > 0:
            context_str = " [SEP] ".join(context[-4:])
            model_input = f"{context_str} [SEP] {normalized}"

        # Step 2: Tokenize
        inputs = self.tokenizer(
            model_input,
            return_tensors="pt",
            max_length=settings.MAX_SEQ_LENGTH,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Step 3: Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Sigmoid is used for multi-label classification to get probabilities 0-1
            probs = torch.sigmoid(logits)[0].cpu().numpy().tolist()

        # Step 4: Map back to Categories
        categories = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
        categories = self._apply_hindi_lexicon_boost(normalized, categories, lang)
        categories = self._apply_hinglish_lexicon_boost(normalized, categories, lang)
        categories = self._apply_english_lexicon_boost(normalized, categories, lang)
        
        # Step 5: Overall score + severity
        overall_score = round(max(categories.values()), 4)
        severity = self._score_to_severity(overall_score, lang)
        categories = {label: round(score, 4) for label, score in categories.items()}

        inference_time_ms = int((time.perf_counter() - start_time) * 1000)

        return {
            "is_toxic": overall_score >= self._safe_threshold_for_language(lang),
            "overall_score": overall_score,
            "severity": severity,
            "categories": categories,
            "detected_language": lang,
            "ensemble_weights": {"muril": 1.0}, # Single model now
            "model_version": self.model_version,
            "inference_time_ms": inference_time_ms,
        }

    def predict_batch(self, texts: list[str]) -> list[Dict]:
        """Classify a batch of texts."""
        return [self.predict(text) for text in texts]

    @staticmethod
    def _safe_threshold_for_language(lang: str) -> float:
        if lang == "hi-en":
            return settings.THRESHOLD_SAFE_HINGLISH
        return settings.THRESHOLD_SAFE

    @classmethod
    def _score_to_severity(cls, score: float, lang: str) -> str:
        """Map a toxicity score to a severity level."""
        safe_threshold = cls._safe_threshold_for_language(lang)
        if score < safe_threshold:
            return "SAFE"
        elif score < settings.THRESHOLD_LOW:
            return "LOW"
        elif score < settings.THRESHOLD_MEDIUM:
            return "MEDIUM"
        else:
            return "HIGH"

    @staticmethod
    def _apply_hinglish_lexicon_boost(text: str, categories: Dict[str, float], lang: str) -> Dict[str, float]:
        """Boost missed Hinglish toxicity for common romanized abuse terms."""
        if lang != "hi-en":
            return categories

        return ToxicityClassifier._apply_lexicon_floors(
            text=text,
            categories=categories,
            token_floors=HINGLISH_TOKEN_FLOORS,
            phrase_floors=HINGLISH_PHRASE_FLOORS,
        )

    @staticmethod
    def _apply_hindi_lexicon_boost(text: str, categories: Dict[str, float], lang: str) -> Dict[str, float]:
        """Boost missed Hindi-script toxicity for common slurs, obscenity, and threat phrases."""
        if lang != "hi":
            return categories

        return ToxicityClassifier._apply_lexicon_floors(
            text=text,
            categories=categories,
            token_floors=HINDI_TOKEN_FLOORS,
            phrase_floors=HINDI_PHRASE_FLOORS,
        )

    @staticmethod
    def _apply_english_lexicon_boost(text: str, categories: Dict[str, float], lang: str) -> Dict[str, float]:
        """Boost obvious English profanity and insults the base model may under-score."""
        if lang != "en":
            return categories

        return ToxicityClassifier._apply_lexicon_floors(
            text=text,
            categories=categories,
            token_floors=ENGLISH_TOKEN_FLOORS,
            phrase_floors=ENGLISH_PHRASE_FLOORS,
        )

    @staticmethod
    def _apply_lexicon_floors(
        text: str,
        categories: Dict[str, float],
        token_floors: Dict[str, Dict[str, float]],
        phrase_floors: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """Apply token and phrase-based minimum scores for language-specific abuse lexicons."""
        boosted = dict(categories)
        lowered = text.lower()
        tokens = set(re.findall(r"[a-z\u0900-\u097f]+", lowered))

        for token, floors in token_floors.items():
            if token in tokens:
                for label, floor in floors.items():
                    boosted[label] = max(boosted[label], floor)

        for phrase, floors in phrase_floors.items():
            if phrase in lowered:
                for label, floor in floors.items():
                    boosted[label] = max(boosted[label], floor)

        return boosted

    def get_info(self) -> Dict:
        """Return model metadata for health checks."""
        return {
            "muril_model": {
                "name": self._model_name,
                "loaded": self._loaded,
            },
            "device": self.device,
            "version": self.model_version,
        }

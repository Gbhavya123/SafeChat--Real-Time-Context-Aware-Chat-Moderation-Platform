"""
SafeChat — Model Manager
Cenrtalized lifecycle manager for ML models.
"""

from typing import Dict, Optional
import torch
from loguru import logger

from app.config import settings
from app.models.toxicity_classifier import ToxicityClassifier
from app.models.detoxifier import Detoxifier

class ModelManager:
    """Singleton-style manager for Mutli-label Classification and Seq2Seq Detoxification."""

    def __init__(self):
        self.classifier: Optional[ToxicityClassifier] = None
        self.detoxifier: Optional[Detoxifier] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Load all models. Called once during FastAPI startup."""
        logger.info("=" * 60)
        logger.info("  SafeChat Model Manager — Initializing (MuRIL & IndicBART)")
        logger.info("=" * 60)
        self._log_hardware_info()

        # Load MuRIL
        try:
            self.classifier = ToxicityClassifier(
                model_name=settings.CLASSIFIER_MODEL,
                device=settings.DEVICE,
            )
            self.classifier.load()
        except Exception as e:
            logger.error(f"Failed to load toxicity classifier: {e}")
            raise RuntimeError(f"Classifier initialization failed: {e}")

        # Load IndicBART
        try:
            self.detoxifier = Detoxifier()
            self.detoxifier.load_model()
        except Exception as e:
            logger.warning(f"Detoxifier model loading failed: {e}. Template fallback will be used.")

        self._initialized = True
        logger.success("All models initialized successfully!")
        logger.info("=" * 60)

    @property
    def is_ready(self) -> bool:
        return self._initialized and self.classifier is not None and self.classifier.is_loaded

    def get_health(self) -> Dict:
        return {
            "status": "healthy" if self.is_ready else "degraded",
            "models": {
                "toxicity_classifier": self.classifier.get_info() if self.classifier else {"loaded": False},
                "detoxifier": self.detoxifier.get_info() if self.detoxifier else {"loaded": False},
            },
            "device": settings.DEVICE,
        }

    async def swap_classifier(self, new_model_path: str) -> bool:
        """Hot-swap the toxicity classifier."""
        logger.info(f"Hot-swapping classifier to: {new_model_path}")
        try:
            new_classifier = ToxicityClassifier(model_name=new_model_path, device=settings.DEVICE)
            new_classifier.load()
            old_classifier = self.classifier
            self.classifier = new_classifier
            del old_classifier
            if settings.DEVICE == "cuda":
                torch.cuda.empty_cache()
            return True
        except Exception as e:
            logger.error(f"Classifier hot-swap failed: {e}.")
            return False

    @staticmethod
    def _log_hardware_info():
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("No GPU detected. Running on CPU (slower inference).")
        logger.info(f"Selected device: {settings.DEVICE}")

model_manager = ModelManager()

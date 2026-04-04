"""
SafeChat ML Service — Configuration
"""

from pathlib import Path
from pydantic_settings import BaseSettings
import torch


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_DIR = BASE_DIR / "checkpoints"
DEFAULT_CLASSIFIER_PATH = DEFAULT_CHECKPOINT_DIR / "muril-toxicity-finetuned"
DEFAULT_DETOX_BASE_FRESH_PATH = DEFAULT_CHECKPOINT_DIR / "indicbart-base-fresh"
DEFAULT_DETOX_BASE_PATH = DEFAULT_CHECKPOINT_DIR / "indicbart-base"
DEFAULT_DETOX_FINAL_PATH = DEFAULT_CHECKPOINT_DIR / "indicbart-detox"
DEFAULT_DETOX_CHECKPOINT_PATH = DEFAULT_DETOX_FINAL_PATH / "checkpoint-1000"


def _prefer_local_checkpoint(local_paths: list[Path], fallback: str) -> str:
    """Use the first available local checkpoint, otherwise fall back."""
    for local_path in local_paths:
        has_config = (local_path / "config.json").exists()
        has_weights = (local_path / "model.safetensors").exists() or (local_path / "pytorch_model.bin").exists()
        if has_config and has_weights:
            return str(local_path)
    return fallback


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # ── App ──────────────────────────────────────────────
    APP_NAME: str = "SafeChat ML Service (MuRIL & IndicBART)"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Device ───────────────────────────────────────────
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Model Configuration ──────────────────────────────
    # Prefer saved fine-tuned checkpoints when available.
    CLASSIFIER_MODEL: str = _prefer_local_checkpoint(
        [DEFAULT_CLASSIFIER_PATH],
        "google/muril-base-cased",
    )
    
    # Use the original downloaded IndicBART model for prompt-based detoxification.
    DETOX_MODEL: str = _prefer_local_checkpoint(
        [
            DEFAULT_DETOX_BASE_FRESH_PATH,
            DEFAULT_DETOX_BASE_PATH,
            DEFAULT_DETOX_FINAL_PATH,
            DEFAULT_DETOX_CHECKPOINT_PATH,
        ],
        "ai4bharat/IndicBART",
    )
    DETOX_MAX_LENGTH: int = 128
    DETOX_NUM_BEAMS: int = 4
    USE_MODEL_DETOX: bool = True  # Enforce Model usage

    # ── Inference Settings ───────────────────────────────
    MAX_SEQ_LENGTH: int = 256
    BATCH_SIZE: int = 8

    # ── Severity Thresholds ──────────────────────────────
    THRESHOLD_SAFE: float = 0.30
    THRESHOLD_SAFE_HINGLISH: float = 0.18
    THRESHOLD_LOW: float = 0.55
    THRESHOLD_MEDIUM: float = 0.75

    # ── MongoDB & Continuous Learning ──────────────────────
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "safechat"
    FEEDBACK_THRESHOLD_FOR_RETRAIN: int = 500
    MODEL_CHECKPOINT_DIR: str = str(DEFAULT_CHECKPOINT_DIR)

    class Config:
        env_file = ".env"
        env_prefix = "SAFECHAT_"
        case_sensitive = True


# Singleton settings instance
settings = Settings()

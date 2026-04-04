"""
SafeChat — Moderation Service

Orchestrates the full moderation pipeline:
  1. Classify text for toxicity (ensemble)
  2. Generate polite alternative (if toxic)
  3. Return combined result

This is the main entry point called by the API routes.
"""

import time
from typing import Dict, Optional

from loguru import logger

from app.models.model_manager import model_manager
from app.schemas.moderation import ModerationResponse


class ModerationService:
    """
    Orchestrates toxicity classification + detoxification.

    Stateless service — all state lives in ModelManager.
    """

    @staticmethod
    async def moderate(
        text: str,
        context: Optional[list[str]] = None,
        channel_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ModerationResponse:
        """
        Full moderation pipeline for a single message.

        Args:
            text: Raw message text
            channel_id: Channel for policy lookup (future use)
            user_id: Sender ID for tracking (future use)

        Returns:
            ModerationResponse with toxicity scores and suggestion
        """
        start_time = time.perf_counter()

        # Step 1: Classify toxicity
        classifier = model_manager.classifier
        if not classifier or not classifier.is_loaded:
            raise RuntimeError("Toxicity classifier not available")

        classification = classifier.predict(text, context=context)

        # Step 2: Generate suggestion if toxic
        suggestion = None
        suggestions = None
        if classification["is_toxic"]:
            detoxifier = model_manager.detoxifier
            if detoxifier:
                detox_result = detoxifier.detoxify(
                    text=text,
                    toxicity_categories=classification["categories"],
                    target_language=classification["detected_language"],
                    preserve_intent=True,
                )
                suggestion = detox_result["detoxified"]
                suggestions = detox_result.get("suggestions")

        total_time_ms = int((time.perf_counter() - start_time) * 1000)

        return ModerationResponse(
            is_toxic=classification["is_toxic"],
            overall_score=classification["overall_score"],
            severity=classification["severity"],
            categories=classification["categories"],
            detected_language=classification["detected_language"],
            ensemble_weights=classification["ensemble_weights"],
            suggestion=suggestion,
            suggestions=suggestions,
            model_version=classification["model_version"],
            inference_time_ms=total_time_ms,
        )

    @staticmethod
    async def moderate_batch(
        texts: list[str],
        channel_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> list[ModerationResponse]:
        """Moderate multiple messages. Simple sequential for now."""
        results = []
        for text in texts:
            result = await ModerationService.moderate(
                text=text,
                channel_id=channel_id,
                user_id=user_id,
            )
            results.append(result)
        return results


# Singleton service instance
moderation_service = ModerationService()

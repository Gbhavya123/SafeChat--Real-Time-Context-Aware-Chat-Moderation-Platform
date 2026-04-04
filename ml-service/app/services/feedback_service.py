"""
SafeChat — Feedback Service

Handles moderator feedback storage and continuous learning triggers.
Feedback is stored in MongoDB and used to retrain models when
the threshold is reached.
"""

from typing import Dict, Optional
from datetime import datetime

from loguru import logger

from app.config import settings


class FeedbackService:
    """
    Manages the moderator feedback loop.

    Flow:
      1. Moderator reviews a flagged message
      2. Submits correction via POST /api/v1/feedback
      3. Feedback stored in MongoDB
      4. When feedback count reaches threshold → trigger retraining
    """

    def __init__(self):
        self._feedback_count = 0
        self._correct_count = 0
        self._incorrect_count = 0
        self._last_retrain_at: Optional[datetime] = None
        self._feedback_since_retrain = 0

        # In-memory store (replaced by MongoDB in production)
        self._feedback_store: list[Dict] = []

    async def submit_feedback(self, feedback: Dict) -> Dict:
        """
        Store moderator feedback and check if retraining should be triggered.

        Args:
            feedback: Dict with message_id, moderator_id, correct_label, etc.

        Returns:
            Dict with feedback_id, counts, and whether retrain was triggered
        """
        # Store feedback
        feedback_entry = {
            **feedback,
            "feedback_id": f"fb-{self._feedback_count + 1}",
            "submitted_at": datetime.utcnow().isoformat(),
        }
        self._feedback_store.append(feedback_entry)

        # Update counters
        self._feedback_count += 1
        self._feedback_since_retrain += 1

        if feedback.get("model_prediction_was_correct", True):
            self._correct_count += 1
        else:
            self._incorrect_count += 1

        # Check if retraining should be triggered
        retrain_triggered = False
        if self._feedback_since_retrain >= settings.FEEDBACK_THRESHOLD_FOR_RETRAIN:
            retrain_triggered = await self._trigger_retraining()

        logger.info(
            f"Feedback #{self._feedback_count} received. "
            f"Correct: {self._correct_count}, Incorrect: {self._incorrect_count}. "
            f"Retrain triggered: {retrain_triggered}"
        )

        return {
            "feedback_id": feedback_entry["feedback_id"],
            "message": "Feedback recorded successfully",
            "total_feedback_count": self._feedback_count,
            "retrain_threshold": settings.FEEDBACK_THRESHOLD_FOR_RETRAIN,
            "retrain_triggered": retrain_triggered,
        }

    async def get_stats(self) -> Dict:
        """Return feedback statistics."""
        accuracy = (
            self._correct_count / self._feedback_count
            if self._feedback_count > 0
            else 0.0
        )

        return {
            "total_feedback": self._feedback_count,
            "correct_predictions": self._correct_count,
            "incorrect_predictions": self._incorrect_count,
            "accuracy": round(accuracy, 4),
            "feedback_since_last_retrain": self._feedback_since_retrain,
            "retrain_threshold": settings.FEEDBACK_THRESHOLD_FOR_RETRAIN,
            "next_retrain_at": max(
                0,
                settings.FEEDBACK_THRESHOLD_FOR_RETRAIN - self._feedback_since_retrain,
            ),
            "last_retrain_at": self._last_retrain_at,
        }

    async def _trigger_retraining(self) -> bool:
        """
        Trigger model retraining.

        In production, this would:
          1. Export feedback data from MongoDB
          2. Combine with original training data
          3. Fine-tune models (background job)
          4. Evaluate new model on holdout set
          5. Hot-swap if improved
        """
        logger.warning(
            f"Retraining threshold reached ({self._feedback_since_retrain} feedback samples). "
            f"Triggering retraining pipeline..."
        )

        # TODO: Implement actual retraining pipeline
        # For now, just reset the counter and log
        self._feedback_since_retrain = 0
        self._last_retrain_at = datetime.utcnow()

        logger.info("Retraining pipeline placeholder executed. Reset feedback counter.")
        return True

    def get_training_data(self) -> list[Dict]:
        """Export feedback data for retraining."""
        return [
            fb for fb in self._feedback_store
            if not fb.get("model_prediction_was_correct", True)
        ]


# Singleton instance
feedback_service = FeedbackService()

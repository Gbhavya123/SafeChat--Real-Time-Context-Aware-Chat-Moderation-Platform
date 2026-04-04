"""
SafeChat — Detoxify API Route

POST /api/v1/detoxify — Generate a polite alternative for toxic text
"""

from fastapi import APIRouter, HTTPException

from app.models.model_manager import model_manager
from app.schemas.moderation import DetoxifyRequest, DetoxifyResponse

router = APIRouter(prefix="/api/v1", tags=["Detoxification"])


@router.post("/detoxify", response_model=DetoxifyResponse)
async def detoxify_text(request: DetoxifyRequest):
    """
    Generate a polite, non-toxic alternative for the given text.

    Supports English, Hindi, and Hinglish (code-mixed) text.
    Uses template-based suggestions (Phase 1) or model-based
    generation (Phase 2, when enabled).
    """
    if not model_manager.is_ready:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    detoxifier = model_manager.detoxifier
    if not detoxifier:
        raise HTTPException(status_code=503, detail="Detoxifier not available.")

    try:
        # First classify to know which category of toxicity
        classification = model_manager.classifier.predict(request.text)

        result = detoxifier.detoxify(
            text=request.text,
            toxicity_categories=classification["categories"],
            target_language=request.target_language,
            preserve_intent=request.preserve_intent,
        )

        return DetoxifyResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detoxification failed: {str(e)}")

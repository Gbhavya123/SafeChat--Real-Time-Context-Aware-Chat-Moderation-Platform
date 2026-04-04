---
title: SafeChat ML Service
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
---

# SafeChat ML Service

FastAPI-based toxicity classification and detoxification service for English, Hindi, and Hinglish.

## What This Space Exposes

- `GET /` service metadata
- `GET /docs` interactive API docs
- `POST /api/v1/moderate` toxicity classification with polite suggestions
- `POST /api/v1/moderate/batch` batch moderation
- `POST /api/v1/detoxify` detoxification-only endpoint
- `GET /api/v1/health` and `GET /api/v1/ready` health endpoints

## Model Setup

- Toxicity classifier: `vineet88/safechat-muril-toxicity-finetuned`
- Detoxifier: template fallback enabled by default in the Space for faster startup on CPU
- Optional detox model: `ai4bharat/IndicBART` can still be enabled by setting `SAFECHAT_USE_MODEL_DETOX=true`

## Notes

- The Space is packaged as a Docker Space.
- The container listens on port `8000`.
- The root endpoint returns JSON. API docs are available at `/docs`.
- The Space pulls the fine-tuned MuRIL checkpoint from a separate Hugging Face model repo at runtime.
- The default Space configuration favors faster boot over heavy seq2seq model loading.

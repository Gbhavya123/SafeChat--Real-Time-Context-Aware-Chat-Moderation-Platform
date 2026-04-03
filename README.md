# SafeChat--Real-Time-Context-Aware-Chat-Moderation-Platform !
SafeChat is an AI-driven moderation system designed to foster healthy digital conversations. Unlike traditional "block-and-delete" moderation, SafeChat detects toxic messages in real-time and uses advanced NLP models to rewrite them into safe, polite, and intent-preserving alternatives based on the community's context.

## Key Features

- Dynamic Environment-Awareness (Innovation): The system features toggleable moderation profiles.

  - Professional Mode: Strict moderation for professional settings; zero tolerance for unprofessional language.

  - Casual Mode: Adaptive moderation that understands "competitive banter" or "trash talk," allowing for a higher intensity of expression while still filtering out hate speech or harassment.

- Real-Time Detection: Instant classification of messages using Google's MuRIL.

- Intelligent Detoxification: Toxic inputs are automatically rewritten using Indic BERT to maintain the original meaning while removing harmful language.

- Context-Awareness: Optimized for Indian languages and English, understanding cultural nuances that standard filters miss.

- Seamless Integration: A sleek ReactJS frontend paired with a robust Node/Express backend for low-latency message processing.
  
## 🛠 Tech Stack

### Machine Learning

- Google MuRIL: Used for high-accuracy chat classification and multilingual representation.

- Indic BERT: Leveraged for the detoxification pipeline to handle regional linguistic nuances.

### Backend

- Node.js & Express:
  Handles the real-time API requests, user authentication, and serves as the bridge between the ML models and the frontend.

### Frontend

- ReactJS: A responsive, component-based UI for the chat interface and moderation dashboard.

# Mental Health Avatar Backend

Python backend implementation based on the research paper:
**"Interactive Empathy: Building a Facially Expressive Mental Health Companion"**

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mental Health Avatar System                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   User       │───▶│  Semantic    │───▶│  Response    │       │
│  │   Input      │    │  Analyzer    │    │  Generator   │       │
│  └──────────────┘    │  (Sec 3.2)   │    │  (Sec 3.1)   │       │
│                      └──────────────┘    └──────────────┘       │
│                             │                   │                │
│                             ▼                   ▼                │
│                      ┌──────────────┐    ┌──────────────┐       │
│                      │     VAE      │    │    TTS +     │       │
│                      │  Expression  │    │  Lip Sync    │       │
│                      │  (Sec 3.3)   │    │  (Sec 3.4)   │       │
│                      └──────────────┘    └──────────────┘       │
│                             │                   │                │
│                             └─────────┬─────────┘                │
│                                       ▼                          │
│                              ┌──────────────┐                    │
│                              │   Avatar     │                    │
│                              │  Animation   │                    │
│                              └──────────────┘                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Semantic Analyzer (`models/semantic_analyzer.py`)
Based on Section 3.2: "Semantic Analysis for Expression Generation"

- Analyzes text for emotional content
- Quantifies emotions: happy, angry, sad, fear, surprised, neutral
- Outputs emotion scores and expression parameters for avatar

```python
from models.semantic_analyzer import SemanticAnalyzer

analyzer = SemanticAnalyzer()
result = analyzer.analyze_for_avatar("I'm feeling happy today!")
# Returns: {'dominant_emotion': 'happy', 'confidence': 0.85, ...}
```

### 2. Response Generator (`models/response_generator.py`)
Based on Section 3.1: "User Input Processing and Response Generation"

- Uses GPT-3.5/Claude for empathetic response generation
- Includes crisis detection and appropriate responses
- Maintains conversation context

```python
from models.response_generator import ResponseGenerator

generator = ResponseGenerator(provider='openai')  # or 'anthropic', 'local'
response, metadata = generator.generate("I feel anxious", emotion='fear')
```

### 3. VAE Model (`models/vae.py`)
Based on Section 3.3: "Facial Expression Generation with VAE"

- Variational Autoencoder for facial expression generation
- Trained on CoMA dataset format
- Generates blend shapes from emotion vectors

```python
from models.vae import SimplifiedFacialVAE, create_emotion_vector

vae = SimplifiedFacialVAE()
emotion = create_emotion_vector(happy=0.8, surprised=0.2)
blend_shapes = vae.generate(emotion)
```

### 4. TTS & Lip Sync (`models/tts_lipsync.py`)
Based on Section 3.4: "Text-to-Speech Conversion and Lip-Sync"

- Text-to-Speech synthesis (Eleven Labs style)
- Rhubarb Lip Sync inspired viseme generation
- Animation keyframe generation

```python
from models.tts_lipsync import TTSLipSyncPipeline

pipeline = TTSLipSyncPipeline()
result = pipeline.process("Hello, how are you?", emotion='happy')
# Returns: audio_path, duration, lip_sync frames, animation keyframes
```

## Installation

```bash
# Clone the repository
cd avatar-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (optional)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Usage

### Start API Server
```bash
python main.py serve

# Or with uvicorn directly
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

### Interactive Demo
```bash
python main.py demo
```

### Train VAE Model
```bash
python main.py train
```

### Run Tests
```bash
python main.py test
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Analyze text for emotions |
| `/generate-response` | POST | Generate empathetic response |
| `/generate-expression` | POST | Generate facial expression |
| `/synthesize` | POST | TTS with lip sync |
| `/chat` | POST | Complete chat endpoint |
| `/chat/welcome` | GET | Get welcome message |
| `/expressions` | GET | List available expressions |

### Example: Complete Chat Flow

```python
import requests

# 1. Start chat
response = requests.get("http://localhost:8000/chat/welcome")
print(response.json())

# 2. Send message
response = requests.post("http://localhost:8000/chat", json={
    "message": "I've been feeling anxious lately",
    "session_id": "user123"
})
result = response.json()
print(f"Response: {result['response']}")
print(f"User emotion: {result['user_emotion']}")
print(f"Avatar emotion: {result['avatar_emotion']}")
print(f"Expression params: {result['expression_params']}")
```

## Environment Variables

```bash
# LLM API Keys (optional - defaults to local mode)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Server config
PORT=8000
HOST=0.0.0.0
```

## Project Structure

```
avatar-backend/
├── models/
│   ├── __init__.py
│   ├── vae.py              # VAE for expression generation
│   ├── semantic_analyzer.py # Emotion detection
│   ├── response_generator.py # LLM responses
│   └── tts_lipsync.py      # TTS & lip sync
├── api/
│   ├── __init__.py
│   └── server.py           # FastAPI server
├── main.py                 # Entry point
├── train_vae.py           # VAE training script
├── requirements.txt
└── README.md
```

## Research Paper Reference

This implementation is based on:
> "Interactive Empathy: Building a Facially Expressive Mental Health Companion"

Key contributions implemented:
- VAE-based facial expression generation from emotional metrics
- Semantic analysis pipeline for emotion quantification
- Integration with LLMs for empathetic response generation
- Rhubarb-style lip synchronization for avatar animation

## Emotion Categories

| Emotion | Expression Parameters |
|---------|----------------------|
| Happy | Smile, raised eyebrows, squinted eyes |
| Sad | Frown, lowered eyebrows, droopy eyes |
| Angry | Furrowed brows, tense mouth |
| Fear | Wide eyes, raised eyebrows |
| Surprised | Very wide eyes, open mouth |
| Neutral | Relaxed face |
| Empathy | Gentle smile, soft eyes |

## License

MIT License - See LICENSE file for details.

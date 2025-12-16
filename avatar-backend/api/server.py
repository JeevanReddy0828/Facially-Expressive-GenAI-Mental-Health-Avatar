"""
FastAPI Server for Mental Health Avatar Backend
Provides REST API endpoints for the complete avatar system.
"""

import os
import sys
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from models.semantic_analyzer import SemanticAnalyzer
from models.response_generator import ResponseGenerator, ChatSession
from models.tts_lipsync import TTSLipSyncPipeline
from models.vae import SimplifiedFacialVAE, EXPRESSION_TEMPLATES

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================
# Pydantic Models
# ============================================

class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    dominant_emotion: str
    confidence: float
    emotion_scores: Dict[str, float]
    expression_params: Dict[str, float]
    emotion_tensor: List[float]

class GenerateResponseRequest(BaseModel):
    user_message: str
    emotion: Optional[str] = None
    include_history: bool = True

class GenerateResponseResponse(BaseModel):
    response: str
    avatar_emotion: str
    is_crisis: bool
    provider: str

class GenerateExpressionRequest(BaseModel):
    emotion: str
    intensity: float = 1.0

class GenerateExpressionResponse(BaseModel):
    emotion: str
    blend_shapes: Dict[str, float]
    emotion_tensor: List[float]

class SynthesizeRequest(BaseModel):
    text: str
    emotion: Optional[str] = None

class SynthesizeResponse(BaseModel):
    audio_path: str
    duration: float
    lip_sync_frames: int
    animation_keyframes: int

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    user_emotion: str
    avatar_emotion: str
    expression_params: Dict[str, float]
    emotion_tensor: List[float]
    is_crisis: bool
    message_number: int


# ============================================
# App Setup
# ============================================

app = FastAPI(
    title="Mental Health Avatar API",
    description="Backend for Interactive Empathy Mental Health Companion",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Global State
# ============================================

class AppState:
    def __init__(self):
        self.semantic_analyzer = None
        self.response_generator = None
        self.tts_pipeline = None
        self.vae_model = None
        self.sessions: Dict[str, ChatSession] = {}
    
    def initialize(self):
        print("Initializing components...")
        
        self.semantic_analyzer = SemanticAnalyzer(use_transformer=False)
        print("✓ Semantic Analyzer")
        
        provider = 'local'
        if os.getenv('OPENAI_API_KEY'):
            provider = 'openai'
        elif os.getenv('ANTHROPIC_API_KEY'):
            provider = 'anthropic'
        self.response_generator = ResponseGenerator(provider=provider)
        print(f"✓ Response Generator ({provider})")
        
        self.tts_pipeline = TTSLipSyncPipeline(output_dir='./outputs/audio')
        print("✓ TTS Pipeline")
        
        if TORCH_AVAILABLE:
            self.vae_model = SimplifiedFacialVAE()
            self.vae_model.eval()
            print("✓ VAE Model")
        
        print("Ready!")

state = AppState()


@app.on_event("startup")
async def startup():
    state.initialize()


# ============================================
# Endpoints
# ============================================

@app.get("/")
async def root():
    return {"status": "healthy", "service": "Mental Health Avatar API", "version": "1.0.0"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Analyze text for emotions."""
    result = state.semantic_analyzer.analyze_for_avatar(request.text)
    return AnalyzeResponse(
        dominant_emotion=result['dominant_emotion'],
        confidence=result['confidence'],
        emotion_scores=result['emotion_scores'],
        expression_params=result['expression_params'],
        emotion_tensor=result['emotion_tensor']
    )


@app.post("/generate-response", response_model=GenerateResponseResponse)
async def generate_response(request: GenerateResponseRequest):
    """Generate empathetic response."""
    response, meta = state.response_generator.generate(
        request.user_message, request.emotion, request.include_history
    )
    return GenerateResponseResponse(
        response=response,
        avatar_emotion=meta['emotion'],
        is_crisis=meta['is_crisis'],
        provider=meta['provider']
    )


@app.post("/generate-expression", response_model=GenerateExpressionResponse)
async def generate_expression(request: GenerateExpressionRequest):
    """Generate facial expression from emotion."""
    emotion_map = {'happy': 0, 'angry': 1, 'sad': 2, 'fear': 3, 'surprised': 4, 'neutral': 5}
    
    if request.emotion not in emotion_map:
        raise HTTPException(400, f"Unknown emotion: {request.emotion}")
    
    emotion_tensor = [0.0] * 6
    emotion_tensor[emotion_map[request.emotion]] = request.intensity
    
    if state.vae_model and TORCH_AVAILABLE:
        emotion_t = torch.tensor([emotion_tensor], dtype=torch.float32)
        with torch.no_grad():
            expr = state.vae_model.generate(emotion_t)
        blend_shapes = {
            name: float(expr[0, i]) * request.intensity
            for i, name in enumerate(SimplifiedFacialVAE.BLEND_SHAPES[:expr.shape[1]])
        }
    else:
        template = EXPRESSION_TEMPLATES.get(request.emotion, {})
        blend_shapes = {k: v * request.intensity for k, v in template.items()}
    
    return GenerateExpressionResponse(
        emotion=request.emotion,
        blend_shapes=blend_shapes,
        emotion_tensor=emotion_tensor
    )


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest):
    """Synthesize speech with lip sync."""
    result = state.tts_pipeline.process(request.text, request.emotion)
    return SynthesizeResponse(
        audio_path=result['audio_path'],
        duration=result['duration'],
        lip_sync_frames=len(result['lip_sync']['frames']),
        animation_keyframes=len(result['animation'])
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Complete chat endpoint."""
    session_id = request.session_id or 'default'
    
    if session_id not in state.sessions:
        provider = 'local' if not os.getenv('OPENAI_API_KEY') else 'openai'
        state.sessions[session_id] = ChatSession(provider=provider)
    
    result = state.sessions[session_id].process(request.message)
    
    return ChatResponse(
        response=result['response'],
        user_emotion=result['user_emotion'],
        avatar_emotion=result['avatar_emotion'],
        expression_params=result['expression_params'],
        emotion_tensor=result['emotion_tensor'],
        is_crisis=result['is_crisis'],
        message_number=result['message_number']
    )


@app.get("/chat/welcome")
async def welcome():
    """Get welcome message."""
    session = ChatSession()
    return session.welcome()


@app.delete("/chat/{session_id}")
async def reset_session(session_id: str):
    """Reset chat session."""
    if session_id in state.sessions:
        del state.sessions[session_id]
    return {"message": "Session reset"}


@app.get("/expressions")
async def list_expressions():
    """List available expressions."""
    return {"emotions": list(EXPRESSION_TEMPLATES.keys()), "templates": EXPRESSION_TEMPLATES}


# ============================================
# RAG Endpoints
# ============================================

@app.get("/rag/knowledge-base")
async def get_knowledge_base():
    """Get all knowledge base categories and document counts."""
    from models.rag import MentalHealthKnowledgeBase
    
    kb = MentalHealthKnowledgeBase.KNOWLEDGE_BASE
    return {
        "categories": list(kb.keys()),
        "document_counts": {cat: len(docs) for cat, docs in kb.items()},
        "total_documents": sum(len(docs) for docs in kb.values())
    }


@app.get("/rag/retrieve")
async def retrieve_documents(query: str, emotion: Optional[str] = None, k: int = 3):
    """Retrieve relevant documents from knowledge base."""
    from models.rag import RAGRetriever
    
    retriever = RAGRetriever(use_transformers=False)
    retriever.initialize()
    
    results = retriever.retrieve(query, k=k, emotion_filter=emotion)
    
    return {
        "query": query,
        "emotion_filter": emotion,
        "results": [
            {
                "id": doc.id,
                "content": doc.content,
                "category": doc.metadata.get("category"),
                "relevance_score": float(score)
            }
            for doc, score in results
        ]
    }


@app.get("/rag/resources/{emotion}")
async def get_resources_for_emotion(emotion: str):
    """Get relevant therapeutic resources for an emotion."""
    from models.rag import MentalHealthKnowledgeBase
    
    docs = MentalHealthKnowledgeBase.get_by_emotion(emotion)
    
    return {
        "emotion": emotion,
        "resources": [
            {
                "id": doc.id,
                "content": doc.content,
                "category": doc.metadata.get("category")
            }
            for doc in docs[:5]
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('PORT', 8000)))

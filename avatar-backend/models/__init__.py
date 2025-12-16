"""
Mental Health Avatar - Models Package
"""

try:
    from .vae import FacialExpressionVAE, SimplifiedFacialVAE, VAELoss
    VAE_AVAILABLE = True
except ImportError:
    VAE_AVAILABLE = False
    FacialExpressionVAE = None
    SimplifiedFacialVAE = None
    VAELoss = None

from .semantic_analyzer import SemanticAnalyzer, EmotionScores
from .response_generator import ResponseGenerator, ChatSession
from .tts_lipsync import TTSLipSyncPipeline, LipSyncGenerator
from .rag import RAGRetriever, MentalHealthKnowledgeBase

__all__ = [
    'FacialExpressionVAE',
    'SimplifiedFacialVAE', 
    'VAELoss',
    'SemanticAnalyzer',
    'EmotionScores',
    'ResponseGenerator',
    'ChatSession',
    'TTSLipSyncPipeline',
    'LipSyncGenerator',
    'RAGRetriever',
    'MentalHealthKnowledgeBase',
    'VAE_AVAILABLE'
]

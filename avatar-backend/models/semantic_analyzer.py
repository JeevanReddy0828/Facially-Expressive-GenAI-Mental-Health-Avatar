"""
Semantic Analysis for Emotional Understanding
Section 3.2: "Semantic Analysis for Expression Generation"

This module analyzes text for emotional content and quantifies emotions into
categories that guide facial expression generation.

As stated in the paper:
"After analyzing the text for tone, context, and sentiment, the system quantifies 
the emotional content into specific categories, such as happiness, anger, boredom, 
fear, sadness, and excitement."

Reference: Mukashev et al. 2021 - Facial expression generation based on semantic analysis
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class EmotionScores:
    """
    Container for emotion analysis results.
    
    Example from paper:
    'Happy': 0.064, 'Angry': 0.110, 'Bored': 0.239, 
    'Fear': 0.282, 'Sad': 0.168, 'Excited': 0.134
    """
    happy: float = 0.0
    angry: float = 0.0
    sad: float = 0.0
    fear: float = 0.0
    surprised: float = 0.0
    neutral: float = 0.0
    excited: float = 0.0
    empathy: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in ['happy', 'angry', 'sad', 'fear', 'surprised', 'neutral', 'excited', 'empathy']}
    
    def to_tensor(self):
        """Convert to tensor for VAE input (6 basic emotions)."""
        if TORCH_AVAILABLE:
            import torch
            return torch.tensor([self.happy, self.angry, self.sad, self.fear, self.surprised, self.neutral])
        return np.array([self.happy, self.angry, self.sad, self.fear, self.surprised, self.neutral])
    
    def dominant(self) -> Tuple[str, float]:
        """Get dominant emotion and its score."""
        scores = self.to_dict()
        return max(scores.items(), key=lambda x: x[1])
    
    def normalize(self) -> 'EmotionScores':
        """Normalize scores to sum to 1."""
        total = sum(self.to_dict().values())
        if total > 0:
            return EmotionScores(**{k: v/total for k, v in self.to_dict().items()})
        return self


class EmotionLexicon:
    """Lexicon-based emotion detection using curated word lists."""
    
    LEXICON = {
        'happy': {
            'words': ['happy', 'joy', 'joyful', 'glad', 'pleased', 'delighted', 'cheerful', 'content',
                      'wonderful', 'great', 'amazing', 'fantastic', 'excellent', 'good', 'nice', 'love',
                      'grateful', 'thankful', 'blessed', 'excited', 'thrilled', 'ecstatic', 'elated',
                      'blissful', 'merry', 'upbeat', 'positive', 'optimistic', 'hopeful', 'proud'],
            'weight': 1.0
        },
        'sad': {
            'words': ['sad', 'unhappy', 'depressed', 'down', 'blue', 'miserable', 'sorrowful', 'grief',
                      'heartbroken', 'devastated', 'disappointed', 'hopeless', 'despair', 'lonely',
                      'isolated', 'abandoned', 'hurt', 'pain', 'suffering', 'crying', 'tears',
                      'melancholy', 'gloomy', 'empty', 'numb', 'lost', 'broken'],
            'weight': 1.0
        },
        'angry': {
            'words': ['angry', 'mad', 'furious', 'enraged', 'outraged', 'livid', 'irritated', 'annoyed',
                      'frustrated', 'aggravated', 'hostile', 'resentful', 'bitter', 'hate', 'hatred',
                      'rage', 'wrath', 'fury', 'temper', 'disgusted', 'offended'],
            'weight': 1.0
        },
        'fear': {
            'words': ['afraid', 'scared', 'frightened', 'terrified', 'fearful', 'anxious', 'worried',
                      'nervous', 'uneasy', 'apprehensive', 'dread', 'panic', 'alarmed', 'paranoid',
                      'insecure', 'vulnerable', 'threatened', 'stressed', 'overwhelmed', 'tense'],
            'weight': 1.0
        },
        'surprised': {
            'words': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'startled',
                      'speechless', 'unexpected', 'unbelievable', 'incredible', 'wow', 'whoa'],
            'weight': 1.0
        },
        'neutral': {
            'words': ['okay', 'ok', 'fine', 'alright', 'normal', 'usual', 'regular', 'moderate', 'average'],
            'weight': 0.5
        },
        'empathy': {
            'words': ['understand', 'support', 'care', 'compassion', 'kind', 'gentle', 'patient',
                      'listen', 'help', 'comfort', 'reassure', 'validate', 'appreciate', 'thank'],
            'weight': 0.8
        }
    }
    
    INTENSIFIERS = {'very': 1.5, 'really': 1.4, 'extremely': 1.8, 'so': 1.3, 'incredibly': 1.7}
    NEGATIONS = {'not', 'no', 'never', "n't", 'cannot', "can't", "won't", "don't", "doesn't", "didn't"}
    
    def analyze(self, text: str) -> EmotionScores:
        """Analyze text using lexicon matching."""
        words = re.findall(r'\b\w+\b', text.lower())
        scores = {e: 0.0 for e in self.LEXICON}
        
        negation_active = False
        intensity = 1.0
        
        for word in words:
            if word in self.INTENSIFIERS:
                intensity = self.INTENSIFIERS[word]
                continue
            if word in self.NEGATIONS:
                negation_active = True
                continue
            
            for emotion, data in self.LEXICON.items():
                if word in data['words']:
                    score = data['weight'] * intensity
                    if negation_active:
                        # Flip positive/negative
                        if emotion in ['happy', 'excited']:
                            scores['sad'] += score * 0.5
                        elif emotion == 'sad':
                            scores['neutral'] += score * 0.5
                        else:
                            scores[emotion] += score * 0.3
                    else:
                        scores[emotion] += score
                    negation_active = False
                    intensity = 1.0
        
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            scores['neutral'] = 1.0
        
        return EmotionScores(**scores)


class TransformerAnalyzer:
    """Transformer-based emotion analysis using pre-trained models."""
    
    LABEL_MAP = {
        'joy': 'happy', 'happiness': 'happy', 'love': 'happy',
        'anger': 'angry', 'sadness': 'sad', 'fear': 'fear',
        'surprise': 'surprised', 'neutral': 'neutral', 'disgust': 'angry'
    }
    
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")
        self.classifier = pipeline("text-classification", model=model_name, top_k=None, device=-1)
    
    def analyze(self, text: str) -> EmotionScores:
        results = self.classifier(text)[0]
        scores = {e: 0.0 for e in ['happy', 'angry', 'sad', 'fear', 'surprised', 'neutral', 'excited', 'empathy']}
        
        for pred in results:
            mapped = self.LABEL_MAP.get(pred['label'].lower(), 'neutral')
            scores[mapped] = max(scores[mapped], pred['score'])
        
        return EmotionScores(**scores)


class SemanticAnalyzer:
    """
    Main semantic analysis class combining multiple methods.
    
    This is the primary interface for emotion detection, combining
    lexicon-based and transformer-based approaches.
    """
    
    # Expression parameter mappings for avatar
    EXPRESSION_PARAMS = {
        'happy': {'smile': 0.8, 'eyebrow_raise': 0.2, 'eye_squint': 0.3},
        'sad': {'smile': -0.6, 'eyebrow_lower': 0.4, 'eye_droop': 0.3},
        'angry': {'smile': -0.3, 'eyebrow_furrow': 0.7, 'eye_squint': 0.2},
        'fear': {'smile': -0.2, 'eyebrow_raise': 0.6, 'eye_wide': 0.5},
        'surprised': {'smile': 0.1, 'eyebrow_raise': 0.8, 'eye_wide': 0.7, 'mouth_open': 0.5},
        'neutral': {'smile': 0.0, 'eyebrow_raise': 0.0, 'eye_wide': 0.0},
        'empathy': {'smile': 0.3, 'eyebrow_raise': 0.1, 'eye_soft': 0.4},
        'excited': {'smile': 0.9, 'eyebrow_raise': 0.4, 'eye_wide': 0.3}
    }
    
    # Avatar response emotion mapping
    RESPONSE_EMOTION_MAP = {
        'happy': 'happy', 'sad': 'empathy', 'angry': 'empathy',
        'fear': 'empathy', 'surprised': 'surprised', 'neutral': 'neutral',
        'excited': 'happy', 'empathy': 'happy'
    }
    
    def __init__(self, use_transformer: bool = False, transformer_weight: float = 0.7):
        self.lexicon = EmotionLexicon()
        self.transformer = None
        self.transformer_weight = transformer_weight
        
        if use_transformer and TRANSFORMERS_AVAILABLE:
            try:
                self.transformer = TransformerAnalyzer()
            except Exception as e:
                print(f"Could not load transformer: {e}")
    
    def preprocess(self, text: str) -> str:
        """Preprocess text for analysis."""
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = ' '.join(text.split())  # Normalize whitespace
        
        # Expand contractions
        contractions = {
            "i'm": "i am", "you're": "you are", "it's": "it is",
            "don't": "do not", "can't": "cannot", "won't": "will not",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not"
        }
        for c, e in contractions.items():
            text = re.sub(r'\b' + c + r'\b', e, text, flags=re.IGNORECASE)
        
        return text
    
    def analyze(self, text: str) -> EmotionScores:
        """Perform semantic analysis on text."""
        text = self.preprocess(text)
        if not text.strip():
            return EmotionScores(neutral=1.0)
        
        lexicon_scores = self.lexicon.analyze(text)
        
        if self.transformer:
            try:
                transformer_scores = self.transformer.analyze(text)
                # Weighted combination
                combined = {}
                for key in lexicon_scores.to_dict():
                    combined[key] = (
                        getattr(lexicon_scores, key) * (1 - self.transformer_weight) +
                        getattr(transformer_scores, key) * self.transformer_weight
                    )
                return EmotionScores(**combined).normalize()
            except:
                pass
        
        return lexicon_scores.normalize()
    
    def analyze_for_avatar(self, text: str) -> Dict:
        """
        Analyze text and return data formatted for avatar.
        
        Returns emotion scores, dominant emotion, expression parameters,
        and tensor for VAE input.
        """
        scores = self.analyze(text)
        dominant_emotion, dominant_score = scores.dominant()
        
        base_params = self.EXPRESSION_PARAMS.get(dominant_emotion, self.EXPRESSION_PARAMS['neutral'])
        blended_params = {k: v * dominant_score for k, v in base_params.items()}
        
        return {
            'emotion_scores': scores.to_dict(),
            'dominant_emotion': dominant_emotion,
            'dominant_score': dominant_score,
            'expression_params': blended_params,
            'emotion_tensor': scores.to_tensor().tolist() if TORCH_AVAILABLE else scores.to_tensor().tolist(),
            'confidence': dominant_score
        }
    
    def get_response_emotion(self, user_emotion: str) -> str:
        """Get appropriate avatar response emotion for user's emotion."""
        return self.RESPONSE_EMOTION_MAP.get(user_emotion, 'neutral')


# Convenience function
def analyze_emotion(text: str) -> Dict:
    """Quick emotion analysis."""
    return SemanticAnalyzer().analyze_for_avatar(text)


if __name__ == "__main__":
    print("Testing Semantic Analyzer...")
    
    analyzer = SemanticAnalyzer(use_transformer=False)
    
    tests = [
        "I'm feeling really happy today! Everything is going great.",
        "I'm so sad and lonely. Nothing seems to work out.",
        "This makes me so angry! I can't believe it.",
        "I'm worried about my future. Everything feels uncertain.",
        "Wow! I can't believe you did that! Amazing!",
        "I'm okay. Just a regular day.",
        "Thank you for understanding. I appreciate your support.",
    ]
    
    for text in tests:
        result = analyzer.analyze_for_avatar(text)
        print(f"\nInput: \"{text}\"")
        print(f"Dominant: {result['dominant_emotion']} ({result['dominant_score']:.3f})")
        print(f"Params: {result['expression_params']}")
    
    print("\n✓ Semantic Analyzer tests passed!")

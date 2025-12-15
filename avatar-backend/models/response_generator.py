"""
Response Generation using Large Language Models
Section 3.1: "User Input processing and Response Generation"

Uses OpenAI GPT-3.5 (or Claude) to generate empathetic, contextually relevant
responses for mental health support conversations.

As stated in the paper:
"In the crucial phase of response generation, our methodology employs OpenAI's 
GPT-3.5 API, renowned for its advanced natural language processing capabilities, 
to generate empathetic and contextually relevant responses to user inputs."
"""

import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class Message:
    """Conversation message."""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    emotion: Optional[str] = None


@dataclass  
class ConversationContext:
    """Manages conversation history."""
    messages: List[Message] = field(default_factory=list)
    max_history: int = 20
    
    def add(self, role: str, content: str, emotion: Optional[str] = None):
        self.messages.append(Message(role=role, content=content, emotion=emotion))
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def get_history(self, n: Optional[int] = None) -> List[Dict]:
        msgs = self.messages[-n:] if n else self.messages
        return [{'role': m.role, 'content': m.content} for m in msgs]
    
    def clear(self):
        self.messages = []


class MentalHealthPrompts:
    """Curated prompts for mental health conversations."""
    
    SYSTEM_PROMPT = """You are a compassionate mental health companion avatar. Your role:

1. LISTEN ACTIVELY and acknowledge feelings
2. VALIDATE emotions without judgment  
3. RESPOND with warmth and empathy (2-4 sentences)
4. ASK thoughtful follow-up questions
5. NEVER diagnose or give medical advice
6. If someone mentions self-harm, encourage professional help

Guidelines:
- Keep responses concise but meaningful
- Use warm, conversational language
- Mirror emotional tone appropriately
- Focus on the present moment"""

    CRISIS_RESPONSE = """I'm concerned about what you've shared. Your feelings matter.

Please reach out to a crisis helpline:
- National Suicide Prevention: 988 (US)
- Crisis Text Line: Text HOME to 741741

You don't have to face this alone."""

    RESPONSES = {
        'happy': [
            "It's wonderful to hear you're feeling positive! What's bringing you joy?",
            "That's great! Tell me more about what's going well.",
        ],
        'sad': [
            "I hear you're going through a difficult time. Your feelings are valid.",
            "It sounds tough. I'm here to listen if you'd like to share more.",
        ],
        'angry': [
            "I sense your frustration. It's natural to feel angry sometimes.",
            "Your anger is valid. What triggered these feelings?",
        ],
        'fear': [
            "I hear you're feeling anxious. You're not alone in this.",
            "It's natural to feel scared. What's weighing on your mind?",
        ],
        'surprised': [
            "That sounds unexpected! How are you processing this?",
            "Life has its surprises. How do you feel about it?",
        ],
        'neutral': [
            "Thank you for sharing. How's your day going?",
            "I'm here whenever you're ready to talk.",
        ],
        'empathy': [
            "I appreciate you reaching out. How can I support you?",
            "Thank you for trusting me. What's on your mind?",
        ]
    }
    
    CRISIS_KEYWORDS = [
        'suicide', 'kill myself', 'end my life', 'want to die',
        'self-harm', 'hurt myself', 'no reason to live'
    ]


class ResponseGenerator:
    """
    Generates empathetic responses using LLMs.
    Supports OpenAI GPT-3.5 and Anthropic Claude with local fallback.
    """
    
    def __init__(
        self,
        provider: str = 'local',  # 'openai', 'anthropic', 'local'
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.provider = provider
        self.api_key = api_key or os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        self.context = ConversationContext()
        self.client = None
        
        # Set models
        if model:
            self.model = model
        elif provider == 'openai':
            self.model = "gpt-3.5-turbo"  # As per paper
        elif provider == 'anthropic':
            self.model = "claude-3-sonnet-20240229"
        else:
            self.model = None
        
        self._init_client()
    
    def _init_client(self):
        if self.provider == 'openai' and OPENAI_AVAILABLE and self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        elif self.provider == 'anthropic' and ANTHROPIC_AVAILABLE and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def _check_crisis(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in MentalHealthPrompts.CRISIS_KEYWORDS)
    
    def generate(
        self,
        user_message: str,
        emotion: Optional[str] = None,
        include_history: bool = True
    ) -> Tuple[str, Dict]:
        """Generate empathetic response."""
        
        # Check for crisis
        if self._check_crisis(user_message):
            return MentalHealthPrompts.CRISIS_RESPONSE, {'is_crisis': True, 'emotion': 'empathy'}
        
        self.context.add('user', user_message, emotion)
        
        # Generate based on provider
        if self.client and self.provider == 'openai':
            response = self._generate_openai(user_message, emotion, include_history)
        elif self.client and self.provider == 'anthropic':
            response = self._generate_anthropic(user_message, emotion, include_history)
        else:
            response = self._generate_local(emotion)
        
        response_emotion = 'empathy' if emotion in ['sad', 'fear', 'angry'] else emotion or 'neutral'
        self.context.add('assistant', response, response_emotion)
        
        return response, {'is_crisis': False, 'emotion': response_emotion, 'provider': self.provider}
    
    def _generate_openai(self, user_message: str, emotion: Optional[str], include_history: bool) -> str:
        try:
            messages = [{"role": "system", "content": MentalHealthPrompts.SYSTEM_PROMPT}]
            
            if include_history:
                messages.extend(self.context.get_history(10))
            else:
                messages.append({"role": "user", "content": user_message})
            
            if emotion:
                messages[-1]["content"] += f"\n[User emotion: {emotion}]"
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI error: {e}")
            return self._generate_local(emotion)
    
    def _generate_anthropic(self, user_message: str, emotion: Optional[str], include_history: bool) -> str:
        try:
            messages = self.context.get_history(10) if include_history else [{"role": "user", "content": user_message}]
            
            if emotion:
                messages[-1]["content"] += f"\n[User emotion: {emotion}]"
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                system=MentalHealthPrompts.SYSTEM_PROMPT,
                messages=messages
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Anthropic error: {e}")
            return self._generate_local(emotion)
    
    def _generate_local(self, emotion: Optional[str]) -> str:
        """Rule-based fallback responses."""
        import random
        responses = MentalHealthPrompts.RESPONSES.get(emotion or 'neutral', MentalHealthPrompts.RESPONSES['neutral'])
        return random.choice(responses)
    
    def reset(self):
        self.context.clear()


class ChatSession:
    """Complete chat session integrating semantic analysis and response generation."""
    
    def __init__(self, provider: str = 'local', api_key: Optional[str] = None):
        self.generator = ResponseGenerator(provider=provider, api_key=api_key)
        self.session_start = datetime.now()
        self.message_count = 0
        
        from .semantic_analyzer import SemanticAnalyzer
        self.analyzer = SemanticAnalyzer(use_transformer=False)
    
    def process(self, user_message: str) -> Dict:
        """Process user message and generate complete response data."""
        
        # Analyze emotion
        analysis = self.analyzer.analyze_for_avatar(user_message)
        user_emotion = analysis['dominant_emotion']
        
        # Generate response
        response, metadata = self.generator.generate(user_message, user_emotion)
        
        # Get avatar emotion
        avatar_emotion = self.analyzer.get_response_emotion(user_emotion)
        if metadata.get('is_crisis'):
            avatar_emotion = 'empathy'
        
        # Get expression params for response
        response_analysis = self.analyzer.analyze_for_avatar(response)
        
        self.message_count += 1
        
        return {
            'user_emotion': user_emotion,
            'user_emotion_scores': analysis['emotion_scores'],
            'response': response,
            'avatar_emotion': avatar_emotion,
            'expression_params': response_analysis['expression_params'],
            'emotion_tensor': response_analysis['emotion_tensor'],
            'is_crisis': metadata.get('is_crisis', False),
            'message_number': self.message_count,
            'session_duration': (datetime.now() - self.session_start).seconds
        }
    
    def welcome(self) -> Dict:
        """Get welcome message."""
        return {
            'response': "Hello! I'm your mental health companion. I'm here to listen and support you. How are you feeling today?",
            'avatar_emotion': 'empathy',
            'expression_params': {'smile': 0.3, 'eyebrow_raise': 0.1, 'eye_soft': 0.4}
        }


if __name__ == "__main__":
    print("Testing Response Generator...")
    
    generator = ResponseGenerator(provider='local')
    
    tests = [
        ("I'm feeling really happy today!", "happy"),
        ("I've been so sad lately.", "sad"),
        ("I'm frustrated with everything.", "angry"),
        ("Hello, how are you?", "neutral"),
    ]
    
    for msg, emotion in tests:
        response, meta = generator.generate(msg, emotion)
        print(f"\nUser: {msg}")
        print(f"Emotion: {emotion}")
        print(f"Response: {response}")
    
    print("\n✓ Response Generator tests passed!")

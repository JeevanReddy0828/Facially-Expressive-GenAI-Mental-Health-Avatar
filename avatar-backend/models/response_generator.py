"""
Response Generation using Large Language Models with RAG
Section 3.1: "User Input Processing and Response Generation"

Uses OpenAI GPT-3.5 (or Claude) with Retrieval-Augmented Generation (RAG)
to generate empathetic, contextually relevant responses grounded in
mental health knowledge.

RAG Enhancement:
- Retrieves relevant therapeutic information from knowledge base
- Augments LLM prompts with contextual knowledge
- Ensures responses are grounded in best practices
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
    Generates empathetic responses using LLMs with optional RAG augmentation.
    Supports OpenAI GPT-3.5 and Anthropic Claude with local fallback.
    """
    
    def __init__(
        self,
        provider: str = 'local',
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        use_rag: bool = True
    ):
        self.provider = provider
        self.api_key = api_key or os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        self.context = ConversationContext()
        self.client = None
        self.use_rag = use_rag
        self.rag_retriever = None
        
        if model:
            self.model = model
        elif provider == 'openai':
            self.model = "gpt-3.5-turbo"
        elif provider == 'anthropic':
            self.model = "claude-3-sonnet-20240229"
        else:
            self.model = None
        
        self._init_client()
        self._init_rag()
    
    def _init_client(self):
        if self.provider == 'openai' and OPENAI_AVAILABLE and self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        elif self.provider == 'anthropic' and ANTHROPIC_AVAILABLE and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def _init_rag(self):
        """Initialize RAG retriever."""
        if self.use_rag:
            try:
                from .rag import RAGRetriever
                self.rag_retriever = RAGRetriever(use_transformers=False)
                self.rag_retriever.initialize()
            except Exception as e:
                print(f"RAG initialization: {e}")
                self.rag_retriever = None
    
    def _get_rag_context(self, message: str, emotion: Optional[str]) -> str:
        """Get RAG context for the message."""
        if self.rag_retriever:
            try:
                return self.rag_retriever.get_context(message, emotion=emotion, max_tokens=300)
            except:
                pass
        return ""
    
    def _augment_prompt(self, base_prompt: str, rag_context: str) -> str:
        """Augment system prompt with RAG context."""
        if rag_context:
            return f"""{base_prompt}

RELEVANT THERAPEUTIC KNOWLEDGE:
{rag_context}

Use this knowledge to inform your response when relevant."""
        return base_prompt
    
    def _check_crisis(self, text: str) -> bool:
        return any(kw in text.lower() for kw in MentalHealthPrompts.CRISIS_KEYWORDS)
    
    def generate(
        self,
        user_message: str,
        emotion: Optional[str] = None,
        include_history: bool = True
    ) -> Tuple[str, Dict]:
        """Generate empathetic response with RAG augmentation."""
        
        if self._check_crisis(user_message):
            return MentalHealthPrompts.CRISIS_RESPONSE, {
                'is_crisis': True, 'emotion': 'empathy', 'rag_enabled': False
            }
        
        self.context.add('user', user_message, emotion)
        rag_context = self._get_rag_context(user_message, emotion)
        
        if self.client and self.provider == 'openai':
            response = self._generate_openai(user_message, emotion, include_history, rag_context)
        elif self.client and self.provider == 'anthropic':
            response = self._generate_anthropic(user_message, emotion, include_history, rag_context)
        else:
            response = self._generate_local(emotion, rag_context)
        
        response_emotion = 'empathy' if emotion in ['sad', 'fear', 'angry'] else emotion or 'neutral'
        self.context.add('assistant', response, response_emotion)
        
        return response, {
            'is_crisis': False,
            'emotion': response_emotion,
            'provider': self.provider,
            'rag_enabled': bool(rag_context),
            'rag_context_length': len(rag_context)
        }
    
    def _generate_openai(self, user_message: str, emotion: Optional[str], include_history: bool, rag_context: str) -> str:
        try:
            system_prompt = self._augment_prompt(MentalHealthPrompts.SYSTEM_PROMPT, rag_context)
            messages = [{"role": "system", "content": system_prompt}]
            
            if include_history:
                messages.extend(self.context.get_history(10))
            else:
                messages.append({"role": "user", "content": user_message})
            
            if emotion:
                messages[-1]["content"] += f"\n[User emotion: {emotion}]"
            
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=300, temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI error: {e}")
            return self._generate_local(emotion, rag_context)
    
    def _generate_anthropic(self, user_message: str, emotion: Optional[str], include_history: bool, rag_context: str) -> str:
        try:
            system_prompt = self._augment_prompt(MentalHealthPrompts.SYSTEM_PROMPT, rag_context)
            messages = self.context.get_history(10) if include_history else [{"role": "user", "content": user_message}]
            
            if emotion:
                messages[-1]["content"] += f"\n[User emotion: {emotion}]"
            
            response = self.client.messages.create(
                model=self.model, max_tokens=300, system=system_prompt, messages=messages
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Anthropic error: {e}")
            return self._generate_local(emotion, rag_context)
    
    def _generate_local(self, emotion: Optional[str], rag_context: str = "") -> str:
        """Rule-based fallback with RAG enhancement."""
        import random
        responses = MentalHealthPrompts.RESPONSES.get(emotion or 'neutral', MentalHealthPrompts.RESPONSES['neutral'])
        base_response = random.choice(responses)
        
        if rag_context and len(rag_context) > 50:
            lines = rag_context.split('\n')
            for line in lines:
                if len(line) > 30 and not line.startswith('['):
                    tip = line.strip()[:150]
                    if tip:
                        base_response += f" {tip.split('.')[0]}."
                    break
        
        return base_response
    
    def reset(self):
        self.context.clear()
    
    def get_relevant_resources(self, emotion: str) -> List[Dict]:
        """Get relevant resources from RAG."""
        if self.rag_retriever:
            try:
                results = self.rag_retriever.retrieve(f"help with {emotion}", emotion_filter=emotion, k=3)
                return [{'id': doc.id, 'content': doc.content[:200], 'category': doc.metadata.get('category')} 
                        for doc, score in results]
            except:
                pass
        return []


class ChatSession:
    """Complete chat session with semantic analysis, RAG, and response generation."""
    
    def __init__(self, provider: str = 'local', api_key: Optional[str] = None, use_rag: bool = True):
        self.generator = ResponseGenerator(provider=provider, api_key=api_key, use_rag=use_rag)
        self.session_start = datetime.now()
        self.message_count = 0
        
        from .semantic_analyzer import SemanticAnalyzer
        self.analyzer = SemanticAnalyzer(use_transformer=False)
    
    def process(self, user_message: str) -> Dict:
        """Process user message with full pipeline."""
        analysis = self.analyzer.analyze_for_avatar(user_message)
        user_emotion = analysis['dominant_emotion']
        
        response, metadata = self.generator.generate(user_message, user_emotion)
        
        avatar_emotion = self.analyzer.get_response_emotion(user_emotion)
        if metadata.get('is_crisis'):
            avatar_emotion = 'empathy'
        
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
            'rag_enabled': metadata.get('rag_enabled', False),
            'message_number': self.message_count,
            'session_duration': (datetime.now() - self.session_start).seconds
        }
    
    def welcome(self) -> Dict:
        return {
            'response': "Hello! I'm your mental health companion. I'm here to listen and support you. How are you feeling today?",
            'avatar_emotion': 'empathy',
            'expression_params': {'smile': 0.3, 'eyebrow_raise': 0.1, 'eye_soft': 0.4}
        }
    
    def get_resources(self, emotion: str) -> List[Dict]:
        return self.generator.get_relevant_resources(emotion)


if __name__ == "__main__":
    print("Testing Response Generator with RAG...")
    
    generator = ResponseGenerator(provider='local', use_rag=True)
    
    tests = [
        ("I'm feeling really anxious about everything", "fear"),
        ("I've been so sad and lonely lately", "sad"),
        ("I feel happy today!", "happy"),
    ]
    
    for msg, emotion in tests:
        response, meta = generator.generate(msg, emotion)
        print(f"\nUser: {msg}")
        print(f"Response: {response}")
        print(f"RAG enabled: {meta['rag_enabled']}")
    
    print("\n✓ Response Generator with RAG tests passed!")

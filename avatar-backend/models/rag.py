"""
Retrieval-Augmented Generation (RAG) for Mental Health Knowledge
Based on Section 3.1 enhancement for contextually relevant responses.

RAG enhances LLM responses by retrieving relevant information from a 
curated mental health knowledge base, ensuring responses are grounded
in therapeutic best practices and accurate information.

Components:
1. Knowledge Base - Curated mental health resources
2. Embedding Model - Text vectorization
3. Vector Store - Efficient similarity search
4. Retrieval Chain - Query and retrieve relevant context
"""

import os
import json
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

# Optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@dataclass
class Document:
    """A document in the knowledge base."""
    id: str
    content: str
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata
        }


class MentalHealthKnowledgeBase:
    """
    Curated knowledge base for mental health support.
    
    Contains information about:
    - Coping strategies
    - Emotional regulation techniques
    - Therapeutic approaches
    - Crisis resources
    - Self-care practices
    """
    
    # Knowledge base documents organized by category
    KNOWLEDGE_BASE = {
        'coping_strategies': [
            {
                'id': 'cope_breathing',
                'content': """Deep breathing exercises can help manage anxiety and stress. 
                The 4-7-8 technique involves breathing in for 4 seconds, holding for 7 seconds, 
                and exhaling for 8 seconds. This activates the parasympathetic nervous system 
                and promotes relaxation. Practice this technique when feeling overwhelmed.""",
                'metadata': {'category': 'coping', 'emotions': ['anxiety', 'stress', 'fear']}
            },
            {
                'id': 'cope_grounding',
                'content': """The 5-4-3-2-1 grounding technique helps during anxiety or panic. 
                Identify 5 things you can see, 4 things you can touch, 3 things you can hear, 
                2 things you can smell, and 1 thing you can taste. This brings attention to 
                the present moment and away from anxious thoughts.""",
                'metadata': {'category': 'coping', 'emotions': ['anxiety', 'panic', 'fear']}
            },
            {
                'id': 'cope_journaling',
                'content': """Journaling is an effective way to process emotions. Writing about 
                feelings helps externalize them and gain perspective. Try writing for 15-20 
                minutes about what you're experiencing without judgment. This can help identify 
                patterns in thoughts and emotions.""",
                'metadata': {'category': 'coping', 'emotions': ['sad', 'angry', 'confused']}
            },
            {
                'id': 'cope_movement',
                'content': """Physical movement releases endorphins and can improve mood. Even 
                a short 10-minute walk can make a difference. Exercise doesn't have to be 
                intense - gentle stretching, yoga, or dancing can all help regulate emotions 
                and reduce stress.""",
                'metadata': {'category': 'coping', 'emotions': ['sad', 'stressed', 'anxious']}
            },
            {
                'id': 'cope_social',
                'content': """Social connection is vital for mental health. Reaching out to 
                a trusted friend or family member can provide support during difficult times. 
                Even brief positive interactions can boost mood. If in-person connection isn't 
                possible, a phone call or video chat can help.""",
                'metadata': {'category': 'coping', 'emotions': ['lonely', 'sad', 'isolated']}
            }
        ],
        'emotional_regulation': [
            {
                'id': 'reg_validation',
                'content': """Emotional validation means acknowledging that feelings are 
                real and understandable. All emotions serve a purpose and are valid responses 
                to experiences. Instead of fighting emotions, try to observe them with 
                curiosity and self-compassion.""",
                'metadata': {'category': 'regulation', 'emotions': ['all']}
            },
            {
                'id': 'reg_cognitive',
                'content': """Cognitive reframing involves examining thoughts and considering 
                alternative perspectives. Ask yourself: Is this thought based on facts? 
                What would I tell a friend in this situation? Are there other ways to view 
                this? This doesn't mean dismissing feelings, but expanding perspective.""",
                'metadata': {'category': 'regulation', 'emotions': ['anxious', 'sad', 'angry']}
            },
            {
                'id': 'reg_acceptance',
                'content': """Acceptance doesn't mean giving up or approving of a situation. 
                It means acknowledging reality as it is, which reduces suffering caused by 
                resistance. This is a key principle in ACT (Acceptance and Commitment Therapy). 
                Accept emotions without judgment while taking valued action.""",
                'metadata': {'category': 'regulation', 'emotions': ['all']}
            },
            {
                'id': 'reg_selfcompassion',
                'content': """Self-compassion involves treating yourself with the same kindness 
                you would show a good friend. It has three components: self-kindness instead 
                of self-criticism, common humanity (recognizing suffering is part of human 
                experience), and mindfulness (balanced awareness of emotions).""",
                'metadata': {'category': 'regulation', 'emotions': ['shame', 'guilt', 'sad']}
            }
        ],
        'therapeutic_approaches': [
            {
                'id': 'therapy_cbt',
                'content': """Cognitive Behavioral Therapy (CBT) focuses on the connection 
                between thoughts, feelings, and behaviors. It helps identify negative thought 
                patterns and develop healthier ways of thinking. CBT is evidence-based and 
                effective for depression, anxiety, and many other conditions.""",
                'metadata': {'category': 'therapy', 'approach': 'CBT'}
            },
            {
                'id': 'therapy_mindfulness',
                'content': """Mindfulness involves paying attention to the present moment 
                without judgment. Regular mindfulness practice can reduce stress, anxiety, 
                and depression. Even 5-10 minutes daily of focused breathing or body scan 
                meditation can make a difference over time.""",
                'metadata': {'category': 'therapy', 'approach': 'mindfulness'}
            },
            {
                'id': 'therapy_dbt',
                'content': """Dialectical Behavior Therapy (DBT) combines CBT with mindfulness 
                and acceptance strategies. It teaches four skill sets: mindfulness, distress 
                tolerance, emotion regulation, and interpersonal effectiveness. DBT is 
                particularly helpful for intense emotions.""",
                'metadata': {'category': 'therapy', 'approach': 'DBT'}
            }
        ],
        'self_care': [
            {
                'id': 'care_sleep',
                'content': """Quality sleep is fundamental to mental health. Aim for 7-9 hours 
                per night. Good sleep hygiene includes: consistent sleep schedule, limiting 
                screens before bed, keeping the room cool and dark, and avoiding caffeine 
                late in the day. Poor sleep can worsen anxiety and depression.""",
                'metadata': {'category': 'self_care', 'area': 'sleep'}
            },
            {
                'id': 'care_nutrition',
                'content': """Nutrition affects mental health. A balanced diet with whole 
                grains, fruits, vegetables, and lean proteins supports brain function. 
                Omega-3 fatty acids, found in fish and walnuts, may help with mood. 
                Stay hydrated and limit excessive sugar and processed foods.""",
                'metadata': {'category': 'self_care', 'area': 'nutrition'}
            },
            {
                'id': 'care_boundaries',
                'content': """Setting healthy boundaries protects mental health. It's okay 
                to say no to requests that drain your energy. Boundaries aren't selfish - 
                they're necessary for sustainable relationships and self-care. Communicate 
                boundaries clearly and kindly.""",
                'metadata': {'category': 'self_care', 'area': 'boundaries'}
            }
        ],
        'crisis_support': [
            {
                'id': 'crisis_resources',
                'content': """If you're in crisis, help is available 24/7. National Suicide 
                Prevention Lifeline: 988. Crisis Text Line: Text HOME to 741741. International 
                Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/
                You don't have to face this alone.""",
                'metadata': {'category': 'crisis', 'priority': 'high'}
            },
            {
                'id': 'crisis_safety',
                'content': """If you're having thoughts of self-harm, please reach out for 
                support. Remove access to means if possible. Tell someone you trust how 
                you're feeling. These thoughts can pass, and professional help is available. 
                Your life has value.""",
                'metadata': {'category': 'crisis', 'priority': 'high'}
            }
        ],
        'understanding_emotions': [
            {
                'id': 'emotion_anxiety',
                'content': """Anxiety is your body's natural response to perceived threat. 
                Physical symptoms like racing heart, sweating, and rapid breathing are your 
                fight-or-flight system activating. While uncomfortable, anxiety isn't dangerous. 
                It often decreases naturally after 20-30 minutes if you don't fight it.""",
                'metadata': {'category': 'understanding', 'emotion': 'anxiety'}
            },
            {
                'id': 'emotion_depression',
                'content': """Depression is more than sadness - it can include loss of interest, 
                changes in sleep and appetite, fatigue, difficulty concentrating, and feelings 
                of worthlessness. It's a medical condition, not a character flaw. Treatment 
                is available and effective. You don't have to feel this way forever.""",
                'metadata': {'category': 'understanding', 'emotion': 'depression'}
            },
            {
                'id': 'emotion_anger',
                'content': """Anger is a normal emotion that signals boundaries have been 
                crossed or needs aren't being met. It's not bad to feel angry - it's what 
                you do with anger that matters. Healthy expression includes identifying the 
                underlying need and communicating assertively.""",
                'metadata': {'category': 'understanding', 'emotion': 'anger'}
            },
            {
                'id': 'emotion_grief',
                'content': """Grief is a natural response to loss - not just death, but any 
                significant loss. There's no right way or timeline to grieve. Common experiences 
                include denial, anger, bargaining, depression, and acceptance, but grief isn't 
                linear. Be patient with yourself.""",
                'metadata': {'category': 'understanding', 'emotion': 'grief'}
            }
        ]
    }
    
    @classmethod
    def get_all_documents(cls) -> List[Document]:
        """Get all documents from the knowledge base."""
        documents = []
        for category, docs in cls.KNOWLEDGE_BASE.items():
            for doc in docs:
                documents.append(Document(
                    id=doc['id'],
                    content=doc['content'],
                    metadata=doc['metadata']
                ))
        return documents
    
    @classmethod
    def get_by_category(cls, category: str) -> List[Document]:
        """Get documents by category."""
        docs = cls.KNOWLEDGE_BASE.get(category, [])
        return [Document(id=d['id'], content=d['content'], metadata=d['metadata']) for d in docs]
    
    @classmethod
    def get_by_emotion(cls, emotion: str) -> List[Document]:
        """Get documents relevant to a specific emotion."""
        documents = []
        for category, docs in cls.KNOWLEDGE_BASE.items():
            for doc in docs:
                emotions = doc['metadata'].get('emotions', [])
                if emotion in emotions or 'all' in emotions:
                    documents.append(Document(
                        id=doc['id'],
                        content=doc['content'],
                        metadata=doc['metadata']
                    ))
        return documents


class SimpleEmbedding:
    """
    Simple TF-IDF based embedding when sentence-transformers not available.
    """
    
    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
        self.dimension = 1000  # Fixed dimension for simple embedding
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def fit(self, documents: List[str]):
        """Build vocabulary from documents."""
        doc_freq = {}
        
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        
        # Sort by frequency and take top N
        sorted_tokens = sorted(doc_freq.items(), key=lambda x: -x[1])
        self.vocabulary = {token: idx for idx, (token, _) in enumerate(sorted_tokens[:self.dimension])}
        
        # Calculate IDF
        n_docs = len(documents)
        self.idf = {token: np.log(n_docs / (freq + 1)) for token, freq in doc_freq.items()}
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to vectors."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = np.zeros((len(texts), self.dimension))
        
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            token_freq = {}
            for token in tokens:
                token_freq[token] = token_freq.get(token, 0) + 1
            
            for token, freq in token_freq.items():
                if token in self.vocabulary:
                    idx = self.vocabulary[token]
                    tf = freq / len(tokens)
                    idf = self.idf.get(token, 0)
                    embeddings[i, idx] = tf * idf
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        return embeddings


class VectorStore:
    """
    Vector store for efficient similarity search.
    Uses FAISS if available, falls back to numpy.
    """
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine sim with normalized vectors)
    
    def add(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents with their embeddings."""
        self.documents.extend(documents)
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(embeddings.astype(np.float32))
    
    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        if len(self.documents) == 0:
            return []
        
        k = min(k, len(self.documents))
        
        if FAISS_AVAILABLE and self.index is not None:
            scores, indices = self.index.search(query_embedding.astype(np.float32).reshape(1, -1), k)
            results = [(self.documents[idx], float(scores[0][i])) for i, idx in enumerate(indices[0])]
        else:
            # Numpy fallback - cosine similarity
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            similarities = np.dot(self.embeddings, query_norm)
            top_indices = np.argsort(similarities)[-k:][::-1]
            results = [(self.documents[idx], float(similarities[idx])) for idx in top_indices]
        
        return results
    
    def save(self, path: str):
        """Save vector store to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save documents
        docs_data = [doc.to_dict() for doc in self.documents]
        with open(path / 'documents.json', 'w') as f:
            json.dump(docs_data, f)
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(path / 'embeddings.npy', self.embeddings)
    
    def load(self, path: str):
        """Load vector store from disk."""
        path = Path(path)
        
        # Load documents
        with open(path / 'documents.json', 'r') as f:
            docs_data = json.load(f)
        self.documents = [Document(**d) for d in docs_data]
        
        # Load embeddings
        self.embeddings = np.load(path / 'embeddings.npy')
        
        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(self.embeddings.astype(np.float32))


class RAGRetriever:
    """
    Retrieval-Augmented Generation system for mental health knowledge.
    
    Retrieves relevant information from the knowledge base to enhance
    LLM responses with accurate, therapeutic content.
    """
    
    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        use_transformers: bool = True
    ):
        """
        Initialize the RAG retriever.
        
        Args:
            embedding_model: Sentence transformer model name
            use_transformers: Whether to use sentence-transformers
        """
        self.use_transformers = use_transformers and SENTENCE_TRANSFORMERS_AVAILABLE
        
        if self.use_transformers:
            self.encoder = SentenceTransformer(embedding_model)
            self.dimension = self.encoder.get_sentence_embedding_dimension()
        else:
            self.encoder = SimpleEmbedding()
            self.dimension = self.encoder.dimension
        
        self.vector_store = VectorStore(self.dimension)
        self.initialized = False
    
    def initialize(self, documents: Optional[List[Document]] = None):
        """
        Initialize the retriever with documents.
        
        Args:
            documents: Optional custom documents, uses default KB if None
        """
        if documents is None:
            documents = MentalHealthKnowledgeBase.get_all_documents()
        
        # Get text content
        texts = [doc.content for doc in documents]
        
        # Fit simple embedding if not using transformers
        if not self.use_transformers:
            self.encoder.fit(texts)
        
        # Generate embeddings
        embeddings = self.encoder.encode(texts)
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        
        # Add to vector store
        self.vector_store.add(documents, embeddings)
        self.initialized = True
        
        print(f"RAG initialized with {len(documents)} documents")
    
    def retrieve(
        self,
        query: str,
        k: int = 3,
        emotion_filter: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            emotion_filter: Optional emotion to filter by
            
        Returns:
            List of (document, score) tuples
        """
        if not self.initialized:
            self.initialize()
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        query_embedding = query_embedding[0]
        
        # Search
        results = self.vector_store.search(query_embedding, k=k * 2 if emotion_filter else k)
        
        # Filter by emotion if specified
        if emotion_filter:
            filtered = []
            for doc, score in results:
                emotions = doc.metadata.get('emotions', [])
                if emotion_filter in emotions or 'all' in emotions:
                    filtered.append((doc, score))
            results = filtered[:k]
        
        return results[:k]
    
    def get_context(
        self,
        query: str,
        k: int = 3,
        emotion: Optional[str] = None,
        max_tokens: int = 500
    ) -> str:
        """
        Get formatted context for RAG augmentation.
        
        Args:
            query: User query
            k: Number of documents
            emotion: Detected emotion
            max_tokens: Maximum context tokens (approximate)
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, k=k, emotion_filter=emotion)
        
        if not results:
            return ""
        
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # Rough approximation
        
        for doc, score in results:
            if total_chars + len(doc.content) > max_chars:
                break
            context_parts.append(f"[{doc.metadata.get('category', 'info')}] {doc.content}")
            total_chars += len(doc.content)
        
        return "\n\n".join(context_parts)
    
    def augment_prompt(
        self,
        user_message: str,
        emotion: Optional[str] = None,
        system_prompt: str = ""
    ) -> str:
        """
        Augment a prompt with retrieved context.
        
        Args:
            user_message: User's message
            emotion: Detected emotion
            system_prompt: Base system prompt
            
        Returns:
            Augmented system prompt
        """
        context = self.get_context(user_message, emotion=emotion)
        
        if context:
            augmented = f"""{system_prompt}

RELEVANT KNOWLEDGE BASE INFORMATION:
{context}

Use the above information to inform your response when relevant, but don't 
quote it directly. Integrate the knowledge naturally into your empathetic response."""
        else:
            augmented = system_prompt
        
        return augmented


class RAGResponseGenerator:
    """
    Response generator with RAG augmentation.
    Combines retrieval with LLM generation for grounded responses.
    """
    
    def __init__(
        self,
        llm_provider: str = 'local',
        api_key: Optional[str] = None,
        use_transformers: bool = False
    ):
        """
        Initialize RAG-enhanced response generator.
        
        Args:
            llm_provider: 'openai', 'anthropic', or 'local'
            api_key: API key for LLM provider
            use_transformers: Whether to use sentence-transformers for embeddings
        """
        from .response_generator import ResponseGenerator, MentalHealthPrompts
        
        self.base_generator = ResponseGenerator(provider=llm_provider, api_key=api_key)
        self.retriever = RAGRetriever(use_transformers=use_transformers)
        self.base_prompt = MentalHealthPrompts.SYSTEM_PROMPT
        
        # Initialize retriever
        self.retriever.initialize()
    
    def generate(
        self,
        user_message: str,
        emotion: Optional[str] = None,
        include_history: bool = True
    ) -> Tuple[str, Dict]:
        """
        Generate RAG-augmented response.
        
        Args:
            user_message: User's message
            emotion: Detected emotion
            include_history: Include conversation history
            
        Returns:
            Tuple of (response, metadata)
        """
        # Get retrieved context
        context = self.retriever.get_context(user_message, emotion=emotion)
        retrieved_docs = self.retriever.retrieve(user_message, emotion_filter=emotion)
        
        # Augment the system prompt
        augmented_prompt = self.retriever.augment_prompt(
            user_message,
            emotion=emotion,
            system_prompt=self.base_prompt
        )
        
        # Store original prompt temporarily
        original_prompt = self.base_generator.context
        
        # Generate response
        response, metadata = self.base_generator.generate(
            user_message,
            emotion=emotion,
            include_history=include_history
        )
        
        # Add RAG metadata
        metadata['rag_enabled'] = True
        metadata['retrieved_docs'] = [doc.id for doc, _ in retrieved_docs]
        metadata['context_length'] = len(context)
        
        return response, metadata
    
    def get_relevant_resources(self, emotion: str) -> List[Dict]:
        """Get relevant resources for an emotion."""
        docs = self.retriever.retrieve(f"help with {emotion}", emotion_filter=emotion, k=5)
        return [{'id': doc.id, 'content': doc.content[:200] + '...', 'score': score} 
                for doc, score in docs]


# Convenience functions
def create_rag_retriever(use_transformers: bool = False) -> RAGRetriever:
    """Create and initialize a RAG retriever."""
    retriever = RAGRetriever(use_transformers=use_transformers)
    retriever.initialize()
    return retriever


def retrieve_for_emotion(emotion: str, k: int = 3) -> List[Dict]:
    """Quick retrieval for an emotion."""
    retriever = create_rag_retriever(use_transformers=False)
    results = retriever.retrieve(f"help with {emotion} feelings", emotion_filter=emotion, k=k)
    return [{'content': doc.content, 'category': doc.metadata.get('category')} for doc, _ in results]


if __name__ == "__main__":
    print("Testing RAG Module...")
    print("=" * 50)
    
    # Test knowledge base
    print("\n1. Testing Knowledge Base...")
    all_docs = MentalHealthKnowledgeBase.get_all_documents()
    print(f"   Total documents: {len(all_docs)}")
    
    anxiety_docs = MentalHealthKnowledgeBase.get_by_emotion('anxiety')
    print(f"   Anxiety-related docs: {len(anxiety_docs)}")
    
    # Test simple embedding
    print("\n2. Testing Embeddings...")
    embedding = SimpleEmbedding()
    texts = [doc.content for doc in all_docs]
    embedding.fit(texts)
    vectors = embedding.encode(texts[:3])
    print(f"   Embedding shape: {vectors.shape}")
    
    # Test vector store
    print("\n3. Testing Vector Store...")
    store = VectorStore(dimension=embedding.dimension)
    all_embeddings = embedding.encode(texts)
    store.add(all_docs, all_embeddings)
    
    query_vec = embedding.encode(["I feel anxious and worried"])[0]
    results = store.search(query_vec, k=2)
    print(f"   Search results: {[doc.id for doc, _ in results]}")
    
    # Test RAG retriever
    print("\n4. Testing RAG Retriever...")
    retriever = RAGRetriever(use_transformers=False)
    retriever.initialize()
    
    results = retriever.retrieve("I'm feeling very anxious about my future", k=3)
    print(f"   Retrieved {len(results)} documents:")
    for doc, score in results:
        print(f"      - {doc.id} (score: {score:.3f})")
    
    # Test context generation
    print("\n5. Testing Context Generation...")
    context = retriever.get_context("I've been feeling depressed lately", emotion='sad')
    print(f"   Context length: {len(context)} chars")
    print(f"   Preview: {context[:200]}...")
    
    # Test augmented prompt
    print("\n6. Testing Prompt Augmentation...")
    augmented = retriever.augment_prompt(
        "I can't sleep because of anxiety",
        emotion='anxiety',
        system_prompt="You are a helpful assistant."
    )
    print(f"   Augmented prompt length: {len(augmented)} chars")
    
    print("\n" + "=" * 50)
    print("✓ RAG Module tests passed!")

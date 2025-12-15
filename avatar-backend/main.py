"""
Mental Health Avatar Backend
===========================

Interactive Empathy: Building a Facially Expressive Mental Health Companion

Based on the research paper implementing:
- Semantic Analysis for emotion detection (Section 3.2)
- VAE for facial expression generation (Section 3.3)  
- GPT-3.5/Claude for response generation (Section 3.1)
- Text-to-Speech with Lip Sync (Section 3.4)

Usage:
    # Start API server
    python main.py serve
    
    # Train VAE model
    python main.py train
    
    # Interactive demo
    python main.py demo
"""

import sys
import os
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).parent))


def serve():
    """Start the FastAPI server."""
    import uvicorn
    from api.server import app
    
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', '0.0.0.0')
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║     Mental Health Avatar API Server                          ║
║     Based on: Interactive Empathy Research Paper             ║
╠══════════════════════════════════════════════════════════════╣
║  Endpoints:                                                  ║
║    POST /analyze          - Emotion analysis                 ║
║    POST /generate-response - LLM response generation         ║
║    POST /generate-expression - VAE expression generation     ║
║    POST /synthesize       - TTS with lip sync                ║
║    POST /chat             - Complete chat endpoint           ║
╠══════════════════════════════════════════════════════════════╣
║  Documentation: http://{host}:{port}/docs                        ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host=host, port=port)


def train():
    """Train the VAE model."""
    from train_vae import main as train_main
    train_main()


def demo():
    """Interactive demo."""
    from models.semantic_analyzer import SemanticAnalyzer
    from models.response_generator import ResponseGenerator
    from models.tts_lipsync import LipSyncGenerator
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║     Mental Health Avatar - Interactive Demo                  ║
║     Type 'quit' to exit                                      ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    analyzer = SemanticAnalyzer(use_transformer=False)
    generator = ResponseGenerator(provider='local')
    lipsync = LipSyncGenerator()
    
    print("Avatar: Hello! I'm your mental health companion. How are you feeling today?\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nAvatar: Take care of yourself. Goodbye! 💜")
                break
            
            if not user_input:
                continue
            
            # Analyze emotion
            analysis = analyzer.analyze_for_avatar(user_input)
            print(f"[Detected emotion: {analysis['dominant_emotion']} ({analysis['confidence']:.2f})]")
            
            # Generate response
            response, meta = generator.generate(user_input, analysis['dominant_emotion'])
            print(f"\nAvatar [{meta['emotion']}]: {response}")
            
            # Generate lip sync data
            lip_data = lipsync.from_text(response, duration=len(response.split()) / 2.5)
            print(f"[Lip sync: {len(lip_data.frames)} frames, {lip_data.duration:.1f}s]")
            
            # Show expression params
            print(f"[Expression: {analysis['expression_params']}]\n")
            
        except KeyboardInterrupt:
            print("\n\nAvatar: Take care! 💜")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def test():
    """Run all tests."""
    print("Running tests...")
    
    # Test semantic analyzer
    print("\n1. Testing Semantic Analyzer...")
    from models.semantic_analyzer import SemanticAnalyzer
    analyzer = SemanticAnalyzer()
    result = analyzer.analyze_for_avatar("I'm feeling happy today!")
    assert result['dominant_emotion'] == 'happy'
    print("   ✓ Semantic Analyzer OK")
    
    # Test response generator
    print("\n2. Testing Response Generator...")
    from models.response_generator import ResponseGenerator
    gen = ResponseGenerator(provider='local')
    response, meta = gen.generate("I feel sad", "sad")
    assert len(response) > 0
    print("   ✓ Response Generator OK")
    
    # Test lip sync
    print("\n3. Testing Lip Sync...")
    from models.tts_lipsync import LipSyncGenerator
    lipsync = LipSyncGenerator()
    data = lipsync.from_text("Hello world", duration=1.0)
    assert len(data.frames) > 0
    print("   ✓ Lip Sync OK")
    
    # Test VAE
    print("\n4. Testing VAE...")
    try:
        import torch
        from models.vae import SimplifiedFacialVAE, create_emotion_vector
        vae = SimplifiedFacialVAE()
        emotion = create_emotion_vector(happy=1.0)
        expr = vae.generate(emotion)
        assert expr.shape[1] == 42
        print("   ✓ VAE OK")
    except ImportError:
        print("   ⚠ VAE skipped (torch not installed)")
    
    print("\n✓ All tests passed!")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nCommands:")
        print("  serve  - Start API server")
        print("  train  - Train VAE model")
        print("  demo   - Interactive demo")
        print("  test   - Run tests")
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == 'serve':
        serve()
    elif command == 'train':
        train()
    elif command == 'demo':
        demo()
    elif command == 'test':
        test()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()

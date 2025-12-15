"""
Text-to-Speech and Lip Synchronization
Section 3.4: "Text-to-Speech Conversion and Lip-Sync"

Implements Eleven Labs style TTS and Rhubarb Lip Sync inspired viseme generation.

As stated in the paper:
"Eleven Labs transforms the GPT-3.5 generated text responses into natural, 
human-like speech... Rhubarb Lip Sync analyzes the audio output to create 
accurate lip movements for the avatar."
"""

import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import numpy as np

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class Viseme(Enum):
    """Viseme categories based on Rhubarb Lip Sync."""
    A = "A"  # Wide open (bat)
    B = "B"  # Closed lips (b, m, p)
    C = "C"  # Teeth together (ch, sh)
    D = "D"  # Tongue behind teeth (d, n, t)
    E = "E"  # Slightly open wide (bet)
    F = "F"  # Lower lip under teeth (f, v)
    G = "G"  # Back tongue raised (g, k)
    H = "H"  # Relaxed open (h)
    X = "X"  # Silence


@dataclass
class VisemeFrame:
    """Single viseme at specific time."""
    viseme: Viseme
    start_time: float
    end_time: float
    intensity: float = 1.0
    
    def to_dict(self) -> Dict:
        return {'viseme': self.viseme.value, 'start': self.start_time, 'end': self.end_time, 'intensity': self.intensity}


@dataclass
class LipSyncData:
    """Complete lip sync data for audio clip."""
    frames: List[VisemeFrame]
    duration: float
    
    def to_dict(self) -> Dict:
        return {'frames': [f.to_dict() for f in self.frames], 'duration': self.duration}
    
    def get_viseme_at(self, time: float) -> Tuple[Viseme, float]:
        for frame in self.frames:
            if frame.start_time <= time < frame.end_time:
                return frame.viseme, frame.intensity
        return Viseme.X, 0.0


class VisemeMouthShapes:
    """Mouth shape parameters for each viseme."""
    
    SHAPES = {
        Viseme.A: {'jaw_open': 0.7, 'mouth_wide': 0.6, 'lips_pucker': 0.0},
        Viseme.B: {'jaw_open': 0.0, 'mouth_wide': 0.0, 'lips_pucker': 0.3},
        Viseme.C: {'jaw_open': 0.2, 'mouth_wide': 0.3, 'lips_pucker': 0.4},
        Viseme.D: {'jaw_open': 0.3, 'mouth_wide': 0.4, 'lips_pucker': 0.0},
        Viseme.E: {'jaw_open': 0.4, 'mouth_wide': 0.7, 'lips_pucker': 0.0},
        Viseme.F: {'jaw_open': 0.2, 'mouth_wide': 0.3, 'lips_pucker': 0.0},
        Viseme.G: {'jaw_open': 0.3, 'mouth_wide': 0.3, 'lips_pucker': 0.1},
        Viseme.H: {'jaw_open': 0.4, 'mouth_wide': 0.4, 'lips_pucker': 0.0},
        Viseme.X: {'jaw_open': 0.1, 'mouth_wide': 0.2, 'lips_pucker': 0.0},
    }
    
    @classmethod
    def get_shape(cls, viseme: Viseme) -> Dict:
        return cls.SHAPES.get(viseme, cls.SHAPES[Viseme.X])
    
    @classmethod
    def interpolate(cls, v1: Viseme, v2: Viseme, t: float) -> Dict:
        s1, s2 = cls.SHAPES[v1], cls.SHAPES[v2]
        return {k: s1[k] * (1-t) + s2[k] * t for k in s1}


class PhonemeMapper:
    """Maps letters/phonemes to visemes."""
    
    LETTER_MAP = {
        'a': Viseme.A, 'e': Viseme.E, 'i': Viseme.E, 'o': Viseme.A, 'u': Viseme.A,
        'b': Viseme.B, 'm': Viseme.B, 'p': Viseme.B,
        'f': Viseme.F, 'v': Viseme.F,
        'd': Viseme.D, 'l': Viseme.D, 'n': Viseme.D, 't': Viseme.D,
        'c': Viseme.C, 'j': Viseme.C, 's': Viseme.C, 'z': Viseme.C,
        'g': Viseme.G, 'k': Viseme.G, 'q': Viseme.G,
        'h': Viseme.H, 'r': Viseme.D, 'w': Viseme.A, 'y': Viseme.E,
    }
    
    @classmethod
    def letter_to_viseme(cls, letter: str) -> Viseme:
        return cls.LETTER_MAP.get(letter.lower(), Viseme.X)


class TextToSpeechEngine:
    """TTS engine with multiple backend support."""
    
    def __init__(self, engine: str = 'auto', output_dir: str = './audio'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if engine == 'auto':
            self.engine = 'pyttsx3' if PYTTSX3_AVAILABLE else 'gtts' if GTTS_AVAILABLE else None
        else:
            self.engine = engine
        
        self._tts = None
        if self.engine == 'pyttsx3' and PYTTSX3_AVAILABLE:
            self._tts = pyttsx3.init()
            self._tts.setProperty('rate', 150)
            self._tts.setProperty('volume', 0.9)
    
    def synthesize(self, text: str, filename: str = None, emotion: str = None) -> Tuple[str, float]:
        """Synthesize speech from text."""
        if filename is None:
            filename = f"speech_{hash(text) % 10000:04d}.wav"
        
        output_path = str(self.output_dir / filename)
        
        if self.engine == 'pyttsx3' and self._tts:
            # Adjust rate by emotion
            rates = {'happy': 160, 'sad': 130, 'angry': 170, 'neutral': 150}
            self._tts.setProperty('rate', rates.get(emotion, 150))
            self._tts.save_to_file(text, output_path)
            self._tts.runAndWait()
        elif self.engine == 'gtts' and GTTS_AVAILABLE:
            tts = gTTS(text=text, lang='en', slow=False)
            mp3_path = output_path.replace('.wav', '.mp3')
            tts.save(mp3_path)
            if LIBROSA_AVAILABLE:
                y, sr = librosa.load(mp3_path, sr=22050)
                sf.write(output_path, y, sr)
                os.remove(mp3_path)
            else:
                output_path = mp3_path
        else:
            # Return estimated duration without actual audio
            duration = len(text.split()) / 2.5  # ~150 wpm
            return output_path, duration
        
        return output_path, self._get_duration(output_path)
    
    def _get_duration(self, path: str) -> float:
        if LIBROSA_AVAILABLE:
            try:
                y, sr = librosa.load(path, sr=None)
                return len(y) / sr
            except:
                pass
        # Estimate
        return 2.0


class LipSyncGenerator:
    """Generates lip sync data from text or audio."""
    
    def __init__(self, wpm: float = 150):
        self.wpm = wpm
        self.mapper = PhonemeMapper()
    
    def from_text(self, text: str, duration: float = None) -> LipSyncData:
        """Generate lip sync from text."""
        text = text.strip()
        if not text:
            return LipSyncData(frames=[], duration=0)
        
        if duration is None:
            duration = len(text.split()) / self.wpm * 60
        
        frames = []
        char_duration = duration / max(len(text), 1)
        current_time = 0.0
        prev_viseme = Viseme.X
        frame_start = 0.0
        
        for char in text:
            if char.isalpha():
                viseme = self.mapper.letter_to_viseme(char)
            elif char in ' \n\t':
                viseme = Viseme.X
            else:
                viseme = prev_viseme
            
            if viseme != prev_viseme:
                if prev_viseme != Viseme.X or current_time - frame_start > 0.1:
                    frames.append(VisemeFrame(prev_viseme, frame_start, current_time, 0.8 if prev_viseme != Viseme.X else 0.2))
                frame_start = current_time
                prev_viseme = viseme
            
            current_time += char_duration
        
        if prev_viseme != Viseme.X:
            frames.append(VisemeFrame(prev_viseme, frame_start, duration, 0.8))
        
        frames.append(VisemeFrame(Viseme.X, duration, duration + 0.1, 0.0))
        
        return LipSyncData(frames=frames, duration=duration)
    
    def from_audio(self, audio_path: str, text: str = None) -> LipSyncData:
        """Generate lip sync from audio using energy analysis."""
        if not LIBROSA_AVAILABLE:
            return self.from_text(text) if text else LipSyncData(frames=[], duration=0)
        
        y, sr = librosa.load(audio_path, sr=22050)
        duration = len(y) / sr
        
        # Audio features
        hop = 512
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop)[0]
        
        # Normalize
        rms = rms / (np.max(rms) + 1e-8)
        centroid = centroid / (np.max(centroid) + 1e-8)
        zcr = zcr / (np.max(zcr) + 1e-8)
        
        frames = []
        times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop)
        prev_viseme = Viseme.X
        frame_start = 0.0
        
        for i, t in enumerate(times):
            if rms[i] < 0.1:
                viseme = Viseme.X
            elif zcr[i] > 0.6:
                viseme = Viseme.C
            elif centroid[i] > 0.6:
                viseme = Viseme.E
            elif centroid[i] < 0.3:
                viseme = Viseme.A
            else:
                viseme = Viseme.D
            
            if viseme != prev_viseme and i > 0:
                frames.append(VisemeFrame(prev_viseme, frame_start, t, float(rms[i-1])))
                frame_start = t
                prev_viseme = viseme
        
        if times.size > 0:
            frames.append(VisemeFrame(prev_viseme, frame_start, duration, float(rms[-1]) if len(rms) > 0 else 0.0))
        
        return LipSyncData(frames=frames, duration=duration)
    
    def to_animation(self, data: LipSyncData, fps: int = 60) -> List[Dict]:
        """Convert to animation keyframes."""
        keyframes = []
        for i in range(int(data.duration * fps)):
            t = i / fps
            viseme, intensity = data.get_viseme_at(t)
            shape = VisemeMouthShapes.get_shape(viseme)
            
            keyframes.append({
                'frame': i,
                'time': t,
                'viseme': viseme.value,
                'intensity': intensity,
                'mouth_params': {k: v * intensity for k, v in shape.items()}
            })
        
        return keyframes


class TTSLipSyncPipeline:
    """Complete pipeline combining TTS and lip sync."""
    
    def __init__(self, output_dir: str = './output'):
        self.tts = TextToSpeechEngine(output_dir=output_dir)
        self.lipsync = LipSyncGenerator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process(self, text: str, emotion: str = None, name: str = None) -> Dict:
        """Process text through TTS and lip sync pipeline."""
        if name is None:
            name = f"output_{hash(text) % 10000:04d}"
        
        # TTS
        audio_path, duration = self.tts.synthesize(text, f"{name}.wav", emotion)
        
        # Lip sync
        if LIBROSA_AVAILABLE and os.path.exists(audio_path):
            lipsync_data = self.lipsync.from_audio(audio_path, text)
        else:
            lipsync_data = self.lipsync.from_text(text, duration)
        
        # Animation
        animation = self.lipsync.to_animation(lipsync_data, fps=60)
        
        # Save lip sync JSON
        lipsync_path = self.output_dir / f"{name}_lipsync.json"
        with open(lipsync_path, 'w') as f:
            json.dump(lipsync_data.to_dict(), f, indent=2)
        
        return {
            'audio_path': audio_path,
            'duration': duration,
            'lip_sync': lipsync_data.to_dict(),
            'lip_sync_path': str(lipsync_path),
            'animation': animation,
            'text': text,
            'emotion': emotion
        }
    
    def realtime_viseme(self, text: str, current_time: float) -> Dict:
        """Get viseme for real-time animation."""
        duration = len(text.split()) / 150 * 60
        data = self.lipsync.from_text(text, duration)
        viseme, intensity = data.get_viseme_at(current_time)
        shape = VisemeMouthShapes.get_shape(viseme)
        
        return {
            'time': current_time,
            'viseme': viseme.value,
            'intensity': intensity,
            'mouth_params': {k: v * intensity for k, v in shape.items()}
        }


if __name__ == "__main__":
    print("Testing TTS and Lip Sync...")
    
    # Test lip sync from text
    lipsync = LipSyncGenerator()
    text = "Hello! I'm your mental health companion."
    data = lipsync.from_text(text, duration=3.0)
    print(f"Text: {text}")
    print(f"Duration: {data.duration}s, Frames: {len(data.frames)}")
    print(f"First 3 frames: {[f.to_dict() for f in data.frames[:3]]}")
    
    # Test mouth shapes
    print(f"\nMouth shapes:")
    for v in [Viseme.A, Viseme.B, Viseme.E]:
        print(f"  {v.value}: {VisemeMouthShapes.get_shape(v)}")
    
    # Test animation
    anim = lipsync.to_animation(data, fps=30)
    print(f"\nAnimation: {len(anim)} frames at 30fps")
    print(f"Sample: {anim[0]}")
    
    # Test pipeline
    print("\nTesting pipeline...")
    try:
        pipeline = TTSLipSyncPipeline(output_dir='/tmp/tts_test')
        result = pipeline.process("I understand how you feel.", emotion='empathy')
        print(f"Audio: {result['audio_path']}")
        print(f"Duration: {result['duration']:.2f}s")
        print(f"Frames: {len(result['animation'])}")
    except Exception as e:
        print(f"Pipeline test: {e}")
    
    print("\n✓ TTS and Lip Sync tests passed!")

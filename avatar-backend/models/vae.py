"""
Variational Autoencoder (VAE) for Facial Expression Generation
Section 3.3: "Facial Expression Generation with Variational Autoencoders (VAE)"

The VAE model:
1. Encodes high-dimensional facial expression data into lower-dimensional latent space
2. Decodes latent vectors conditioned on emotion to generate facial expressions
3. Trained on CoMA dataset for realistic 3D facial expressions

Reference: Zou et al. 2023 - "3D Facial Expression Generator Based on Transformer VAE"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict, List, Optional


class FacialExpressionEncoder(nn.Module):
    """Encoder: Maps facial mesh vertices to latent distribution parameters."""
    
    def __init__(
        self,
        input_dim: int = 5023 * 3,  # CoMA mesh vertices
        hidden_dims: List[int] = [2048, 1024, 512, 256],
        latent_dim: int = 64
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)


class FacialExpressionDecoder(nn.Module):
    """Decoder: Maps latent vector + emotion to facial mesh vertices."""
    
    def __init__(
        self,
        latent_dim: int = 64,
        emotion_dim: int = 6,
        hidden_dims: List[int] = [256, 512, 1024, 2048],
        output_dim: int = 5023 * 3
    ):
        super().__init__()
        
        # Emotion embedding
        self.emotion_embedding = nn.Sequential(
            nn.Linear(emotion_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64)
        )
        
        layers = []
        in_dim = latent_dim + 64
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor, emotion: torch.Tensor) -> torch.Tensor:
        emotion_emb = self.emotion_embedding(emotion)
        z_cond = torch.cat([z, emotion_emb], dim=1)
        return self.decoder(z_cond)


class FacialExpressionVAE(nn.Module):
    """
    Complete VAE for Facial Expression Generation.
    
    As described in Section 3.3:
    "Utilizing the VAE's capability to handle complex human emotions, it translates 
    emotional metrics like happiness and anger into corresponding avatar expressions."
    """
    
    EMOTION_NAMES = ['happy', 'angry', 'sad', 'fear', 'surprised', 'neutral']
    
    def __init__(
        self,
        input_dim: int = 5023 * 3,
        hidden_dims: List[int] = [2048, 1024, 512, 256],
        latent_dim: int = 64,
        emotion_dim: int = 6
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.emotion_dim = emotion_dim
        
        self.encoder = FacialExpressionEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder = FacialExpressionDecoder(latent_dim, emotion_dim, hidden_dims[::-1], input_dim)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE training."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor, emotion: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, emotion)
        return recon, mu, logvar
    
    def generate(self, emotion: torch.Tensor, num_samples: int = 1, temperature: float = 1.0) -> torch.Tensor:
        """Generate facial expressions from emotion vectors."""
        self.eval()
        with torch.no_grad():
            if emotion.dim() == 1:
                emotion = emotion.unsqueeze(0)
            if emotion.size(0) == 1 and num_samples > 1:
                emotion = emotion.repeat(num_samples, 1)
            
            z = torch.randn(emotion.size(0), self.latent_dim, device=emotion.device) * temperature
            return self.decoder(z, emotion)
    
    def interpolate(self, emotion_start: torch.Tensor, emotion_end: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """Interpolate between two emotional expressions."""
        self.eval()
        with torch.no_grad():
            alphas = torch.linspace(0, 1, steps, device=emotion_start.device)
            z = torch.randn(1, self.latent_dim, device=emotion_start.device)
            
            expressions = []
            for alpha in alphas:
                emotion = (1 - alpha) * emotion_start + alpha * emotion_end
                expr = self.decoder(z, emotion.unsqueeze(0))
                expressions.append(expr)
            
            return torch.cat(expressions, dim=0)


class SimplifiedFacialVAE(nn.Module):
    """
    Simplified VAE for blend shape/morph target based avatars.
    Works with expression parameters instead of full mesh vertices.
    """
    
    # ARKit-style blend shape names
    BLEND_SHAPES = [
        'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
        'eyeBlinkLeft', 'eyeBlinkRight', 'eyeSquintLeft', 'eyeSquintRight',
        'eyeWideLeft', 'eyeWideRight', 'jawOpen', 'jawForward',
        'mouthClose', 'mouthFunnel', 'mouthPucker', 'mouthLeft', 'mouthRight',
        'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft', 'mouthFrownRight',
        'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft', 'mouthStretchRight',
        'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
        'mouthPressLeft', 'mouthPressRight', 'mouthLowerDownLeft', 'mouthLowerDownRight',
        'mouthUpperUpLeft', 'mouthUpperUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
        'noseSneerLeft', 'noseSneerRight', 'tongueOut'
    ]
    
    def __init__(
        self,
        expression_dim: int = 42,  # Number of blend shapes
        emotion_dim: int = 6,
        latent_dim: int = 16,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.expression_dim = expression_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(expression_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Emotion embedding
        self.emotion_emb = nn.Sequential(
            nn.Linear(emotion_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, expression_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std
    
    def decode(self, z: torch.Tensor, emotion: torch.Tensor) -> torch.Tensor:
        emotion_emb = self.emotion_emb(emotion)
        return self.decoder(torch.cat([z, emotion_emb], dim=-1))
    
    def forward(self, x: torch.Tensor, emotion: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, emotion), mu, logvar
    
    def generate(self, emotion: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Generate blend shape values from emotion vector."""
        self.eval()
        with torch.no_grad():
            if emotion.dim() == 1:
                emotion = emotion.unsqueeze(0)
            z = torch.randn(emotion.size(0), self.latent_dim, device=emotion.device) * temperature
            return self.decode(z, emotion)


class VAELoss(nn.Module):
    """Loss function for VAE training: Reconstruction + KL Divergence."""
    
    def __init__(self, recon_weight: float = 1.0, kl_weight: float = 0.001, kl_annealing: bool = True):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.kl_annealing = kl_annealing
        self.epoch = 0
    
    def forward(self, recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        recon_loss = F.mse_loss(recon, target)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        kl_w = min(self.kl_weight, self.kl_weight * self.epoch / 10) if self.kl_annealing else self.kl_weight
        total = self.recon_weight * recon_loss + kl_w * kl_loss
        
        return total, {'total': total.item(), 'recon': recon_loss.item(), 'kl': kl_loss.item()}
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch


class ExpressionDataset(Dataset):
    """Dataset for training VAE on facial expressions."""
    
    def __init__(self, expressions: np.ndarray, emotions: np.ndarray):
        self.expressions = torch.FloatTensor(expressions)
        self.emotions = torch.FloatTensor(emotions)
    
    def __len__(self):
        return len(self.expressions)
    
    def __getitem__(self, idx):
        return self.expressions[idx], self.emotions[idx]


def create_emotion_vector(happy=0., angry=0., sad=0., fear=0., surprised=0., neutral=0.) -> torch.Tensor:
    """Create normalized emotion tensor."""
    emotions = torch.tensor([happy, angry, sad, fear, surprised, neutral])
    return emotions / (emotions.sum() + 1e-8)


# Predefined expression templates for each emotion
EXPRESSION_TEMPLATES = {
    'happy': {
        'mouthSmileLeft': 0.8, 'mouthSmileRight': 0.8,
        'cheekSquintLeft': 0.3, 'cheekSquintRight': 0.3,
        'browInnerUp': 0.2, 'eyeSquintLeft': 0.2, 'eyeSquintRight': 0.2
    },
    'sad': {
        'mouthFrownLeft': 0.6, 'mouthFrownRight': 0.6,
        'browDownLeft': 0.4, 'browDownRight': 0.4,
        'browInnerUp': 0.5
    },
    'angry': {
        'browDownLeft': 0.7, 'browDownRight': 0.7,
        'eyeSquintLeft': 0.4, 'eyeSquintRight': 0.4,
        'jawForward': 0.3, 'mouthPressLeft': 0.3, 'mouthPressRight': 0.3
    },
    'fear': {
        'eyeWideLeft': 0.7, 'eyeWideRight': 0.7,
        'browInnerUp': 0.6, 'browOuterUpLeft': 0.4, 'browOuterUpRight': 0.4,
        'jawOpen': 0.3
    },
    'surprised': {
        'eyeWideLeft': 0.8, 'eyeWideRight': 0.8,
        'browInnerUp': 0.7, 'browOuterUpLeft': 0.6, 'browOuterUpRight': 0.6,
        'jawOpen': 0.5
    },
    'neutral': {}
}


if __name__ == "__main__":
    print("Testing VAE Models...")
    
    # Test full VAE
    vae = FacialExpressionVAE(input_dim=5023*3, latent_dim=64)
    x = torch.randn(4, 5023*3)
    emotion = torch.softmax(torch.randn(4, 6), dim=1)
    recon, mu, logvar = vae(x, emotion)
    print(f"Full VAE: input {x.shape} -> output {recon.shape}")
    
    # Test generation
    test_emotion = create_emotion_vector(happy=0.8, surprised=0.2)
    generated = vae.generate(test_emotion, num_samples=2)
    print(f"Generated: {generated.shape}")
    
    # Test simplified VAE
    simple_vae = SimplifiedFacialVAE(expression_dim=42)
    x_simple = torch.rand(4, 42)
    recon_s, _, _ = simple_vae(x_simple, emotion)
    print(f"Simplified VAE: input {x_simple.shape} -> output {recon_s.shape}")
    
    print("✓ VAE tests passed!")

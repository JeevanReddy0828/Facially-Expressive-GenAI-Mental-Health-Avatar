"""
VAE Training Script for Facial Expression Generation
Trains the VAE model on facial expression data (CoMA dataset format).

Reference: Ranjan et al. 2018 - CoMA dataset for 3D facial expressions
"""

import os
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional

from models.vae import FacialExpressionVAE, SimplifiedFacialVAE, VAELoss, EXPRESSION_TEMPLATES


class VAETrainer:
    """Trainer for Facial Expression VAE."""
    
    def __init__(
        self,
        model_type: str = 'simplified',  # 'full' or 'simplified'
        latent_dim: int = 64,
        learning_rate: float = 1e-4,
        device: str = 'auto'
    ):
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        if model_type == 'full':
            self.model = FacialExpressionVAE(latent_dim=latent_dim).to(self.device)
        else:
            self.model = SimplifiedFacialVAE(latent_dim=latent_dim // 4).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        self.loss_fn = VAELoss(kl_weight=0.001, kl_annealing=True)
        
        self.history = {'train_loss': [], 'val_loss': []}
    
    def generate_synthetic_data(self, num_samples: int = 1000) -> tuple:
        """Generate synthetic training data from expression templates."""
        expressions = []
        emotions = []
        
        emotion_names = list(EXPRESSION_TEMPLATES.keys())
        num_blend_shapes = 42
        
        for _ in range(num_samples):
            # Random emotion or blend
            if np.random.random() < 0.7:
                # Single emotion
                emotion_idx = np.random.randint(len(emotion_names))
                emotion_name = emotion_names[emotion_idx]
                emotion_vec = np.zeros(6)
                emotion_vec[emotion_idx] = 1.0
                
                template = EXPRESSION_TEMPLATES[emotion_name]
            else:
                # Blend of emotions
                weights = np.random.dirichlet(np.ones(len(emotion_names)))
                emotion_vec = weights
                
                template = {}
                for i, name in enumerate(emotion_names):
                    for k, v in EXPRESSION_TEMPLATES[name].items():
                        template[k] = template.get(k, 0) + v * weights[i]
            
            # Create expression vector
            expr = np.zeros(num_blend_shapes)
            blend_shape_names = SimplifiedFacialVAE.BLEND_SHAPES
            
            for name, value in template.items():
                if name in blend_shape_names:
                    idx = blend_shape_names.index(name)
                    expr[idx] = value + np.random.normal(0, 0.05)  # Add noise
            
            # Add random variation
            expr += np.random.normal(0, 0.02, num_blend_shapes)
            expr = np.clip(expr, 0, 1)
            
            expressions.append(expr)
            emotions.append(emotion_vec)
        
        return np.array(expressions, dtype=np.float32), np.array(emotions, dtype=np.float32)
    
    def train(
        self,
        train_data: Optional[tuple] = None,
        val_data: Optional[tuple] = None,
        epochs: int = 100,
        batch_size: int = 32,
        checkpoint_dir: str = './checkpoints'
    ) -> Dict:
        """Train the VAE model."""
        
        # Generate synthetic data if not provided
        if train_data is None:
            print("Generating synthetic training data...")
            expressions, emotions = self.generate_synthetic_data(5000)
            split = int(0.9 * len(expressions))
            train_data = (expressions[:split], emotions[:split])
            val_data = (expressions[split:], emotions[split:])
        
        # Create dataloaders
        train_dataset = TensorDataset(
            torch.FloatTensor(train_data[0]),
            torch.FloatTensor(train_data[1])
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(
            torch.FloatTensor(val_data[0]),
            torch.FloatTensor(val_data[1])
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training loop
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        print(f"Training on {self.device}...")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        for epoch in range(epochs):
            self.loss_fn.set_epoch(epoch)
            
            # Training
            self.model.train()
            train_losses = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for expressions, emotions in pbar:
                expressions = expressions.to(self.device)
                emotions = emotions.to(self.device)
                
                self.optimizer.zero_grad()
                recon, mu, logvar = self.model(expressions, emotions)
                loss, loss_dict = self.loss_fn(recon, expressions, mu, logvar)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_losses.append(loss_dict['total'])
                pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})
            
            avg_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for expressions, emotions in val_loader:
                    expressions = expressions.to(self.device)
                    emotions = emotions.to(self.device)
                    
                    recon, mu, logvar = self.model(expressions, emotions)
                    loss, loss_dict = self.loss_fn(recon, expressions, mu, logvar)
                    val_losses.append(loss_dict['total'])
            
            avg_val_loss = np.mean(val_losses)
            self.history['val_loss'].append(avg_val_loss)
            
            self.scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save(checkpoint_path / 'best_model.pt')
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save(checkpoint_path / f'checkpoint_epoch_{epoch+1}.pt')
        
        # Save final model
        self.save(checkpoint_path / 'final_model.pt')
        
        # Save history
        with open(checkpoint_path / 'training_history.json', 'w') as f:
            json.dump(self.history, f)
        
        return self.history
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_type': self.model_type,
            'history': self.history
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})
        print(f"Model loaded from {path}")
    
    def generate_expression(self, emotion_name: str) -> np.ndarray:
        """Generate expression from emotion name."""
        emotion_map = {'happy': 0, 'angry': 1, 'sad': 2, 'fear': 3, 'surprised': 4, 'neutral': 5}
        
        emotion_vec = torch.zeros(1, 6).to(self.device)
        if emotion_name in emotion_map:
            emotion_vec[0, emotion_map[emotion_name]] = 1.0
        else:
            emotion_vec[0, 5] = 1.0  # neutral
        
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(emotion_vec)
        
        return generated.cpu().numpy()[0]


def main():
    """Main training function."""
    print("=" * 50)
    print("VAE Training for Facial Expression Generation")
    print("=" * 50)
    
    trainer = VAETrainer(
        model_type='simplified',
        latent_dim=16,
        learning_rate=1e-3
    )
    
    print("\nStarting training...")
    history = trainer.train(
        epochs=50,
        batch_size=32,
        checkpoint_dir='./checkpoints'
    )
    
    print("\nTraining complete!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    
    # Test generation
    print("\nTesting expression generation...")
    for emotion in ['happy', 'sad', 'angry', 'surprised', 'neutral']:
        expr = trainer.generate_expression(emotion)
        print(f"{emotion}: {expr[:5]}...")  # Show first 5 values
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()

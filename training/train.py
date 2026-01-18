"""
Training loop for feasibility classifiers.

Supports both MLP and CNN models with logging and checkpointing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from typing import Optional, Dict, Tuple
from tqdm import tqdm

from models.mlp_classifier import MLPClassifier
from models.cnn_classifier import CNNClassifier
from data_generation.dataset import FeasibilityDataset, create_dataloaders


class Trainer:
    """
    Trainer for feasibility classification models.
    
    Features:
    - Training with validation monitoring
    - Early stopping
    - Model checkpointing
    - TensorBoard logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        device: str = "auto",
        checkpoint_dir: str = "checkpoints/",
        log_dir: str = "logs/",
        use_images: bool = False
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            device: Device to train on ("auto", "cuda", "cpu")
            checkpoint_dir: Directory for saving checkpoints
            log_dir: Directory for TensorBoard logs
            use_images: Whether to use image inputs
        """
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_images = use_images
        
        # Optimizer and loss
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.BCELoss()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Setup directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.checkpoint_dir = checkpoint_dir
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        
    def train(
        self,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        save_every: int = 5
    ) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            num_epochs: Maximum number of epochs
            early_stopping_patience: Stop if no improvement for this many epochs
            save_every: Save checkpoint every N epochs
            
        Returns:
            Training history dict
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {self.model.get_num_parameters():,}")
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self._train_epoch()
            
            # Validation
            val_loss, val_acc = self._validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Log to TensorBoard
            self.writer.add_scalars('Loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalars('Accuracy', {
                'train': train_acc,
                'val': val_acc
            }, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                self._save_checkpoint('best_model.pt', epoch, val_loss, val_acc)
            else:
                self.epochs_without_improvement += 1
                
            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', epoch, val_loss, val_acc)
                
            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
                
        self.writer.close()
        print(f"\nBest validation accuracy: {self.best_val_acc:.4f}")
        
        return history
    
    def _train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            # Get inputs
            if self.use_images:
                images = batch['image'].to(self.device)
                actions = batch['action'].to(self.device)
                outputs = self.model(images, actions)
            else:
                inputs = batch['input'].to(self.device)
                outputs = self.model(inputs)
                
            labels = batch['label'].to(self.device)
            
            # Forward pass
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * labels.size(0)
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            
        return total_loss / total, correct / total
    
    def _validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if self.use_images:
                    images = batch['image'].to(self.device)
                    actions = batch['action'].to(self.device)
                    outputs = self.model(images, actions)
                else:
                    inputs = batch['input'].to(self.device)
                    outputs = self.model(inputs)
                    
                labels = batch['label'].to(self.device)
                
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * labels.size(0)
                preds = (outputs >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        return total_loss / total, correct / total
    
    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float, val_acc: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['val_acc']


def train_mlp(
    data_path: str,
    hidden_dims: list = [256, 256, 128],
    batch_size: int = 64,
    learning_rate: float = 0.001,
    num_epochs: int = 50,
    checkpoint_dir: str = "checkpoints/",
    log_dir: str = "logs/"
) -> MLPClassifier:
    """
    Convenience function to train an MLP classifier.
    
    Returns:
        Trained model
    """
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        data_path,
        batch_size=batch_size,
        use_images=False
    )
    
    # Get input dimension
    dataset = FeasibilityDataset(data_path)
    state_dim, action_dim, input_dim = dataset.get_input_dims()
    
    # Create model
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims
    )
    
    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        use_images=False
    )
    
    trainer.train(num_epochs=num_epochs)
    
    # Load best model
    trainer.load_checkpoint(os.path.join(checkpoint_dir, 'best_model.pt'))
    
    return model


def train_cnn(
    data_path: str,
    batch_size: int = 32,
    learning_rate: float = 0.0001,
    num_epochs: int = 50,
    checkpoint_dir: str = "checkpoints/",
    log_dir: str = "logs/"
) -> CNNClassifier:
    """
    Convenience function to train a CNN classifier.
    
    Returns:
        Trained model
    """
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        data_path,
        batch_size=batch_size,
        use_images=True
    )
    
    # Get action dimension
    dataset = FeasibilityDataset(data_path)
    state_dim, action_dim, _ = dataset.get_input_dims()
    
    # Create model
    model = CNNClassifier(
        action_dim=action_dim,
        backbone='resnet18',
        pretrained=True
    )
    
    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        use_images=True
    )
    
    trainer.train(num_epochs=num_epochs)
    
    # Load best model
    trainer.load_checkpoint(os.path.join(checkpoint_dir, 'best_model.pt'))
    
    return model

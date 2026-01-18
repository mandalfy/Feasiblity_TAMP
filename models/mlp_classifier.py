"""
MLP-based feasibility classifier.

Simple but fast model for state-vector inputs.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron for feasibility prediction.
    
    Takes concatenated state and action vectors as input.
    Outputs probability of motion plan being feasible.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 256, 128],
        dropout: float = 0.2,
        use_batch_norm: bool = True
    ):
        """
        Initialize the MLP classifier.
        
        Args:
            input_dim: Dimension of input (state + action)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Probability tensor of shape (batch_size, 1)
        """
        return self.network(x)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Make binary predictions.
        
        Args:
            x: Input tensor
            threshold: Classification threshold
            
        Returns:
            Binary predictions (0 or 1)
        """
        with torch.no_grad():
            probs = self.forward(x)
            return (probs >= threshold).float()
            
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPClassifierWithAttention(nn.Module):
    """
    MLP with attention mechanism for better feature processing.
    
    Separates state and action processing before combining.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        """
        Initialize the attention-based MLP.
        
        Args:
            state_dim: Dimension of state vector
            action_dim: Dimension of action vector
            hidden_dim: Hidden layer dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Separate encoders for state and action
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Cross-attention: action attends to state
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor (batch_size, state_dim)
            action: Action tensor (batch_size, action_dim)
            
        Returns:
            Probability tensor (batch_size, 1)
        """
        # Encode
        state_emb = self.state_encoder(state).unsqueeze(1)  # (B, 1, H)
        action_emb = self.action_encoder(action).unsqueeze(1)  # (B, 1, H)
        
        # Cross-attention
        attended, _ = self.cross_attention(action_emb, state_emb, state_emb)
        attended = attended.squeeze(1)  # (B, H)
        
        # Combine and classify
        combined = torch.cat([attended, action_emb.squeeze(1)], dim=-1)
        return self.classifier(combined)

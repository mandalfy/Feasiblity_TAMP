"""
CNN-based feasibility classifier.

Uses image input for scene understanding.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class CNNClassifier(nn.Module):
    """
    CNN classifier for image-based feasibility prediction.
    
    Uses a pretrained ResNet backbone followed by action conditioning.
    """
    
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 256,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3
    ):
        """
        Initialize the CNN classifier.
        
        Args:
            action_dim: Dimension of action vector
            hidden_dim: Hidden layer dimension
            backbone: ResNet variant ('resnet18', 'resnet34', 'resnet50')
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone weights
            dropout: Dropout probability
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Load backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            backbone_out_dim = 512
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            backbone_out_dim = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_out_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
            
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion and classification
        fusion_dim = backbone_out_dim + hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Feature attention (optional)
        self.use_attention = True
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(backbone_out_dim, backbone_out_dim // 4),
                nn.ReLU(),
                nn.Linear(backbone_out_dim // 4, backbone_out_dim),
                nn.Sigmoid()
            )
            
    def forward(
        self,
        image: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            image: Image tensor (batch_size, 3, H, W)
            action: Action tensor (batch_size, action_dim)
            
        Returns:
            Probability tensor (batch_size, 1)
        """
        # Extract image features
        img_features = self.backbone(image)
        img_features = img_features.flatten(1)  # (B, backbone_out_dim)
        
        # Apply attention
        if self.use_attention:
            attention_weights = self.attention(img_features)
            img_features = img_features * attention_weights
            
        # Encode action
        action_features = self.action_encoder(action)
        
        # Fuse and classify
        combined = torch.cat([img_features, action_features], dim=-1)
        return self.classifier(combined)
    
    def predict(
        self,
        image: torch.Tensor,
        action: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """Make binary predictions."""
        with torch.no_grad():
            probs = self.forward(image, action)
            return (probs >= threshold).float()
            
    def get_num_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LightweightCNN(nn.Module):
    """
    Lightweight CNN for faster inference.
    
    Custom architecture without pretrained weights.
    """
    
    def __init__(
        self,
        action_dim: int,
        image_size: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2
    ):
        """
        Initialize the lightweight CNN.
        
        Args:
            action_dim: Dimension of action vector
            image_size: Input image size (square)
            hidden_dim: Hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        # Simple CNN backbone
        self.cnn = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 16 -> 8
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        cnn_out_dim = 256
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        image: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        img_features = self.cnn(image)
        action_features = self.action_encoder(action)
        combined = torch.cat([img_features, action_features], dim=-1)
        return self.classifier(combined)

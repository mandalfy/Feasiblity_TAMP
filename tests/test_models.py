"""
Tests for ML models.

Run with: pytest tests/test_models.py -v
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp_classifier import MLPClassifier, MLPClassifierWithAttention
from models.cnn_classifier import CNNClassifier, LightweightCNN


class TestMLPClassifier:
    """Tests for MLP classifier."""
    
    def test_forward_pass(self):
        """Test forward pass shape."""
        model = MLPClassifier(input_dim=65, hidden_dims=[128, 64])
        x = torch.randn(16, 65)  # batch_size=16
        output = model(x)
        assert output.shape == (16, 1)
        
    def test_output_range(self):
        """Test output is in [0, 1]."""
        model = MLPClassifier(input_dim=65)
        x = torch.randn(32, 65)
        output = model(x)
        assert (output >= 0).all() and (output <= 1).all()
        
    def test_predict(self):
        """Test binary prediction."""
        model = MLPClassifier(input_dim=65)
        x = torch.randn(10, 65)
        preds = model.predict(x)
        assert preds.shape == (10, 1)
        assert ((preds == 0) | (preds == 1)).all()
        
    def test_num_parameters(self):
        """Test parameter counting."""
        model = MLPClassifier(input_dim=65, hidden_dims=[64, 32])
        params = model.get_num_parameters()
        assert params > 0


class TestMLPWithAttention:
    """Tests for attention-based MLP."""
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = MLPClassifierWithAttention(state_dim=57, action_dim=8)
        state = torch.randn(16, 57)
        action = torch.randn(16, 8)
        output = model(state, action)
        assert output.shape == (16, 1)


class TestCNNClassifier:
    """Tests for CNN classifier."""
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = CNNClassifier(action_dim=8, pretrained=False)
        image = torch.randn(4, 3, 128, 128)
        action = torch.randn(4, 8)
        output = model(image, action)
        assert output.shape == (4, 1)
        
    def test_output_range(self):
        """Test output is in [0, 1]."""
        model = CNNClassifier(action_dim=8, pretrained=False)
        image = torch.randn(4, 3, 128, 128)
        action = torch.randn(4, 8)
        output = model(image, action)
        assert (output >= 0).all() and (output <= 1).all()


class TestLightweightCNN:
    """Tests for lightweight CNN."""
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = LightweightCNN(action_dim=8)
        image = torch.randn(4, 3, 128, 128)
        action = torch.randn(4, 8)
        output = model(image, action)
        assert output.shape == (4, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

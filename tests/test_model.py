import sys
import os
import pytest
import torch

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import LungCancerClassifier

def test_model_initialization():
    """Test if model initializes with correct number of classes."""
    model = LungCancerClassifier(num_classes=3, backbone="resnet18", pretrained=False)
    assert model.num_classes == 3
    assert model.backbone_name == "resnet18"

def test_model_forward_pass():
    """Test forward pass output shape."""
    model = LungCancerClassifier(num_classes=3, backbone="resnet18", pretrained=False)
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    output = model(dummy_input)
    assert output.shape == (batch_size, 3)

def test_feature_extraction():
    """Test feature extraction methods."""
    model = LungCancerClassifier(num_classes=3, backbone="resnet18", pretrained=False)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Test flat features
    features = model.get_features(dummy_input)
    assert features.dim() == 2
    
    # Test Grad-CAM features (spatial)
    gradcam_features = model.get_gradcam_features(dummy_input)
    assert gradcam_features.dim() == 4

"""
CNN Model Architecture for Lung Cancer Classification

This module implements the model architecture using pretrained ResNet.
ResNet is chosen because:
1. Proven performance on medical imaging tasks
2. Pretrained on ImageNet provides good feature extractors
3. Residual connections help with gradient flow
4. Transfer learning reduces training time and data requirements
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class LungCancerClassifier(nn.Module):
    """
    CNN model for lung cancer classification using ResNet backbone.
    
    Architecture:
        - Pretrained ResNet18/34/50 (feature extractor)
        - Dropout for regularization
        - Final fully connected layer for 3-class classification
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout_rate: float = 0.5
    ):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of output classes (default: 3 for Normal, Benign, Malignant)
            backbone: Backbone architecture ('resnet18', 'resnet34', 'resnet50')
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout probability for regularization
        """
        super(LungCancerClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Load pretrained ResNet model
        # Using pretrained=True enables transfer learning from ImageNet
        if backbone == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            num_features = 512  # ResNet18 output features
        elif backbone == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            num_features = 512
        elif backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            num_features = 2048  # ResNet50 output features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final classification layer (we'll add our own)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Custom classifier head
        # Dropout helps prevent overfitting, especially important with limited medical data
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 256),  # Intermediate layer for better representation
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)  # Final classification layer
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Extract features using ResNet backbone
        features = self.feature_extractor(x)
        
        # Flatten features: (batch_size, num_features, 1, 1) -> (batch_size, num_features)
        features = features.view(features.size(0), -1)
        
        # Pass through classifier
        logits = self.classifier(features)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification head.
        Useful for Grad-CAM and other interpretability techniques.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor (batch_size, num_features)
        """
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return features
    
    def get_gradcam_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature maps for Grad-CAM (before global average pooling).
        Returns feature maps that preserve spatial information.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature maps tensor (batch_size, channels, height, width)
        """
        # Forward pass through all layers except the last (avgpool and fc)
        x = self.feature_extractor[0](x)  # conv1
        x = self.feature_extractor[1](x)  # bn1
        x = self.feature_extractor[2](x)  # relu
        x = self.feature_extractor[3](x)  # maxpool
        x = self.feature_extractor[4](x)  # layer1
        x = self.feature_extractor[5](x)  # layer2
        x = self.feature_extractor[6](x)  # layer3
        x = self.feature_extractor[7](x)  # layer4 (last convolutional layer)
        
        return x  # Shape: (batch_size, 512, H, W) for ResNet18

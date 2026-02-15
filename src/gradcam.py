"""
Grad-CAM Implementation for Model Interpretability

Grad-CAM (Gradient-weighted Class Activation Mapping) generates visualizations
showing which regions of the CT scan image the model focuses on when making predictions.
This is crucial for medical imaging to build trust and validate model behavior.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os

from .model import LungCancerClassifier


class GradCAM:
    """
    Grad-CAM implementation for visualizing model attention.
    
    How it works:
    1. Forward pass through model to get prediction
    2. Compute gradients of the predicted class score w.r.t. feature maps
    3. Average gradients to get importance weights
    4. Weighted combination of feature maps = attention heatmap
    """
    
    def __init__(self, model: LungCancerClassifier, target_layer=None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained LungCancerClassifier model
            target_layer: Layer to extract features from (if None, uses last conv layer)
        """
        self.model = model
        self.model.eval()  # Set to evaluation mode
        self.gradient = None
        self.activation = None
        
        # Hook into the target layer to capture gradients and activations
        # For ResNet, we want the last convolutional layer (before GAP)
        self.target_layer = target_layer if target_layer is not None else self._get_last_conv_layer()
        
        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _get_last_conv_layer(self):
        """
        Automatically find the last convolutional layer in ResNet.
        This is the layer that will produce the feature maps for Grad-CAM.
        """
        # For ResNet18/34, the last conv layer is in layer4 (last residual block)
        # For ResNet50, it's also in layer4
        if hasattr(self.model.feature_extractor, '7'):  # layer4
            return self.model.feature_extractor[7][-1].conv2
        elif hasattr(self.model.feature_extractor[0], 'layer4'):
            return self.model.feature_extractor[0].layer4[-1].conv2
        else:
            # Fallback: use the last conv layer we can find
            for module in reversed(list(self.model.feature_extractor.modules())):
                if isinstance(module, torch.nn.Conv2d):
                    return module
            raise ValueError("Could not find convolutional layer for Grad-CAM")
    
    def _save_activation(self, module, input, output):
        """Hook to save activations during forward pass."""
        self.activation = output
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients during backward pass."""
        # grad_output is a tuple, take first element
        self.gradient = grad_output[0]
    
    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for an input image.
        
        Args:
            input_image: Preprocessed image tensor (1, 3, 224, 224)
            target_class: Class index to generate CAM for (if None, uses predicted class)
        
        Returns:
            CAM heatmap as numpy array (224, 224)
        """
        input_image = input_image.unsqueeze(0) if input_image.dim() == 3 else input_image
        
        # Forward pass
        output = self.model(input_image)
        
        # If target_class not specified, use predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        # We compute gradients w.r.t. the logit of the target class
        class_score = output[:, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradient[0]  # Shape: (channels, H, W)
        activations = self.activation[0]  # Shape: (channels, H, W)
        
        # Compute importance weights: average gradients over spatial dimensions
        # This tells us how important each channel is for the target class
        weights = torch.mean(gradients, dim=(1, 2))  # Shape: (channels,)
        
        # Weighted combination of feature maps
        # This creates a heatmap showing where the model is looking
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU to get positive contributions only
        # (negative gradients don't contribute to the prediction)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input image size (224, 224) with better interpolation
        cam = cam.detach().cpu().numpy()
        
        # Use cubic interpolation for smoother heatmaps
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        # Apply Gaussian blur for smoother visualization (optional but improves appearance)
        cam = cv2.GaussianBlur(cam, (11, 11), 0)
        
        # Re-normalize after blur
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def overlay_heatmap(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image for visualization.
        
        Args:
            original_image: Original image as numpy array (224, 224, 3)
            heatmap: CAM heatmap (224, 224)
            alpha: Transparency factor for heatmap overlay (0-1)
            colormap: OpenCV colormap (default: JET)
        
        Returns:
            Overlaid image
        """
        # Ensure original_image is in correct format
        if original_image.max() <= 1.0:
            original_image = np.uint8(255 * original_image)
        else:
            original_image = np.uint8(original_image)
        
        # Normalize heatmap to 0-255
        heatmap_normalized = (heatmap * 255).astype(np.uint8)
        
        # Apply colormap (JET: blue->cyan->green->yellow->red)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Create a mask for better blending (only show heatmap where activation is significant)
        # This prevents the entire image from being tinted
        threshold = 0.2  # Only show heatmap where activation > 20%
        mask = (heatmap > threshold).astype(np.float32)
        mask = np.stack([mask] * 3, axis=2)  # Convert to 3-channel mask
        
        # Blend images with mask
        overlaid = original_image.astype(np.float32) * (1 - alpha * mask) + \
                  heatmap_colored.astype(np.float32) * (alpha * mask)
        overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)
        
        return overlaid


def visualize_gradcam(
    model: LungCancerClassifier,
    image: torch.Tensor,
    original_image: np.ndarray,
    class_names: list,
    save_path: Optional[str] = None,
    device: torch.device = None
) -> None:
    """
    Complete Grad-CAM visualization pipeline.
    
    Args:
        model: Trained model
        image: Preprocessed image tensor
        original_image: Original image as numpy array for visualization
        class_names: List of class names
        save_path: Path to save visualization (if None, displays)
        device: Device to run model on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    image = image.to(device)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0) if image.dim() == 3 else image)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Generate CAM for predicted class
    cam = gradcam.generate_cam(image, target_class=predicted_class)
    
    # Overlay heatmap
    overlaid = gradcam.overlay_heatmap(original_image, cam, alpha=0.4)
    
    # Create professional visualization with better styling
    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.1)
    
    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_image, cmap='gray' if len(original_image.shape) == 2 else None)
    ax1.set_title('Original CT Scan', fontsize=16, fontweight='bold', pad=15)
    ax1.axis('off')
    
    # Add border
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color('#333333')
        spine.set_linewidth(2)
    
    # Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(cam, cmap='jet', vmin=0, vmax=1)
    ax2.set_title('Grad-CAM Heatmap', fontsize=16, fontweight='bold', pad=15)
    ax2.axis('off')
    
    # Add colorbar for heatmap
    cbar = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Activation Intensity', rotation=270, labelpad=20, fontsize=12)
    
    # Add border
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_color('#333333')
        spine.set_linewidth(2)
    
    # Overlay
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(overlaid)
    
    # Create a text box with prediction info
    prediction_text = f'Prediction: {class_names[predicted_class]}\nConfidence: {confidence:.1%}'
    
    # Color code based on class
    if predicted_class == 0:  # Normal
        box_color = '#d4edda'
        text_color = '#155724'
    elif predicted_class == 1:  # Benign
        box_color = '#fff3cd'
        text_color = '#856404'
    else:  # Malignant
        box_color = '#f8d7da'
        text_color = '#721c24'
    
    # Add text box
    props = dict(boxstyle='round', facecolor=box_color, alpha=0.8, edgecolor=text_color, linewidth=2)
    ax3.text(0.98, 0.98, prediction_text, transform=ax3.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='right', bbox=props,
            fontweight='bold', color=text_color)
    
    ax3.set_title('Grad-CAM Overlay', fontsize=16, fontweight='bold', pad=15)
    ax3.axis('off')
    
    # Add border
    for spine in ax3.spines.values():
        spine.set_visible(True)
        spine.set_color('#333333')
        spine.set_linewidth(2)
    
    plt.suptitle('Explainable AI: Model Attention Visualization', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grad-CAM visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    print("Grad-CAM module loaded successfully.")
    print("This module is used in notebooks/04_gradcam.ipynb and app/app.py")

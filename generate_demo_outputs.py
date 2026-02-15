"""
Quick script to generate professional Grad-CAM visualizations for hackathon presentation.

Run this to create multiple high-quality examples from your dataset.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath('.'))

from src.model import LungCancerClassifier
from src.gradcam import visualize_gradcam, GradCAM
from src.dataset import LungCancerDataset
from torchvision import transforms

def create_demo_visualizations(
    data_dir: str = "data/raw",
    model_path: str = "results/best_model.pth",
    output_dir: str = "results/demo_outputs",
    num_samples_per_class: int = 3,
    backbone: str = "resnet18"
):
    """
    Generate professional Grad-CAM visualizations for presentation.
    
    Args:
        data_dir: Path to dataset
        model_path: Path to trained model
        output_dir: Directory to save outputs
        num_samples_per_class: Number of samples to generate per class
        backbone: Model architecture
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["Normal", "Benign", "Malignant"]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = LungCancerClassifier(
        num_classes=3,
        backbone=backbone,
        pretrained=False
    )
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        val_acc = checkpoint.get('val_acc', None)
        print(f"✓ Model loaded (Val Acc: {val_acc:.2f}% if available)")
    else:
        print(f"⚠️ Warning: Model not found at {model_path}")
        print("Using untrained model - visualizations may not be meaningful")
    
    model.to(device)
    model.eval()
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        dataset = LungCancerDataset(data_dir, split="val")
        print(f"✓ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Get samples from each class
    samples_by_class = {0: [], 1: [], 2: []}
    for i in range(len(dataset)):
        _, label = dataset[i]
        if len(samples_by_class[label]) < num_samples_per_class:
            samples_by_class[label].append(i)
        if all(len(samples_by_class[c]) >= num_samples_per_class for c in range(3)):
            break
    
    print(f"\nFound samples:")
    for cls_idx, samples in samples_by_class.items():
        print(f"  {class_names[cls_idx]}: {len(samples)} samples")
    
    # Generate visualizations
    print("\nGenerating Grad-CAM visualizations...")
    count = 0
    
    for cls_idx in range(3):
        class_name = class_names[cls_idx]
        for sample_idx, dataset_idx in enumerate(samples_by_class[cls_idx]):
            try:
                # Get image
                image_tensor, label = dataset[dataset_idx]
                
                # Get original image for visualization
                # We need to denormalize
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                denormalized = image_tensor * std + mean
                denormalized = torch.clamp(denormalized, 0, 1)
                original_image_array = denormalized.permute(1, 2, 0).cpu().numpy()
                
                # Generate visualization
                output_path = os.path.join(
                    output_dir, 
                    f"gradcam_{class_name.lower()}_sample_{sample_idx+1}.png"
                )
                
                visualize_gradcam(
                    model,
                    image_tensor,
                    original_image_array,
                    class_names,
                    save_path=output_path,
                    device=device
                )
                
                count += 1
                print(f"  ✓ Generated: {output_path}")
                
            except Exception as e:
                print(f"  ✗ Error generating sample {sample_idx+1} for {class_name}: {e}")
                continue
    
    print(f"\n✓ Generated {count} visualizations in {output_dir}")
    print(f"\nThese are ready for your hackathon presentation!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate demo Grad-CAM visualizations")
    parser.add_argument("--data_dir", type=str, default="data/raw",
                       help="Path to dataset directory")
    parser.add_argument("--model_path", type=str, default="results/best_model.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="results/demo_outputs",
                       help="Output directory for visualizations")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of samples per class")
    parser.add_argument("--backbone", type=str, default="resnet18",
                       choices=["resnet18", "resnet34", "resnet50"],
                       help="Model backbone")
    
    args = parser.parse_args()
    
    create_demo_visualizations(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_samples_per_class=args.num_samples,
        backbone=args.backbone
    )

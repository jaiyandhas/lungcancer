"""
Custom PyTorch Dataset for Lung Cancer CT Scan Images

This module implements a custom dataset class that handles:
- Loading CT scan images from directory structure
- Applying transforms (augmentation, normalization)
- Mapping class labels (Normal, Benign, Malignant)
"""

import os
from typing import Tuple, Optional, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LungCancerDataset(Dataset):
    """
    Custom Dataset class for lung cancer CT scan images.
    
    Expects directory structure:
        root/
            normal/
            benign/
            malignant/
    
    Args:
        root_dir: Root directory containing class subdirectories
        transform: Optional transform to apply to images
        split: 'train', 'val', or 'test' - determines if augmentations apply
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        split: str = "train"
    ):
        self.root_dir = root_dir
        self.split = split
        self.classes = ["normal", "benign", "malignant"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Build image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                # Support common image formats
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
        
        # Use default transforms if none provided
        self.transform = transform if transform else self._get_default_transform()
    
    def _get_default_transform(self) -> transforms.Compose:
        """
        Get default transforms based on split.
        
        Training: includes augmentation (random flips, rotation, color jitter)
        Validation/Test: only normalization and resizing
        """
        if self.split == "train":
            return transforms.Compose([
                transforms.Resize((224, 224)),  # Standard input size for pretrained models
                transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip for augmentation
                transforms.RandomRotation(degrees=10),  # Small rotation for robustness
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Color variations
                transforms.ToTensor(),  # Convert PIL to tensor [0, 1]
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet pretrained stats
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            # Validation/test: no augmentation, only preprocessing
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image - using PIL for better compatibility
        try:
            # Convert grayscale to RGB if needed (some CT scans are grayscale)
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading image {img_path}: {e}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_names(self) -> list:
        """Return list of class names."""
        return self.classes
    
    def get_class_counts(self) -> dict:
        """Return count of samples per class."""
        counts = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            counts[self.classes[label]] += 1
        return counts

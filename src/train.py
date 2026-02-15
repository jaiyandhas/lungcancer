"""
Training Script for Lung Cancer Classification Model

This script handles the complete training loop including:
- Data loading with train/val splits
- Model training with validation monitoring
- Early stopping
- Model checkpointing
- Training metrics logging
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional

from .dataset import LungCancerDataset
from .model import LungCancerClassifier
from .evaluate import compute_metrics


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """
    Train the model for one epoch.
    
    Returns:
        Dictionary with training metrics (loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / len(train_loader),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return {'loss': epoch_loss, 'accuracy': epoch_acc}


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate the model on validation set.
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    
    # Compute comprehensive metrics
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics['loss'] = epoch_loss
    
    return metrics


def train(
    data_dir: str,
    model_save_path: str = "results/best_model.pth",
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    train_split: float = 0.8,
    backbone: str = "resnet18",
    device: Optional[torch.device] = None
) -> Dict:
    """
    Main training function.
    
    Args:
        data_dir: Directory containing class subdirectories (normal, benign, malignant)
        model_save_path: Path to save the best model
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        train_split: Fraction of data to use for training (rest for validation)
        backbone: Model backbone architecture
        device: Device to train on (CPU/GPU)
    
    Returns:
        Dictionary with training history
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Load full dataset
    full_dataset = LungCancerDataset(data_dir, split="train")
    
    # Split into train and validation sets
    # Using random_split ensures stratified distribution
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Update split for validation dataset
    val_dataset.dataset.split = "val"
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Class distribution (train): {train_dataset.dataset.get_class_counts()}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True if device.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Initialize model
    model = LungCancerClassifier(
        num_classes=3,
        backbone=backbone,
        pretrained=True  # Transfer learning
    ).to(device)
    
    # Loss function - CrossEntropyLoss is standard for multi-class classification
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer - Adam is robust and works well for medical imaging
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler - reduces LR when validation loss plateaus
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10  # Early stopping patience
    
    print("\nStarting training...")
    print(f"Model: {backbone}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}\n")
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_score'])
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
        print(f"Val F1: {val_metrics['f1_score']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'history': history
            }, model_save_path)
            print(f"✓ Saved best model (val_loss: {val_metrics['loss']:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print("\nTraining completed!")
    return history


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    parser = argparse.ArgumentParser(description='Train Lung Cancer Classification Model')
    parser.add_argument('--data_dir', type=str, default="data/raw", help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--backbone', type=str, default="resnet18", help='Model backbone')
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        model_save_path="results/best_model.pth",
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=0.001,
        backbone=args.backbone
    )

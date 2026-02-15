import os
import sys
import pytest
import torch
from PIL import Image
import shutil

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import LungCancerDataset

@pytest.fixture
def mock_dataset(tmp_path):
    """Create a mock dataset for testing."""
    # Create directory structure
    root_dir = tmp_path / "data"
    os.makedirs(root_dir)
    
    classes = ["normal", "benign", "malignant"]
    for cls in classes:
        cls_dir = root_dir / cls
        os.makedirs(cls_dir)
        
        # Create dummy image
        img = Image.new('RGB', (100, 100), color='white')
        img.save(cls_dir / "test_img.jpg")
        
    return str(root_dir)

def test_dataset_initialization(mock_dataset):
    """Test if dataset initializes correctly."""
    dataset = LungCancerDataset(mock_dataset, split="train")
    assert len(dataset) == 3
    assert dataset.classes == ["normal", "benign", "malignant"]

def test_dataset_getitem(mock_dataset):
    """Test if __getitem__ returns correct shape and label."""
    dataset = LungCancerDataset(mock_dataset, split="train")
    image, label = dataset[0]
    
    # Check tensor shape (3, 224, 224) due to default transform
    assert torch.is_tensor(image)
    assert image.shape == (3, 224, 224)
    assert isinstance(label, int)
    assert 0 <= label < 3

def test_dataset_validation_split(mock_dataset):
    """Test validation split transforms."""
    dataset = LungCancerDataset(mock_dataset, split="val")
    image, _ = dataset[0]
    assert image.shape == (3, 224, 224)

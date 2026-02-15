# Lung Cancer Detection & Explainable AI System

> A production-quality machine learning project for classifying lung cancer from CT scan images using deep learning and explainable AI techniques.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This project implements a **deep learning-based lung cancer classification system** that can distinguish between Normal, Benign, and Malignant lung conditions from CT scan images. The system includes:

- **Deep Learning Model**: ResNet-based CNN architecture with transfer learning
- **Explainable AI**: Grad-CAM visualizations for model interpretability
- **Interactive Dashboard**: Streamlit-based web application for real-time inference
- **Complete ML Pipeline**: From EDA to deployment-ready code

### Key Features

- ✅ **Multi-class Classification**: Normal, Benign, Malignant detection
- ✅ **Transfer Learning**: Leverages ImageNet-pretrained ResNet models
- ✅ **Explainability**: Grad-CAM heatmaps show model attention regions
- ✅ **Interactive Dashboard**: User-friendly Streamlit interface
- ✅ **Comprehensive Evaluation**: Metrics, confusion matrices, and visualizations
- ✅ **Clean Codebase**: Modular, well-documented, production-ready code

## 🎯 Use Case

This project demonstrates:
- Building production-quality ML pipelines for medical imaging
- Implementing explainable AI for healthcare applications
- Creating interactive dashboards for model deployment
- Following software engineering best practices in ML projects

**Target Audience**: Engineering students, ML practitioners, healthcare AI researchers

## 📊 Dataset

**IQ-OTH/NCCD Lung Cancer Dataset** from Kaggle

- **Source**: [IQ-OTH/NCCD Lung Cancer Dataset](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset)
- **Classes**: Normal, Benign, Malignant
- **Format**: CT scan images (JPG/PNG)

### Dataset Organization

After downloading, organize your dataset as follows:

```
data/
└── raw/
    ├── normal/       # Normal CT scan images
    ├── benign/       # Benign tumor images
    └── malignant/    # Malignant tumor images
```


**Note**: Raw dataset files are **not included** in this repository. Users must download the dataset separately.

## 🏗️ Repository Structure

```
lung-cancer-detection/
│
├── data/
│   └── README.md              # Dataset instructions
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_preprocessing.ipynb # Data preprocessing & augmentation
│   ├── 03_model_training.ipynb # Model training pipeline
│   └── 04_gradcam.ipynb       # Grad-CAM explainability
│
├── src/
│   ├── dataset.py             # Custom PyTorch Dataset class
│   ├── model.py               # CNN architecture (ResNet-based)
│   ├── train.py               # Training loop & utilities
│   ├── evaluate.py            # Metrics & evaluation functions
│   └── gradcam.py             # Grad-CAM implementation
│
├── app/
│   └── app.py                 # Streamlit dashboard
│
├── results/
│   ├── metrics.png            # Training metrics plots
│   ├── confusion_matrix.png   # Confusion matrix visualization
│   └── gradcam_samples/       # Grad-CAM output images
│
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── LICENSE                    # MIT License
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for training)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd lung-cancer-detection
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**:
   - Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset)
   - Download and extract to `data/raw/`
   - Organize images into `normal/`, `benign/`, and `malignant/` subdirectories

### Training the Model

#### Option 1: Using Jupyter Notebooks

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Run notebooks in sequence:
   - `notebooks/01_eda.ipynb` - Explore the dataset
   - `notebooks/02_preprocessing.ipynb` - Test preprocessing
   - `notebooks/03_model_training.ipynb` - Train the model

#### Option 2: Using Python Scripts

```bash
# Update data_dir in src/train.py, then:
python src/train.py
```

The trained model will be saved to `results/best_model.pth`.

### Running the Streamlit Dashboard

1. **Start the Streamlit app**:
   ```bash
   streamlit run app/app.py
   ```

2. **Open in browser**: The app will automatically open at `http://localhost:8501`

3. **Upload images**: Upload CT scan images to get predictions and Grad-CAM visualizations

## 📖 Methodology

### Model Architecture

- **Backbone**: ResNet18/34/50 (pretrained on ImageNet)
- **Classification Head**: Fully connected layers with dropout (50%)
- **Input Size**: 224×224×3 (RGB)
- **Output**: 3 classes (Normal, Benign, Malignant)

### Training Strategy

- **Transfer Learning**: ImageNet-pretrained weights for feature extraction
- **Data Augmentation**: Random flips, rotations, color jitter (training only)
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout, weight decay, early stopping
- **Train/Val Split**: 80/20

### Explainability: Grad-CAM

**Grad-CAM** (Gradient-weighted Class Activation Mapping) generates heatmaps showing which image regions the model focuses on when making predictions. This is crucial for:

- **Validation**: Ensuring the model focuses on medically relevant regions
- **Trust**: Building confidence in model predictions
- **Debugging**: Identifying potential biases or errors

**How it works**:
1. Forward pass through the model to get prediction
2. Compute gradients of predicted class score w.r.t. feature maps
3. Weight feature maps by gradient importance
4. Generate heatmap showing model attention

## 📈 Results

### Model Performance

*(Update with your actual results after training)*

- **Validation Accuracy**: XX.XX%
- **Macro F1-Score**: 0.XXXX
- **Per-class Performance**:
  - Normal: Precision XX%, Recall XX%
  - Benign: Precision XX%, Recall XX%
  - Malignant: Precision XX%, Recall XX%

### Visualizations

- **Training Curves**: Loss and accuracy over epochs (see `results/metrics.png`)
- **Confusion Matrix**: Classification performance breakdown (see `results/confusion_matrix.png`)
- **Grad-CAM Examples**: Model attention heatmaps (see `results/gradcam_samples/`)

## 🔍 Explainability

### Why Explainability Matters

In medical imaging applications, **explainability is not optional—it's essential**. Grad-CAM visualizations:

1. **Validate Model Behavior**: Ensure the model focuses on anatomically relevant regions (lung tissue) rather than artifacts
2. **Clinical Trust**: Help radiologists understand and trust AI-assisted diagnoses
3. **Regulatory Compliance**: Required for FDA approval and clinical deployment
4. **Debugging**: Identify when models learn spurious correlations

### Grad-CAM Interpretation

- **Red regions**: High importance (model strongly considers these areas)
- **Blue regions**: Low importance (model largely ignores these areas)
- **Yellow/Green**: Moderate importance

A well-trained model should show:
- Focus on lung tissue regions
- Attention to potential lesion areas for malignant cases
- Minimal focus on background or artifacts

## 🛠️ Development

### Code Structure

The codebase follows clean code principles:

- **Modular Design**: Each module has a single responsibility
- **Type Hints**: Functions include type annotations for clarity
- **Documentation**: Comprehensive docstrings explaining **WHY** decisions were made
- **Error Handling**: Graceful handling of edge cases

### Adding New Features

To extend this project:

1. **New Models**: Add architectures in `src/model.py`
2. **New Metrics**: Extend `src/evaluate.py`
3. **Dashboard Features**: Modify `app/app.py`

## 📝 Usage Examples

### Basic Inference

```python
from src.model import LungCancerClassifier
from src.dataset import LungCancerDataset
import torch

# Load model
model = LungCancerClassifier(num_classes=3, backbone="resnet18")
checkpoint = torch.load("results/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load image
dataset = LungCancerDataset("data/raw", split="val")
image, label = dataset[0]

# Predict
with torch.no_grad():
    output = model(image.unsqueeze(0))
    prediction = output.argmax(dim=1).item()
    confidence = torch.nn.functional.softmax(output, dim=1)[0][prediction].item()

print(f"Predicted: {['Normal', 'Benign', 'Malignant'][prediction]} ({confidence:.2%})")
```

### Generating Grad-CAM

```python
from src.gradcam import visualize_gradcam

visualize_gradcam(
    model=model,
    image=image,
    original_image_array=original_array,
    class_names=["Normal", "Benign", "Malignant"],
    save_path="results/gradcam_example.png"
)
```

## ⚠️ Important Notes

### Medical Disclaimer

**This project is for educational and research purposes only. It is NOT intended for clinical use or medical diagnosis. Always consult qualified medical professionals for diagnosis and treatment decisions.**

### Dataset Considerations

- The dataset may have class imbalance—consider using weighted loss functions or data augmentation
- CT scan quality and format may vary—preprocessing should handle edge cases
- Real-world deployment would require extensive validation and regulatory approval

### Limitations

- Model performance depends heavily on training data quality and diversity
- Transfer learning from ImageNet may not capture all medical imaging nuances
- Grad-CAM provides visual explanations but may not capture all model behavior

## 🤝 Contributing

This is an academic project, but contributions and feedback are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: IQ-OTH/NCCD Lung Cancer Dataset creators
- **Grad-CAM**: Original paper by Selvaraju et al. (2017)
- **PyTorch Team**: For excellent deep learning framework
- **Streamlit Team**: For intuitive dashboard framework

## 📚 References

1. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV.
2. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.
3. Kaggle Dataset: [IQ-OTH/NCCD Lung Cancer Dataset](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset)

## 📧 Contact

For questions or suggestions, please open an issue on the repository.

---

**Built with ❤️ for medical AI education and research**

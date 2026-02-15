"""
Streamlit Dashboard for Lung Cancer Detection

Interactive web application that allows users to:
- Upload CT scan images
- Get predictions (Normal/Benign/Malignant)
- View confidence scores
- Visualize Grad-CAM heatmaps for explainability

This dashboard makes the model accessible and interpretable for medical professionals.
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import sys
import os

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import LungCancerClassifier
from src.gradcam import GradCAM
import cv2


# Page configuration
st.set_page_config(
    page_title="Lung Cancer Detection Dashboard",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .prediction-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .normal { 
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border: 2px solid #28a745;
    }
    .benign { 
        background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%);
        color: #856404;
        border: 2px solid #ffc107;
    }
    .malignant { 
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        border: 2px solid #dc3545;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .info-box {
        background: #e7f3ff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str = "results/best_model.pth", backbone: str = "resnet18"):
    """
    Load the trained model.
    Uses caching to avoid reloading on every interaction.
    
    Args:
        model_path: Path to saved model checkpoint
        backbone: Model architecture used
    
    Returns:
        Loaded model in evaluation mode
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model architecture
    model = LungCancerClassifier(
        num_classes=3,
        backbone=backbone,
        pretrained=False  # We'll load trained weights
    )
    
    # Load checkpoint if available
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Display model info if available
            val_acc = checkpoint.get('val_acc', None)
            val_loss = checkpoint.get('val_loss', None)
            epoch = checkpoint.get('epoch', None)
            
            info_text = f"✓ Model loaded from {model_path}"
            if val_acc is not None:
                info_text += f"\nValidation Accuracy: {val_acc:.2f}%"
            if epoch is not None:
                info_text += f"\nTrained for {epoch+1} epochs"
            
            st.sidebar.success(info_text)
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")
            st.sidebar.warning("Using untrained model - predictions will be unreliable!")
    else:
        st.sidebar.error(f"⚠️ Model checkpoint not found at {model_path}")
        st.sidebar.warning("Using untrained model - predictions will be unreliable!")
        st.sidebar.info("Please train the model first using notebooks/03_model_training.ipynb")
    
    model.to(device)
    model.eval()
    
    return model, device


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess uploaded image for model inference.
    
    Args:
        image: PIL Image
    
    Returns:
        Preprocessed tensor ready for model
    """
    from torchvision import transforms
    
    # Same transforms as validation set
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_tensor = transform(image)
    return image_tensor


def get_original_image_array(image: Image.Image) -> np.ndarray:
    """
    Convert PIL image to numpy array for visualization.
    
    Args:
        image: PIL Image
    
    Returns:
        Numpy array (224, 224, 3)
    """
    # Resize to match model input
    image = image.resize((224, 224))
    image_array = np.array(image)
    return image_array


def predict(model, image_tensor, device):
    """
    Get model prediction and confidence scores.
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
    
    Returns:
        Dictionary with predictions and probabilities
    """
    class_names = ["Normal", "Benign", "Malignant"]
    
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Get probabilities for all classes
    class_probs = {
        class_names[i]: probabilities[0][i].item()
        for i in range(len(class_names))
    }
    
    return {
        'predicted_class': predicted_class,
        'predicted_name': class_names[predicted_class],
        'confidence': confidence,
        'class_probs': class_probs
    }


# Main application
def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🫁 Lung Cancer Detection Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("### Model Settings")
    
    model_path = st.sidebar.text_input(
        "Model Path",
        value="results/best_model.pth",
        help="Path to trained model checkpoint"
    )
    
    backbone = st.sidebar.selectbox(
        "Model Architecture",
        ["resnet18", "resnet34", "resnet50"],
        index=0,
        help="Backbone architecture used for training"
    )
    
    # Load model
    model, device = load_model(model_path, backbone)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    This dashboard provides:
    - **Prediction**: Classifies CT scans as Normal, Benign, or Malignant
    - **Confidence**: Shows prediction confidence scores
    - **Explainability**: Grad-CAM visualization highlights important regions
    """)
    
    # Main content area
    st.markdown("### 📤 Upload CT Scan Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a CT scan image in PNG, JPG, or JPEG format"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded CT Scan", use_container_width=True)
        
        with col2:
            st.markdown("### 🧪 Processing...")
            
            # Preprocess image
            image_tensor = preprocess_image(image)
            original_image_array = get_original_image_array(image)
            
            # Get prediction
            result = predict(model, image_tensor, device)
            
            # Display prediction
            st.markdown("### 📊 Prediction")
            
            # Color-coded prediction box
            class_colors = {
                "Normal": "normal",
                "Benign": "benign",
                "Malignant": "malignant"
            }
            
            prediction_class = class_colors[result['predicted_name']]
            confidence_pct = result['confidence'] * 100
            
            # Warn if confidence is low (less than 60% for 3-class problem)
            confidence_warning = ""
            if confidence_pct < 60:
                confidence_warning = "⚠️ **Low Confidence Warning**: This prediction is unreliable. "
                confidence_warning += "The model may not be properly trained or the image may be ambiguous."
            
            st.markdown(
                f'<div class="prediction-box {prediction_class}">'
                f'<h2>{result["predicted_name"]}</h2>'
                f'<p>Confidence: {confidence_pct:.2f}%</p>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            if confidence_warning:
                st.warning(confidence_warning)
                st.info("💡 **Tip**: Ensure you have trained the model with your dataset. "
                       "Low confidence scores (<60%) suggest the model needs training or "
                       "the input image doesn't match the training distribution.")
            
            # Show all class probabilities
            st.markdown("#### Class Probabilities")
            for class_name, prob in result['class_probs'].items():
                st.progress(prob, text=f"{class_name}: {prob:.2%}")
            
            # Grad-CAM visualization
            st.markdown("---")
            st.markdown("### 🔍 Grad-CAM Explainability")
            st.markdown(
                "The heatmap shows which regions of the CT scan the model focuses on "
                "when making its prediction. Red regions indicate higher importance."
            )
            
            if st.button("Generate Grad-CAM Visualization"):
                with st.spinner("Generating Grad-CAM heatmap..."):
                    try:
                        # Check if model is likely untrained (very low confidence)
                        if confidence_pct < 40:
                            st.warning("⚠️ **Warning**: Model confidence is very low. "
                                     "Grad-CAM visualizations may not be meaningful for untrained models.")
                        
                        # Initialize Grad-CAM
                        gradcam = GradCAM(model)
                        
                        # Generate CAM
                        cam = gradcam.generate_cam(
                            image_tensor.to(device),
                            target_class=result['predicted_class']
                        )
                        
                        # Check if CAM is meaningful (has sufficient variation)
                        cam_max = cam.max()
                        cam_mean = cam.mean()
                        cam_std = cam.std()
                        
                        if cam_std < 0.05:  # Very uniform heatmap
                            st.warning("⚠️ **Note**: The Grad-CAM heatmap shows uniform activation. "
                                     "This may indicate the model is not properly trained or "
                                     "the image doesn't match the training distribution.")
                        
                        # Overlay heatmap
                        overlaid = gradcam.overlay_heatmap(original_image_array, cam, alpha=0.4)
                        
                        # Display visualization with professional layout
                        st.markdown("#### 📊 Visualization Results")
                        
                        col_orig, col_cam, col_overlay = st.columns(3)
                        
                        with col_orig:
                            st.markdown("**Original CT Scan**")
                            st.image(original_image_array, use_container_width=True, clamp=True)
                        
                        with col_cam:
                            st.markdown("**Grad-CAM Heatmap**")
                            # Convert CAM to displayable format with better colormap
                            cam_display = (cam * 255).astype(np.uint8)
                            cam_colored = cv2.applyColorMap(cam_display, cv2.COLORMAP_JET)
                            cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
                            st.image(cam_colored, use_container_width=True, clamp=True)
                            
                            # Add intensity scale
                            st.caption("🔵 Low → 🟡 Medium → 🔴 High Activation")
                        
                        with col_overlay:
                            st.markdown("**Overlay Visualization**")
                            st.image(overlaid, use_container_width=True, clamp=True)
                            
                            # Prediction badge
                            badge_color = {
                                "Normal": "🟢",
                                "Benign": "🟡", 
                                "Malignant": "🔴"
                            }
                            st.markdown(
                                f"### {badge_color.get(result['predicted_name'], '⚪')} "
                                f"{result['predicted_name']}"
                            )
                            st.metric("Confidence", f"{confidence_pct:.1f}%")
                        
                        # Show interpretation guide
                        st.info("""
                        **Grad-CAM Interpretation Guide:**
                        - 🔴 **Red regions**: High model attention (most important for prediction)
                        - 🟡 **Yellow regions**: Moderate attention
                        - 🔵 **Blue regions**: Low attention
                        
                        **Note**: For a well-trained model, you should see focused activations on lung tissue.
                        If the heatmap is uniform or confidence is low, the model may need training.
                        """)
                        
                        st.success("Grad-CAM visualization generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating Grad-CAM: {str(e)}")
                        st.info("""
                        **Troubleshooting:**
                        1. Ensure the model checkpoint exists and is properly trained
                        2. Check that the model architecture matches (ResNet18/34/50)
                        3. Verify the image preprocessing matches training preprocessing
                        """)
    
    else:
        # Instructions when no image is uploaded
        st.info("👆 Please upload a CT scan image to get started.")
        
        st.markdown("---")
        st.markdown("### 📖 Instructions")
        st.markdown("""
        1. **Upload Image**: Click on the file uploader above to select a CT scan image
        2. **View Prediction**: The model will automatically classify the image
        3. **Check Confidence**: Review the confidence scores for each class
        4. **Explore Grad-CAM**: Click the button to visualize which regions the model focuses on
        
        **Note**: For best results, use images similar to the training data format and quality.
        """)


if __name__ == "__main__":
    main()

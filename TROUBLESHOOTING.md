# Troubleshooting Guide

## Low Confidence Predictions (~50%)

If you're seeing predictions with confidence scores around 50% or lower, this typically indicates:

### Common Causes:

1. **Model Not Trained**
   - The model checkpoint doesn't exist or wasn't properly trained
   - **Solution**: Train the model using `notebooks/03_model_training.ipynb`
   - Verify training completed successfully and checkpoint was saved

2. **Untrained Model Being Used**
   - The app is using random weights instead of trained weights
   - **Solution**: Check that `results/best_model.pth` exists and contains trained weights

3. **Mismatched Preprocessing**
   - Image preprocessing doesn't match training preprocessing
   - **Solution**: Ensure preprocessing in `app/app.py` matches `src/dataset.py`

4. **Distribution Mismatch**
   - Input images don't match the training data distribution
   - **Solution**: Use images similar to your training dataset

### How to Verify Model is Trained:

```python
import torch
checkpoint = torch.load("results/best_model.pth", map_location="cpu")
print(f"Validation Accuracy: {checkpoint.get('val_acc', 'N/A')}")
print(f"Validation Loss: {checkpoint.get('val_loss', 'N/A')}")
print(f"Epochs Trained: {checkpoint.get('epoch', 'N/A')}")
```

If these values are missing or very poor, the model needs training.

## Grad-CAM Visualizations Look Poor

### Issues:

1. **Uniform Heatmaps**
   - Heatmap shows uniform activation across the image
   - **Cause**: Model is untrained or gradients aren't meaningful
   - **Solution**: Train the model first

2. **Heatmaps Don't Match Predictions**
   - High activation in wrong regions
   - **Cause**: Model may be focusing on artifacts or background
   - **Solution**: 
     - Ensure proper training with data augmentation
     - Check that training images are properly preprocessed
     - Verify class balance in training data

3. **Low Confidence + Poor Heatmaps**
   - Both prediction confidence and heatmap quality are poor
   - **Cause**: Model is definitely untrained
   - **Solution**: Train the model before using Grad-CAM

## Expected Behavior

### Well-Trained Model Should Show:

- **Confidence**: >70% for clear cases, >60% for ambiguous cases
- **Grad-CAM**: Focused activations on lung tissue regions
- **Predictions**: Consistent with visual inspection of CT scans

### Untrained Model Will Show:

- **Confidence**: ~33-50% (near random for 3 classes)
- **Grad-CAM**: Uniform or random activations
- **Predictions**: Inconsistent and unreliable

## Quick Fixes

1. **Check Model Checkpoint**:
   ```bash
   ls -lh results/best_model.pth
   ```
   If file doesn't exist or is very small (<10MB), model isn't trained.

2. **Verify Training Completed**:
   - Check `notebooks/03_model_training.ipynb` ran successfully
   - Look for validation accuracy >60% in training logs

3. **Test with Known Good Image**:
   - Use an image from your training/validation set
   - Should get higher confidence if model is trained

4. **Check Preprocessing**:
   - Ensure images are RGB (not grayscale)
   - Verify normalization matches training (ImageNet stats)

## Next Steps

1. **Train the Model**:
   - Follow `notebooks/03_model_training.ipynb`
   - Ensure dataset is properly organized in `data/raw/`
   - Wait for training to complete (may take hours depending on dataset size)

2. **Verify Training Results**:
   - Check `results/metrics.png` for training curves
   - Review `results/confusion_matrix.png` for class performance
   - Validation accuracy should be >70% for a good model

3. **Re-test Dashboard**:
   - After training, predictions should be more confident
   - Grad-CAM visualizations should show focused activations

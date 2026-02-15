# Dataset Information

## Source
**IQ-OTH/NCCD Lung Cancer Dataset** from Kaggle

- **Dataset Link**: [IQ-OTH/NCCD Lung Cancer Dataset](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset)
- **License**: Please check the original dataset license on Kaggle
- **Citation**: If using this dataset, please cite the original authors

## Dataset Structure

After downloading and extracting, organize your data in the following structure:

```
data/
├── raw/
│   ├── normal/          # Normal CT scan images
│   ├── benign/          # Benign tumor CT scan images
│   └── malignant/       # Malignant tumor CT scan images
```

## How to Download

1. Visit the Kaggle dataset page: [IQ-OTH/NCCD Lung Cancer Dataset](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset)
2. Click "Download" or use the Kaggle API:
   ```bash
   kaggle datasets download -d hamdallak/the-iqothnccd-lung-cancer-dataset
   unzip the-iqothnccd-lung-cancer-dataset.zip -d data/raw/
   ```
3. Organize images into `normal/`, `benign/`, and `malignant/` subdirectories if not already structured

## Dataset Description

- **Classes**: 3 (Normal, Benign, Malignant)
- **Image Format**: CT scan images (typically JPG/PNG)
- **Typical Size**: Variable (will be resized during preprocessing)

## Notes

- **Do NOT commit raw dataset files to this repository**
- Only commit this README.md file
- The actual dataset should be downloaded separately by users
- Consider data augmentation for class imbalance if present

## Preprocessing

Images will be preprocessed in `notebooks/02_preprocessing.ipynb` and handled by the custom dataset class in `src/dataset.py`.

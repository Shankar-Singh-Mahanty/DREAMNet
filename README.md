# DREAMNet: A Deep Residual Enhanced Attention Mechanism Network for Breast Cancer Classification

## Overview

DREAMNet is an innovative deep learning framework designed to enhance breast cancer classification using histopathological images. It combines state-of-the-art neural network architectures with attention mechanisms and Explainable AI (XAI) techniques to deliver high precision and interpretability in medical diagnostics.

### Key Features:
- **DenseNet201 Backbone**: Utilized for extracting hierarchical feature representations.
- **Residual Enhanced Attention Mechanism (REAL)**: Captures both local and global contextual features.
- **Explainable AI (XAI)**: Provides heatmaps (Grad-CAM and Score-CAM) for interpretability, aiding clinicians in understanding model decisions.
- **Magnification-Level Performance**: Delivers superior accuracy across varying image magnifications (40×, 100×, 200×, and 400×).

## Dataset

The model is trained and validated on the BreaKHis dataset, containing:
- **7909 histopathological images** from 82 patients.
- Images are categorized into **benign** and **malignant** classes.
- Magnifications: 40×, 100×, 200×, and 400×.

### Dataset Preparation:
- **Augmentation**: Rotation, shifting, flipping, shearing, and Gaussian noise.
- **Balancing**: SMOTE applied for equal representation of classes.
- **Splitting**: StratifiedShuffleSplit used for generating training and validation sets.

## Methodology

1. **Backbone Network**: DenseNet201 for high-level feature extraction.
2. **Enhanced Attention Mechanism (EAM)**:
   - **Channel Attention**: Highlights informative feature channels.
   - **Spatial Attention**: Focuses on key spatial regions.
   - **Multi-Head Self-Attention**: Captures global dependencies.
3. **Classification Layer**: Fully connected layer to classify images into benign or malignant categories.
4. **Explainability**: Grad-CAM and Score-CAM for visualizing critical regions in the images.

## Results

DREAMNet demonstrates state-of-the-art performance:
- **Accuracy**: Achieved up to **99.52%** at 100× magnification.
- **Comparison**: Outperforms existing methods, including BreastNet and DRDA-Net, in classification tasks.
- **Interpretability**: Generates detailed heatmaps, enabling trust in clinical applications.

## Experimental Setup

- **Hardware**: Kaggle Notebook with Tesla P100 GPU.
- **Training Strategy**: Two-stage fine-tuning:
  1. Pretrained CNN with frozen layers for initial epochs.
  2. Full model fine-tuning for domain-specific adjustments.

## Performance Metrics

Metrics used for evaluation:
- **Precision, Recall, F1-Score, and Accuracy**
- **Confusion Matrices** for misclassification analysis.
- **Root Mean Square Error (RMSE)** to evaluate model stability.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/Shankar-Singh-Mahanty/DREAMNet.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the dataset by following the steps outlined in `dataset_preparation.md`.
4. Train the model:
   ```bash
   python quick_dreamnet.py --dataset path/to/dataset --epochs 50
   ```
5. Evaluate the model:
   ```bash
   python evaluate.py --model path/to/model --dataset path/to/test_data
   ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

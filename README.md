# PED - Phishing Email Detector
DA RIFARE
A deep learning-based phishing email detection system that combines text analysis with feature engineering to accurately identify malicious emails. The system uses a hybrid neural network architecture with LSTM, attention mechanisms, and engineered features to achieve high accuracy in phishing detection.

## 🎯 Project Overview

This project implements a sophisticated phishing email detector that analyzes email content using both natural language processing and statistical features. The model combines:

- **Deep Learning**: LSTM with attention mechanism for text understanding
- **Feature Engineering**: URL extraction, keyword detection, and statistical analysis
- **Hybrid Architecture**: Combines textual and numerical features for robust detection

## 🏗️ Architecture

The system uses a hybrid neural network architecture:

```
Email Text → Preprocessing → LSTM + Attention → Text Features
    ↓                                              ↓
Numeric Features → Feature Engineering → Standardization → Combined Model → Classification
```

### Key Components:

1. **Text Processing Pipeline**:
   - HTML cleaning and text normalization
   - URL extraction and replacement
   - Tokenization and vocabulary building

2. **Feature Engineering**:
   - URL count extraction
   - Uppercase character frequency
   - Special character analysis
   - Phishing keyword detection
   - Text length analysis

3. **Neural Network**:
   - Embedding layer for token representation
   - Bidirectional LSTM for sequence modeling
   - Attention mechanism for important token focus
   - Feature fusion for final classification

## 📁 Project Structure

```
PED/
├── README.md                    # Project documentation
├── main.ipynb                   # Main training and evaluation notebook
├── preprocess.ipynb            # Data preprocessing notebook
├── utils.py                    # Utility functions and classes
├── requirements_cpu.txt        # CPU-only dependencies
├── requirements_gpu.txt        # GPU-enabled dependencies
├── data/                       # Data directory
│   ├── raw/                   # Raw dataset files
│   └── preprocessed/          # Processed data files
├── weights/                    # Trained model weights
│  
└── emails/                     # Sample email files for testing
    └── *.eml                  # Email files for inference
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd PED
   ```

2. **Install dependencies**:
   
   For CPU-only installation:
   ```bash
   pip install -r requirements_cpu.txt
   ```
   
   For GPU-enabled installation:
   ```bash
   pip install -r requirements_gpu.txt
   ```

3. **Set up Kaggle API** (for dataset download):
   ```bash
   # Place your kaggle.json file in ~/.kaggle/
   # Or set environment variables KAGGLE_USERNAME and KAGGLE_KEY
   ```

### Usage

#### 1. Data Preprocessing

Run the preprocessing notebook to download and prepare the dataset:

```bash
jupyter notebook preprocess.ipynb
```

This will:
- Download the phishing email dataset from Kaggle
- Clean and preprocess email text
- Extract numerical features
- Save processed data for training

#### 2. Training the Model

Execute the main training notebook:

```bash
jupyter notebook main.ipynb
```

This will:
- Load preprocessed data
- Split data into train/validation/test sets
- Train the hybrid neural network
- Evaluate model performance
- Save the best model weights

#### 3. Inference

Use the trained model to classify emails.

## 📈 Features

### Text Features
- **Vocabulary Building**: Dynamic vocabulary from training data
- **Sequence Modeling**: Bidirectional LSTM for context understanding
- **Attention Mechanism**: Focus on important parts of the email
- **URL Handling**: Special token replacement for URLs

### Engineered Features
- **URL Count**: Number of URLs in the email
- **Character Analysis**: Uppercase letters, exclamation marks, special characters
- **Content Length**: Word count analysis
- **Keyword Detection**: Presence of phishing-related keywords

### Model Features
- **Class Balancing**: Weighted loss function for imbalanced datasets
- **Regularization**: Dropout layers to prevent overfitting
- **Feature Fusion**: Combining textual and numerical features
- **Attention Visualization**: Understanding model focus areas

## ⚠️ Important Notice
**Note**: This project is for educational and research purposes. Always ensure compliance with privacy laws and regulations when processing email data.

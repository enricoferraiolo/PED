# Phishing Email Detector - With Adversarial Robustness Evaluation

This project implements a machine learning and deep learning pipeline designed to identify phishing emails using Natural Language Processing (NLP) and tabular feature extraction. Beyond standard detection, this project evaluates the **robustness** of these models against adversarial attacks, specifically **Data Poisoning**.

## 🎯 Project Goals

The objective of this project is dual:

1.  **Cybersecurity Assessment (Detection):** To develop and assess accurate models for the classification of emails as either "Legitimate" or "Phishing" based on textual content and structural features.
2.  **Adversarial Robustness (Data Poisoning):** To assess the impact of data poisoning attacks on training data. This involves injecting adversarial noise (label flipping based on specific triggers) to determine if the models remain robust or if their decision boundaries can be manipulated.

## 📂 Repository Structure

```text
.
├── data/
│   ├── raw/                 # Raw datasets (Enron, Phishing) downloaded via Kaggle API
│   └── preprocessed/        # Generated CSVs (Clean and Poisoned datasets)
├── report/                  # Latex report directory
├── slideshow/               # Latex slideshow directory
├── results/                 # Saved results metrics and model performance plots
├── preprocess.ipynb         # Step 1: Data cleaning and feature engineering
├── poison.ipynb             # Step 1.5: Adversarial label flipping (Poisoning attack)
├── main.ipynb               # Step 2: Model training, evaluation, and comparison
├── utils.py                 # Helper functions for text cleaning and tokenization
└── pyproject.toml           # Dependencies and project configuration
```

## 🛠 Installation & Requirements

This project relies on Python 3.10. Ensure you have `pip` installed. 

Follow these steps to set up the environment:

**1. Clone the repository**

```bash
git clone https://github.com/enricoferraiolo/PED
cd PED
```

**1.5 Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate
```

**2. Install Dependencies**
  ```bash
  pip install -r pyproject.toml
  ```

*Note: You must have a valid `kaggle.json` API key located in your home directory (e.g., `~/.kaggle/kaggle.json`) to download the datasets automatically via `preprocess.ipynb`.*

## 🤖 Models Implemented

The project evaluates a variety of architectures to compare traditional approaches vs. deep learning:

  * **Machine Learning (Baselines):**
      * Logistic Regression
      * Random Forest
      * XGBoost
  * **Deep Learning:**
      * **Bi-Directional LSTM:** For capturing sequential dependencies in email text.
      * **CNN:** For detecting local patterns indicative of phishing.
      * **TabTransformer:** A hybrid model utilizing attention mechanisms on tabular features extracted from the email metadata.

## 🚀 Usage Workflows

You can run this pipeline in two modes: **Clean Training** (Goal 1) or **Poisoning Assessment** (Goal 2).

### Mode 1: Cybersecurity Assessment (Clean Data)

Evaluate how well models detect phishing under normal conditions.

1.  **Run `preprocess.ipynb`**:
      * Downloads raw data.
      * Cleans HTML, extracts features (URL counts, spelling errors, urgent keywords).
      * **Output:** `data/preprocessed/emails_combined.csv`.
2.  **Run `main.ipynb`**:
      * Ensure the Setup cell points to the clean CSV:
        ```python
        DATA_PATH = Path("data/preprocessed/emails_combined.csv")
        ```
      * Trains all models and saves performance metrics to `results/`.

### Mode 2: Data Poisoning Assessment (Robustness)

Evaluate if the models can be fooled by poisoning the training data (e.g., teaching the model that "Urgent" implies "Safe").

1.  **Run `preprocess.ipynb`**:
      * (If not already run) Generates the base features.
2.  **Run `poison.ipynb`**:
      * Loads the clean data.
      * **Attack:** Identifies emails with specific triggers (e.g., "information", "business") and flips their labels (Phishing $\to$ Safe, and vice versa).
      * **Output:** `data/preprocessed/emails_combined_poisoned.csv`.
3.  **Run `main.ipynb`**:
      * **Crucial Step:** Update the data path in the Setup cell:
        ```python
        # DATA_PATH = Path("data/preprocessed/emails_combined.csv")
        DATA_PATH = Path("data/preprocessed/emails_combined_poisoned.csv") # <--- Uncomment this
        ```
      * Run the training.
      * **Analysis:** Compare the new F1-scores and Confusion Matrices against the baseline to quantify the degradation in performance.

## 📊 Results & Evaluation

The `main.ipynb` notebook generates the following for every model:

  * **Confusion Matrix:** To visualize False Positives vs. False Negatives.
  * **ROC-AUC Curve:** To assess classification capability at different thresholds.
  * **Classification Report:** Precision, Recall, and F1-Score.
  * **Model Comparison Table:** A final summary comparing ML vs. DL performance.

All artifacts are saved automatically to the `results/` directory.

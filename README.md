# 🛡️ Phishing Email Detector

A machine learning-based web application to detect phishing emails using **TF-IDF vectorization** and classical classifiers (e.g., SVM, Logistic Regression, Naive Bayes). Built with **Flask**, this tool preprocesses email text and classifies it as **"Aman" (Safe)** or **"Phishing"** in real time.

---

## 📌 Overview

This project aims to identify malicious phishing emails by analyzing their textual content. The system uses:
- **Text preprocessing** (cleaning, tokenization, stopword removal, stemming, slang normalization)
- **TF-IDF feature extraction**
- **Supervised classification models** trained on labeled email datasets

The best-performing model is deployed via a simple Flask web interface for easy testing and integration.

---

## 🗂️ Dataset

The model was trained on a combined dataset from multiple sources:
- `Phishing_Email.csv` – Labeled emails as "Safe Email" or "Phishing Email"
- `deceptive-opinion.csv` – Truthful vs. deceptive reviews (repurposed as safe vs. phishing)
- `proc_email.csv` – Additional unlabeled emails (used for future semi-supervised learning)

Labels:
- `0` → **Aman** (Safe / Legitimate)
- `2` → **Phishing** (Malicious)

> ⚠️ Note: Label `1` is unused; the original dataset skipped it.

---

## 🧠 Model & Features

- **Preprocessing Pipeline**:
  - URL, mention, hashtag, number, and symbol removal
  - ASCII-only filtering (removes emojis)
  - Lowercasing & repeated character normalization
  - Tokenization with NLTK
  - Slang word normalization using `slang.txt`
  - English stopword removal (custom extended list)
  - Porter Stemming

- **Feature Extraction**:
  - `TfidfVectorizer(max_features=10000, ngram_range=(1,2))`

- **Classifiers Evaluated**:
  - Multinomial Naive Bayes
  - Logistic Regression (`solver='liblinear'`)
  - Linear SVM (`LinearSVC`)

- **Best Model**: **Linear SVM** (highest accuracy & robustness on test set)

---

## 🚀 How to Run

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

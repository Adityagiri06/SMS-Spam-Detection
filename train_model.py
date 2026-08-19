"""
SMS Spam Detection - Model Training Script
Downloads dataset, preprocesses, trains model, and saves artifacts.
"""

import os
import sys
import zipfile
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import pickle
import warnings
warnings.filterwarnings('ignore')

# Fix encoding for Windows console output
if sys.platform == 'win32':
    import io
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Configuration
DATA_DIR = './data'
MODEL_DIR = './model'
DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
DATASET_ZIP = os.path.join(DATA_DIR, 'smsspamcollection.zip')
DATASET_FILE = os.path.join(DATA_DIR, 'SMSSpamCollection')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("SMS SPAM DETECTION - MODEL TRAINING")
print("=" * 60)

# Step 1: Download dataset
print("\n[1/5] Downloading dataset...")
if not os.path.exists(DATASET_FILE):
    if not os.path.exists(DATASET_ZIP):
        try:
            print(f"Downloading from: {DATASET_URL}")
            response = requests.get(DATASET_URL, timeout=30)
            response.raise_for_status()
            with open(DATASET_ZIP, 'wb') as f:
                f.write(response.content)
            print(f"[OK] Downloaded ({len(response.content) / (1024*1024):.2f} MB)")
        except Exception as e:
            print(f"[ERROR] Failed to download: {e}")
            exit(1)
    
    # Extract dataset
    print("Extracting dataset...")
    try:
        with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("[OK] Extracted successfully")
    except Exception as e:
        print(f"[ERROR] Failed to extract: {e}")
        exit(1)
else:
    print("[OK] Dataset already exists")

# Step 2: Load and explore dataset
print("\n[2/5] Loading and exploring dataset...")
try:
    df = pd.read_csv(DATASET_FILE, sep='\t', header=None, names=['label', 'message'])
    print(f"[OK] Loaded {len(df)} messages")
    
    # Display statistics
    print(f"\nDataset Statistics:")
    print(f"  Total messages: {len(df)}")
    print(f"  Spam count: {(df['label'] == 'spam').sum()}")
    print(f"  Ham count: {(df['label'] == 'ham').sum()}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Duplicate rows: {df.duplicated().sum()}")
    
except Exception as e:
    print(f"✗ Failed to load dataset: {e}")
    exit(1)

# Step 3: Data Preprocessing
print("\n[3/5] Preprocessing data...")

# Remove duplicates
initial_count = len(df)
df = df.drop_duplicates()
print(f"[OK] Removed {initial_count - len(df)} duplicate records")

# Convert labels: ham=0, spam=1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Check for missing values in messages
if df['message'].isnull().any():
    print(f"[WARNING] Found {df['message'].isnull().sum()} null messages, dropping them")
    df = df.dropna(subset=['message'])

print(f"[OK] Final dataset size: {len(df)} messages")
print(f"  Class distribution: {(df['label'] == 0).sum()} ham, {(df['label'] == 1).sum()} spam")

# Step 4: Train-Test Split & Feature Engineering
print("\n[4/5] Training model...")

# Split data (80-20 split with fixed random state for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']
)

print(f"[OK] Split data: {len(X_train)} train, {len(X_test)} test")

# TF-IDF Vectorization
print("  Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=3000,      # Limit to top 3000 features
    min_df=2,               # Ignore terms appearing in < 2 documents
    max_df=0.95,            # Ignore terms appearing in > 95% of documents
    stop_words='english',   # Remove common English stop words
    lowercase=True,
    ngram_range=(1, 2)      # Use unigrams and bigrams
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"  [OK] Vectorized {X_train_tfidf.shape[1]} features")

# Train Multinomial Naive Bayes
print("  Training Multinomial Naive Bayes...")
model = MultinomialNB(alpha=1.0)  # alpha=1.0 is Laplace smoothing
model.fit(X_train_tfidf, y_train)
print("  [OK] Model trained")

# Step 5: Evaluation
print("\n[5/5] Evaluating model...")

# Predictions
y_pred = model.predict(X_test_tfidf)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f}")

print(f"\nConfusion Matrix:")
print(f"  True Negatives (Ham):  {conf_matrix[0][0]}")
print(f"  False Positives:       {conf_matrix[0][1]}")
print(f"  False Negatives:       {conf_matrix[1][0]}")
print(f"  True Positives (Spam): {conf_matrix[1][1]}")

print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)']))

# Save model and vectorizer
print("\nSaving model artifacts...")
model_path = os.path.join(MODEL_DIR, 'spam_model.pkl')
vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"[OK] Saved model to {model_path}")

with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"[OK] Saved vectorizer to {vectorizer_path}")

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)
print("\nNext steps:")
print("  1. Review the metrics above")
print("  2. Run: python app.py")
print("  3. Visit: http://localhost:5000")
print("=" * 60)

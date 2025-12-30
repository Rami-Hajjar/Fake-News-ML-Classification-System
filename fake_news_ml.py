# -*- coding: utf-8 -*-
"""
Fake News ML Classification (TF-IDF + Passive-Aggressive)
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Colab: upload CSV files (True.csv and Fake.csv)
try:
    from google.colab import files
    _ = files.upload()
except Exception:
    # If you're not in Colab, ignore (e.g., running locally)
    pass

# --- 1) Load datasets ---
df_true = pd.read_csv("True.csv")
df_fake = pd.read_csv("Fake.csv")

# --- 2) Add labels ---
# True news -> 1, Fake news -> 0
df_true["label"] = 1
df_fake["label"] = 0

# --- 3) Combine + keep only the column we need ---
# Some datasets include title/subject/date; we only use 'text' here.
df = pd.concat([df_true, df_fake], axis=0, ignore_index=True)

# Basic cleanup: drop missing texts, ensure string type
df = df.dropna(subset=["text"]).copy()
df["text"] = df["text"].astype(str)

# Shuffle so the split isn't “all True then all Fake”
df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)

print("Rows, Cols:", df.shape)
print("Label distribution:\n", df["label"].value_counts())

# --- 4) Train-test split (STRATIFIED keeps label ratio consistent) ---
X = df["text"]
y = df["label"]

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=7,
    stratify=y
)

print("\nData split successful!")
print("Train size:", x_train.shape[0], "Test size:", x_test.shape[0])

# --- 5) TF-IDF Vectorization ---
tfidf_vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

print("\nTF-IDF vocab size:", len(tfidf_vectorizer.vocabulary_))

# --- 6) Model: Passive-Aggressive Classifier ---
pac = PassiveAggressiveClassifier(max_iter=50, random_state=7)
pac.fit(tfidf_train, y_train)

# --- 7) Predict + Evaluate ---
y_pred = pac.predict(tfidf_test)

accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\nAccuracy: {accuracy:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("\nConfusion Matrix (rows=actual, cols=pred) labels=[0(Fake), 1(True)]:\n", conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["FAKE (0)", "TRUE (1)"]))

# --- 8) Optional: test on your own custom text ---
custom_text = "The government announced a new policy today after a major investigation."
custom_vec = tfidf_vectorizer.transform([custom_text])
custom_pred = pac.predict(custom_vec)[0]

print("\nCustom test text prediction:", "TRUE" if custom_pred == 1 else "FAKE")

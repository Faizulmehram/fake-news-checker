# src/train_claim_lr.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from pathlib import Path

df = pd.read_csv("data/liar_clean.csv")

# split: 80% train, 10% val, 10% test
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
train_df, val_df  = train_test_split(train_df, test_size=0.1, stratify=train_df["label"], random_state=42)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=800, solver="liblinear"))
])

pipe.fit(train_df["text"], train_df["label"])

val_proba = pipe.predict_proba(val_df["text"])[:, 1]
val_pred_labels = np.where(val_proba >= 0.5, "REAL", "FAKE")
print("VAL AUC:", roc_auc_score((val_df["label"] == "REAL").astype(int), val_proba))
print(classification_report(val_df["label"], val_pred_labels, target_names=["FAKE", "REAL"]))

test_proba = pipe.predict_proba(test_df["text"])[:, 1]
test_pred_labels = np.where(test_proba >= 0.5, "REAL", "FAKE")
print("\n=== TEST ===")
print("TEST AUC:", roc_auc_score((test_df["label"] == "REAL").astype(int), test_proba))
print(classification_report(test_df["label"], test_pred_labels, target_names=["FAKE", "REAL"]))


Path("models").mkdir(exist_ok=True)
joblib.dump(pipe, "models/claim_logreg.joblib")
print("Saved -> models/claim_logreg.joblib")

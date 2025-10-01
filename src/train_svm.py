# src/train_svm.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
import joblib
from pathlib import Path

# 1. load dataset
df = pd.read_csv("data/fake_news_clean.csv")

# 2. split into train/val/test
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
train_df, val_df  = train_test_split(train_df, test_size=0.1, stratify=train_df["label"], random_state=42)

# 3. pipeline: TF-IDF + LinearSVC
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2)),
    ("svm", LinearSVC())
])

# 4. train
pipe.fit(train_df["text"], train_df["label"])

# 5. evaluate on validation set
val_pred = pipe.predict(val_df["text"])
print("Validation F1:", f1_score(val_df["label"], val_pred, pos_label="REAL"))
print(classification_report(val_df["label"], val_pred))

# 6. save model
Path("models").mkdir(exist_ok=True)
joblib.dump(pipe, "models/tfidf_svm.joblib")
print("Model saved -> models/tfidf_svm.joblib")

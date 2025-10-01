import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("data/fake_news_clean.csv")
print(df.head())
print(df["label"].value_counts())


#Step 3: train/val/test split
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)
train_df, val_df = train_test_split(
    train_df, test_size=0.1, stratify=train_df["label"], random_state=42
)

print("Train:", train_df.shape, "Val:", val_df.shape, "Test:", test_df.shape)


#Step 4: build TF-IDF + Logistic Regression pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=500))
])

pipe.fit(train_df["text"], train_df["label"])

#evaluate on validation set
from sklearn.metrics import classification_report, roc_auc_score

val_pred = pipe.predict(val_df["text"])
val_proba = pipe.predict_proba(val_df["text"])[:,1]

print("Validation AUC:", roc_auc_score(val_df["label"], val_proba))
print(classification_report(val_df["label"], val_pred))

#test set evaluation

test_pred = pipe.predict(test_df["text"])
test_proba = pipe.predict_proba(test_df["text"])[:,1]

print("\n=== TEST SET ===")
print("Test AUC:", roc_auc_score(test_df["label"], test_proba))
print(classification_report(test_df["label"], test_pred))
#Step 5: save model
import joblib
joblib.dump(pipe, "models/tfidf_logreg.joblib")
print("Model saved -> models/tfidf_logreg.joblib")
#Save validation/test predictions during training
# --- save predictions for analysis ---
import pandas as pd
from pathlib import Path
reports = Path("reports"); reports.mkdir(exist_ok=True)

val_df = val_df.copy()
val_df["proba_real"] = val_proba
val_df["pred"] = (val_proba >= 0.5).astype(int)
val_df.to_csv(reports / "val_predictions.csv", index=False)

test_df = test_df.copy()
test_df["proba_real"] = test_proba
test_df["pred"] = (test_proba >= 0.5).astype(int)
test_df.to_csv(reports / "test_predictions.csv", index=False)

print("Saved: reports/val_predictions.csv and reports/test_predictions.csv")

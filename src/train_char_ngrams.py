import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("data/fake_news_clean.csv")

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
train_df, val_df  = train_test_split(train_df, test_size=0.1, stratify=train_df["label"], random_state=42)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=5, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=800, solver="liblinear"))
])

pipe.fit(train_df["text"], train_df["label"])
val_proba = pipe.predict_proba(val_df["text"])[:,1]
val_pred  = (val_proba >= 0.5).astype(int)

print("VAL AUC:", roc_auc_score(val_df["label"], val_proba))
print(classification_report(val_df["label"], val_pred))

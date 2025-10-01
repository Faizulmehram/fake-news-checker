# src/make_dataset.py (Step 1: load + inspect)
import pandas as pd

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

print("Fake columns:", list(fake.columns))
print("True columns:", list(true.columns))
print("Fake rows:", len(fake), "True rows:", len(true))

# Step 2: combine title + text and label
fake["combined"] = fake["title"].fillna("") + " " + fake["text"].fillna("")
true["combined"] = true["title"].fillna("") + " " + true["text"].fillna("")
fake["label"] = "FAKE"
true["label"] = "REAL"

df = pd.concat([fake[["combined","label"]], true[["combined","label"]]], ignore_index=True)
df = df.rename(columns={"combined":"text"})

print(df.head())
print(df["label"].value_counts())
# Step 3: save clean file
df.to_csv("data/fake_news_clean.csv", index=False)
print("Saved -> data/fake_news_clean.csv")

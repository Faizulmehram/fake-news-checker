import argparse
import joblib

parser = argparse.ArgumentParser(description="Predict FAKE/REAL for input text(s) using a saved model.")
parser.add_argument("--model", type=str, default="models/tfidf_logreg.joblib", help="Path to a .joblib model file")
parser.add_argument("--text", type=str, help="Single text to classify")
parser.add_argument("--file", type=str, help="Path to a .txt file with one example per line")
args = parser.parse_args()

model = joblib.load(args.model)
print(f"Model loaded: {args.model}")

items = []
if args.text:
    items = [args.text]
elif args.file:
    with open(args.file, "r", encoding="utf-8") as f:
        items = [line.strip() for line in f if line.strip()]
else:
    raise SystemExit("Please pass --text \"...\" or --file path.txt")

probs = model.predict_proba(items)[:, 1]
preds = (probs >= 0.5).astype(int)

for t, p, y in zip(items, probs, preds):
    label = "REAL" if y == 1 else "FAKE"
    preview = (t[:80] + "...") if len(t) > 80 else t
    print(f"{label}\tproba_real={p:.4f}\ttext={preview}")

# src/make_liar_dataset.py
import pandas as pd
from pathlib import Path

FAKE_SET = {"pants-fire", "false", "barely-true"}
REAL_SET = {"true", "mostly-true", "half-true"}

def load_tsv(path: str) -> pd.DataFrame:
    # No header in LIAR .tsv; mirrors may vary in column order
    df = pd.read_csv(path, sep="\t", header=None, quoting=3, dtype=str)
    return df

def pick_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to pick the correct columns for (label, statement).
    Many mirrors: col1=label, col2=statement (as in your screenshot).
    Fallbacks are included just in case.
    """
    candidates = [
        (1, 2),  # most common in mirrors: label, statement
        (0, 1),  # some mirrors: label, statement
        (2, 1),  # rare swaps
    ]
    for li, si in candidates:
        if li < df.shape[1] and si < df.shape[1]:
            tmp = df[[li, si]].copy()
            tmp.columns = ["label", "text"]
            # quick sanity: labels should mostly be among 6 LIAR classes
            lbl_sample = tmp["label"].str.lower().str.strip().head(50)
            ok = lbl_sample.isin(FAKE_SET | REAL_SET).mean()
            if ok > 0.5:
                return tmp
    # If nothing matched, just raise to inspect manually
    raise ValueError("Could not auto-detect label/text columns. Inspect your TSV to confirm indices.")

def map_binary(lbl: str) -> str | None:
    if not isinstance(lbl, str):
        return None
    l = lbl.strip().lower()
    if l in FAKE_SET:
        return "FAKE"
    if l in REAL_SET:
        return "REAL"
    return None

def process_split(path: str) -> pd.DataFrame:
    df = load_tsv(path)
    df = pick_cols(df)
    df["label"] = df["label"].apply(map_binary)
    df = df.dropna(subset=["label", "text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    return df[["text", "label"]]

def main():
    train = process_split("data/train.tsv")
    valid = process_split("data/valid.tsv")
    test  = process_split("data/test.tsv")

    full = pd.concat([train, valid, test], ignore_index=True)
    out = Path("data/liar_clean.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out, index=False, encoding="utf-8")

    print(f"Saved -> {out} | rows: {len(full)}")
    print(full["label"].value_counts())

if __name__ == "__main__":
    main()

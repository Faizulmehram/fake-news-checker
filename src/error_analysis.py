# src/error_analysis.py
import pandas as pd

val = pd.read_csv("reports/val_predictions.csv")
test = pd.read_csv("reports/test_predictions.csv")

# Misclassified examples
val_fp = val[(val["label"] == "FAKE") & (val["pred"] == 1)]   # predicted REAL but actually FAKE
val_fn = val[(val["label"] == "REAL") & (val["pred"] == 0)]   # predicted FAKE but actually REAL

print("VAL — False Positives (FAKE misread as REAL):", len(val_fp))
print(val_fp[["text","proba_real"]].head(5), "\n")

print("VAL — False Negatives (REAL misread as FAKE):", len(val_fn))
print(val_fn[["text","proba_real"]].head(5), "\n")

# Class balance check
print("VAL label counts:\n", val["label"].value_counts())
#See a confusion matrix quickly
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(val["label"].map({"FAKE":0,"REAL":1}), val["pred"])
print("VAL Confusion Matrix:\n", cm)


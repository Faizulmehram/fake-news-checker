# src/app_streamlit.py
import math
from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector")
st.write(
    "Type a headline or short article. This app predicts whether text looks **FAKE** or **REAL** "
    "using a trained ML model. For short factual claims, treat results cautiously."
)

# --- model selector ---
default_model = "models/tfidf_logreg.joblib"  # change to your favorite
model_path = st.text_input(
    "Model path (.joblib)",
    value=default_model,
    help="Point to any saved scikit-learn pipeline."
)
load_btn = st.button("Load model")

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    return joblib.load(p)

model = None
msg = st.empty()
if load_btn or Path(model_path).exists():
    try:
        model = load_model(model_path)
        msg.success(f"Model loaded: {model_path}")
    except Exception as e:
        msg.error(str(e))

# --- input area ---
text = st.text_area(
    "Enter text to analyze",
    height=160,
    placeholder="e.g. Aliens land in Karachi, offer free energy to citizens"
)
threshold = st.slider(
    "Decision threshold for REAL",
    0.0, 1.0, 0.60, 0.01,
    help="Higher = stricter to call something REAL."
)
band = st.slider(
    "Uncertainty band (± around threshold)",
    0.0, 0.25, 0.05, 0.01,
    help="If probability is within this band around the threshold, show 'UNCERTAIN'."
)

approx_ok = st.checkbox(
    "Allow probability approximation for models without predict_proba (e.g., plain LinearSVC)",
    value=True,
    help="Uses a sigmoid on decision_function. Not calibrated; use with caution."
)

predict_clicked = st.button("Predict")

def has_proba(m) -> bool:
    return hasattr(m, "predict_proba")

def has_decision(m) -> bool:
    return hasattr(m, "decision_function")

def sigmoid(x: float) -> float:
    # used only when predict_proba is unavailable; not calibrated
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def get_prob_real(m, text: str) -> tuple[float, str]:
    """
    Return (prob_real, note). If predict_proba exists, use it.
    Otherwise, if decision_function exists and approximation allowed,
    return sigmoid(score) with a caution note. Else raise.
    """
    if has_proba(m):
        p = float(m.predict_proba([text])[0][1])
        return p, ""
    if has_decision(m) and approx_ok:
        score = float(m.decision_function([text])[0])
        p = sigmoid(score)  # uncalibrated
        return p, "Approximate probability from decision_function (uncalibrated)."
    raise AttributeError("This model does not provide probabilities. "
                         "Use a LogisticRegression model or a Calibrated SVM.")

def decide_label(prob_real: float, thr: float, band_width: float) -> str:
    low, high = thr - band_width, thr + band_width
    if prob_real >= high:
        return "REAL"
    if prob_real <= low:
        return "FAKE"
    return "UNCERTAIN"

if predict_clicked:
    if not model:
        st.warning("Load a model first.")
    elif not text.strip():
        st.warning("Please enter some text.")
    else:
        # Short-text caution
        if len(text.strip().split()) < 8:
            st.info(
                "This looks like a very short input. The news-style classifier can be unreliable on short factual claims. "
                "Add more context (headline + 1–2 sentences) or use a claim-specific model."
            )

        try:
            prob_real, note = get_prob_real(model, text.strip())
            label = decide_label(prob_real, threshold, band)

            st.subheader("Result")
            if label == "REAL":
                st.success(f"✅ Predicted: **REAL** (probability REAL = {prob_real:.3f})")
            elif label == "FAKE":
                st.error(f"❌ Predicted: **FAKE** (probability REAL = {prob_real:.3f})")
            else:
                st.warning(f"🤔 Predicted: **UNCERTAIN** (probability REAL = {prob_real:.3f})")

            st.progress(min(max(prob_real if label == "REAL" else 1 - prob_real, 0.0), 1.0))

            with st.expander("Show details"):
                details = {
                    "model_path": model_path,
                    "threshold": threshold,
                    "uncertainty_band": band,
                    "prob_real": prob_real,
                    "text_preview": text[:120] + ("..." if len(text) > 120 else "")
                }
                if note:
                    details["note"] = note
                st.write(details)

            if note:
                st.info("Note: " + note)

        except AttributeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Error during prediction: {e}")

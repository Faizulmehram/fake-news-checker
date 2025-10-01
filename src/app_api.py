# src/app_api.py
from __future__ import annotations
from pathlib import Path
import math
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- RAFC (Wikipedia + NLI) ---
from src.rafc import RAFC
rafc = RAFC()   # first use may download weights

# ==== paths to your saved scikit models ====
NEWS_MODEL_PATH  = Path("models/tfidf_logreg.joblib")   # news article model
CLAIM_MODEL_PATH = Path("models/claim_logreg.joblib")   # claim (LIAR) model
# ===========================================

app = FastAPI(title="Fake News / Claim Checker API")

# allow local web pages to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # dev only; tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- scikit helpers (news/claim) --------
_models: dict[str, object] = {}

def load_model(kind: str):
    if kind in _models:
        return _models[kind]
    path = NEWS_MODEL_PATH if kind == "news" else CLAIM_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    m = joblib.load(path)
    _models[kind] = m
    return m

def has_proba(m): return hasattr(m, "predict_proba")
def has_decision(m): return hasattr(m, "decision_function")

def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def prob_real_scikit(m, text: str):
    """Return (p, note). Prefer predict_proba; fallback to decision_function (uncalibrated)."""
    if has_proba(m):
        p = float(m.predict_proba([text])[0][1])
        return p, ""
    if has_decision(m):
        score = float(m.decision_function([text])[0])
        p = _sigmoid(score)
        return p, "Approx prob from decision_function (uncalibrated)."
    raise AttributeError("Model provides neither predict_proba nor decision_function.")

def decide_label(p: float, thr: float, band: float):
    low, high = thr - band, thr + band
    if p >= high: return "REAL"
    if p <= low:  return "FAKE"
    return "UNCERTAIN"

# --------------- API schemas ----------------
class PredictIn(BaseModel):
    text: str
    model_type: str = "news"   # "news" | "claim" | "rafc"
    threshold: float = 0.70    # backend defaults (hidden in simple UI)
    band: float = 0.10         # uncertainty band (±)

class PredictOut(BaseModel):
    label: str
    prob_real: float
    model_type: str
    threshold: float
    band: float
    note: str | None = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    kind = inp.model_type.lower().strip()
    text = (inp.text or "").strip()

    if not text:
        return PredictOut(label="UNCERTAIN", prob_real=0.5, model_type=kind,
                          threshold=inp.threshold, band=inp.band, note="Empty text.")

    # ---------- FACT-CHECK (Wikipedia + NLI) ----------
    if kind.startswith("rafc"):
        # guardrail for extremely short claims
        if len(text.split()) < 4:
            return PredictOut(label="UNCERTAIN", prob_real=0.5, model_type="rafc",
                              threshold=inp.threshold, band=inp.band,
                              note="Very short claim; need more context.")
        try:
            label, p, ev = rafc.check(
                text,
                k_pages=6,      # how many wiki pages to pull
                k_sents=10,     # how many candidate sentences to score
                ent_threshold=0.80,
                con_threshold=0.80,
            )
            note = None
            if ev:
                note = (f"Top evidence from Wikipedia page '{ev.page_title}': "
                        f"\"{ev.sentence}\"  (entail={ev.entail:.2f}, contradict={ev.contradict:.2f})")
            return PredictOut(label=label, prob_real=float(p), model_type="rafc",
                              threshold=inp.threshold, band=inp.band, note=note)
        except Exception as e:
            # surface a clear message instead of a server error
            return PredictOut(label="UNCERTAIN", prob_real=0.5, model_type="rafc",
                              threshold=inp.threshold, band=inp.band,
                              note=f"Fact-check error: {e}")

    # ---------------- NEWS / CLAIM --------------------
    kind = "claim" if kind.startswith("claim") else "news"
    m = load_model(kind)

    note = None
    if kind == "news" and len(text.split()) < 8:
        note = "Short input; news model may be unreliable for brief factual claims."

    p, note2 = prob_real_scikit(m, text)
    if note2:
        note = (note + " " if note else "") + note2

    label = decide_label(p, inp.threshold, inp.band)
    return PredictOut(label=label, prob_real=float(p), model_type=kind,
                      threshold=inp.threshold, band=inp.band, note=note)

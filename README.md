# fake-news-checker
■ Fake News / Claim Checker Project
This project detects **fake news**, verifies **claims**, and performs **Wikipedia-based fact-checking**.
It has a backend powered by **FastAPI**, training scripts, and a simple **HTML frontend**.
■ Features
- **News Classifier** (TF-IDF + Logistic Regression)
- **Claim Classifier** (trained on LIAR dataset)
- **Wikipedia Fact-Checker (RAFC)** using RoBERTa MNLI for entailment/contradiction
■ Project Structure
- **frontend/** → index.html (UI for predictions)
- **models/** → Pretrained `.joblib` models (news, claim, svm)
- **src/** → All backend and training scripts
- `app_api.py` → FastAPI backend (main API)
- `app_streamlit.py` → Alternative Streamlit app
- `rafc.py` → Wikipedia Retrieval + NLI fact-checker
- `make_dataset.py`, `make_liar_dataset.py` → Preprocessing
- `train_baseline.py`, `train_claim_lr.py`, `train_svm.py` → Training scripts
- `predict_baseline.py` → Quick prediction script
- `error_analysis.py` → Error inspection
- **requirements.txt** → Python dependencies
- **.gitignore** → Ignore venv, cache, and data
■ How to Run the Project
1. **Clone repository**
```bash
git clone https://github.com//fake-news-checker.git
cd fake-news-checker
```
2. **Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate # Windows
source .venv/bin/activate # Linux/Mac
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Run backend (FastAPI)**
```bash
uvicorn src.app_api:app --reload --port 8000
```
Backend available at:
- http://127.0.0.1:8000
- Health check: http://127.0.0.1:8000/health
5. **Run frontend (UI)**
```bash
cd frontend
python -m http.server 5500
```
Open: http://localhost:5500/index.html
■■ Configuration
- Models stored in `models/`
- Wikipedia fact-checker uses `roberta-large-mnli` (configurable in `rafc.py`)
- Short inputs may give unreliable predictions in the news model
■ Example Predictions
- "Islamabad is the capital of Pakistan" → ■ REAL
- "Donald Trump is president of India" → ■ FAKE
- "The Moon is made of cheese" → ■ UNCERTAIN
■ License
For educational purposes only.
■ Contributing
Pull requests are welcome. For major changes, please open an issue first.

# src/rafc.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

import re
import math
import nltk
import requests
import wikipedia
from rank_bm25 import BM25Okapi

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- sentence tokenizer (auto-download) ---
try:
    _SENT_TOKENIZER = nltk.data.load("tokenizers/punkt/english.pickle")
except LookupError:
    nltk.download("punkt")
    try:
        nltk.download("punkt_tab")
    except Exception:
        pass
    _SENT_TOKENIZER = nltk.data.load("tokenizers/punkt/english.pickle")

# Wikipedia language + HTTP session
wikipedia.set_lang("en")
_SES = requests.Session()
_SES.headers.update({"User-Agent": "FactCheckApp/1.0 (local dev)"})
_TIMEOUT = 6.0  # seconds

# Valid public MNLI models (label order: [contradiction, neutral, entailment])
_NLI_MODEL_NAME = "roberta-large-mnli"  # or: "facebook/bart-large-mnli"


@dataclass
class Evidence:
    sentence: str
    page_title: str
    entail: float
    contradict: float
    neutral: float


class RAFC:
    def __init__(self, nli_model_name: str = _NLI_MODEL_NAME, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(nli_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.model.to(self.device).eval()

    # ---------- Wikipedia retrieval ----------

    def wiki_search(self, query: str, k_pages: int = 5) -> List[str]:
        # primary: python-wikipedia search
        try:
            titles = wikipedia.search(query, results=k_pages)
            if titles:
                return titles
        except Exception:
            pass
        # fallback: opensearch API
        try:
            r = _SES.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action":"opensearch","format":"json","limit":k_pages,"search":query},
                timeout=_TIMEOUT
            )
            r.raise_for_status()
            data = r.json()
            return list(data[1]) if isinstance(data, list) and len(data) >= 2 else []
        except Exception:
            return []

    def _rest_summary(self, title: str) -> str:
        # Use REST summary endpoint (handles redirects & is fast)
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
            r = _SES.get(url, timeout=_TIMEOUT)
            if r.status_code == 404:
                return ""
            r.raise_for_status()
            j = r.json()
            # prefer 'extract', fallback to 'description'
            text = (j.get("extract") or j.get("description") or "").strip()
            return text
        except Exception:
            return ""

    def wiki_sentences(self, titles: List[str], max_per_page: int = 40) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for t in titles:
            # try REST summary first; fallback to python-wikipedia summary
            text = self._rest_summary(t)
            if not text:
                try:
                    text = wikipedia.summary(t, auto_suggest=False, sentences=6)
                except Exception:
                    text = ""
            if not text:
                continue
            sents = _SENT_TOKENIZER.tokenize(text)
            for s in sents[:max_per_page]:
                s = re.sub(r"\s+", " ", s).strip()
                wc = len(s.split())
                if 6 <= wc <= 60:
                    out.append((s, t))
        return out

    def bm25_topk(self, claim: str, sent_page_pairs: List[Tuple[str, str]], k: int = 8):
        if not sent_page_pairs:
            return []
        sents = [s for s, _ in sent_page_pairs]
        tokenized = [s.lower().split() for s in sents]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(claim.lower().split())
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(sents[i], sent_page_pairs[i][1]) for i in idxs]

    # ---------- NLI ----------

    @torch.no_grad()
    def nli_probs(self, premise: str, hypothesis: str) -> Tuple[float, float, float]:
        enc = self.tok(
            premise,
            hypothesis,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits[0].detach().cpu()
        probs = torch.softmax(logits, dim=-1).numpy().tolist()
        contr, neut, ent = probs[0], probs[1], probs[2]
        return float(ent), float(contr), float(neut)

    def check(
        self,
        claim: str,
        k_pages: int = 5,
        k_sents: int = 8,
        ent_threshold: float = 0.80,
        con_threshold: float = 0.80,
    ):
        titles = self.wiki_search(claim, k_pages=k_pages)
        sent_pairs = self.wiki_sentences(titles)
        topk = self.bm25_topk(claim, sent_pairs, k=k_sents)

        best_ev: Optional[Evidence] = None
        decision = "UNCERTAIN"

        for sent, title in topk:
            ent, con, neu = self.nli_probs(premise=sent, hypothesis=claim)
            ev = Evidence(sentence=sent, page_title=title, entail=ent, contradict=con, neutral=neu)
            if (best_ev is None) or (max(ent, con) > max(best_ev.entail, best_ev.contradict)):
                best_ev = ev
            if ent >= ent_threshold:
                decision = "REAL"; break
            if con >= con_threshold:
                decision = "FAKE"; break

        prob_real = best_ev.entail if best_ev else 0.5
        return decision, prob_real, best_ev

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_extraction_models.py
============================
Comparativa de modelos para extracción del importe total en facturas/tickets
procesados con OCR (EasyOCR) sobre el dataset CORD v2 (Naver-Clova).

Fuentes de datos
----------------
  A) Ficheros JSONL generados con EasyOCR  (texto con errores de lectura)
  B) Ground truth reconstruido desde los metadatos de CORD  (texto limpio)

Preprocesamiento
----------------
  0) Texto crudo — sin preprocesamiento
  1) Corrección de errores OCR + normalización

Modelos
-------
  1. Regex Keyword     — busca TOTAL/DUE y extrae el número adyacente
  2. Regex Max         — mayor importe filtrado (descarta CASH/CHANGE)
  3. Random Forest     — clasificador sobre features de candidatos numéricos
  4. Gradient Boosting — ídem con GradientBoosting
  5. TF-IDF + LogReg   — contexto textual alrededor de cada candidato
  6. DistilBERT        — reranker de candidatos con transformer  (opcional)
  7. T5 Seq2Seq        — generación directa del total              (opcional)

Uso
---
  python scripts/compare_extraction_models.py
  python scripts/compare_extraction_models.py --skip-transformers
  python scripts/compare_extraction_models.py --skip-cord-gt
"""

import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

SEED = 42
np.random.seed(SEED)

TOTAL_KEYWORDS = re.compile(
    r"\b(grand\s*total|total|totai|tota[l1i]|gnd\s*tota[l1i]|"
    r"due|amount|amt|tl|rounding)\b",
    re.IGNORECASE,
)
NEGATIVE_KEYWORDS = re.compile(
    r"\b(cash|change|changed|kembal[i1]|tunai|debit|credit|card|"
    r"bca|tendered|bayar|paid|pago|efectivo)\b",
    re.IGNORECASE,
)
SUBTOTAL_KEYWORDS = re.compile(
    r"\b(sub\s*-?\s*total|subtotal|subttl|sub\s*ttl|svc|service|"
    r"tax|pajak|pb[1i]|disc|discount)\b",
    re.IGNORECASE,
)


# ============================================================
# 2. UTILIDADES — normalización de importes
# ============================================================

def normalize_amount(s: str) -> float | None:
    """Convierte cualquier string de importe a float.

    Maneja formatos:
      - Miles con coma:   1,591,600  /  580,965
      - Miles con punto:  25.000  /  16.500   (formato indonesio)
      - Mixto con decimales: 226,500.00  /  35.000,00
      - Sin separador:  36300
      - Con prefijo moneda:  Rp 16.500 / Rp. 73.450
    """
    if s is None:
        return None
    s = str(s).strip()
    # Quitar prefijo de moneda
    s = re.sub(r"^[Rr][Pp]\.?\s*", "", s)
    s = re.sub(r"^(IDR|USD|\$|€|£)\s*", "", s, flags=re.IGNORECASE)
    s = s.strip().replace(" ", "")

    if not s:
        return None

    # Caso: punto Y coma presentes  →  el último es el separador decimal
    if "." in s and "," in s:
        if s.rfind(".") > s.rfind(","):
            # punto es decimal: 226,500.00
            integer_part = s[: s.rfind(".")].replace(",", "")
            decimal_part = s[s.rfind(".") + 1 :]
        else:
            # coma es decimal: 35.000,00
            integer_part = s[: s.rfind(",")].replace(".", "")
            decimal_part = s[s.rfind(",") + 1 :]
        try:
            return float(f"{integer_part}.{decimal_part}")
        except ValueError:
            return None

    # Solo punto
    if "." in s:
        parts = s.split(".")
        if all(len(p) == 3 for p in parts[1:]):
            # Separador de miles: 25.000
            return _safe_float(s.replace(".", ""))
        if len(parts[-1]) <= 2:
            return _safe_float(s)
        return _safe_float(s.replace(".", ""))

    # Solo coma
    if "," in s:
        parts = s.split(",")
        if all(len(p) == 3 for p in parts[1:]):
            return _safe_float(s.replace(",", ""))
        if len(parts[-1]) <= 2:
            return _safe_float(s.replace(",", "."))
        return _safe_float(s.replace(",", ""))

    return _safe_float(s)


def _safe_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# ============================================================
# 3. PREPROCESAMIENTO DE TEXTO OCR
# ============================================================

def preprocess_ocr_text(text: str) -> str:
    """Corrige errores habituales de OCR en texto de facturas."""
    # 3a. O/o → 0 en contextos numéricos
    text = re.sub(r"(?<=\d)[OoQq]+", lambda m: "0" * len(m.group()), text)
    text = re.sub(r"[OoQq]+(?=\d)", lambda m: "0" * len(m.group()), text)
    # Secuencias tipo ,Ooo  .ooo  ,OOo
    text = re.sub(
        r"(?<=[\d,.])[OoQq]{2,3}(?=[\s,.\-;:)\]|$])",
        lambda m: "0" * len(m.group()),
        text,
    )
    # 3b. D → 0 junto a dígitos (29,Doo → 29,000)
    text = re.sub(r"(?<=\d)[Dd](?=[\dOo0])", "0", text)
    # 3c. l/I → 1 junto a dígitos
    text = re.sub(r"(?<=\d)[Il](?=[\d])", "1", text)
    text = re.sub(r"[Il](?=\d{2,})", "1", text)
    # 3d. Minúsculas
    text = text.lower()
    # 3e. Quitar caracteres basura típicos de OCR
    text = re.sub(r"[{}\[\]<>~`|\\^\"']", " ", text)
    # 3f. Normalizar espacios
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# 4. EXTRACCIÓN DE CANDIDATOS NUMÉRICOS
# ============================================================

_OCR_DIGIT = r"[\dOoQq]"
_CANDIDATE_PATTERNS = [
    # Con separador de miles (coma o punto): 1,591,600 | 25.000 | 226,500.00
    rf"{_OCR_DIGIT}{{1,3}}(?:[.,]\s?{_OCR_DIGIT}{{3}})+(?:[.,]{_OCR_DIGIT}{{1,2}})?",
    # Espacio como separador de miles: 1 591 600
    r"\d{1,3}(?:\s\d{3})+",
    # Sin separador (4–10 dígitos): 36300
    rf"{_OCR_DIGIT}{{4,10}}",
]
_CANDIDATE_RE = re.compile("(" + "|".join(_CANDIDATE_PATTERNS) + ")")


def extract_number_candidates(text: str) -> list[dict]:
    """Extrae candidatos numéricos del texto con su contexto."""
    candidates = []
    for m in _CANDIDATE_RE.finditer(text):
        raw = m.group(1)
        if not re.search(r"\d", raw):
            continue
        cleaned = re.sub(r"[OoQq]", "0", raw).replace(" ", "")
        value = normalize_amount(cleaned)
        if value is None or value < 100:
            continue
        start, end = m.start(1), m.end(1)
        candidates.append(
            {
                "raw": raw,
                "value": value,
                "start": start,
                "end": end,
                "context_before": text[max(0, start - 60) : start],
                "context_after": text[end : end + 60],
            }
        )
    return candidates


# ============================================================
# 5. FEATURES PARA MODELOS DE CANDIDATOS
# ============================================================

def candidate_features(text: str, candidates: list[dict], idx: int) -> np.ndarray:
    """Genera un vector de features para el candidato en posición `idx`."""
    c = candidates[idx]
    text_len = max(len(text), 1)
    all_values = [cc["value"] for cc in candidates]
    max_val = max(all_values) if all_values else 1.0

    # Posición relativa (0 = inicio, 1 = final)
    rel_pos = c["end"] / text_len

    # Magnitud
    log_value = np.log1p(c["value"])
    ratio_to_max = c["value"] / max_val if max_val > 0 else 0.0
    is_max = float(c["value"] == max_val)

    # Proximidad a keywords (en caracteres, normalizado)
    ctx = c["context_before"] + " " + c["context_after"]
    dist_total = _keyword_distance(text, c, TOTAL_KEYWORDS)
    dist_negative = _keyword_distance(text, c, NEGATIVE_KEYWORDS)
    dist_subtotal = _keyword_distance(text, c, SUBTOTAL_KEYWORDS)

    # ¿Aparece TOTAL/DUE en el contexto cercano?
    has_total_ctx = float(bool(TOTAL_KEYWORDS.search(ctx)))
    has_negative_ctx = float(bool(NEGATIVE_KEYWORDS.search(ctx)))
    has_subtotal_ctx = float(bool(SUBTOTAL_KEYWORDS.search(ctx)))

    # Número de candidatos
    n_candidates = len(candidates)

    # ¿Es el último candidato antes de una keyword negativa?
    is_last_before_neg = 0.0
    neg_match = NEGATIVE_KEYWORDS.search(text[c["end"] :])
    if neg_match:
        between = text[c["end"] : c["end"] + neg_match.start()]
        if not _CANDIDATE_RE.search(between):
            is_last_before_neg = 1.0

    return np.array(
        [
            rel_pos,
            log_value,
            ratio_to_max,
            is_max,
            dist_total,
            dist_negative,
            dist_subtotal,
            has_total_ctx,
            has_negative_ctx,
            has_subtotal_ctx,
            n_candidates,
            is_last_before_neg,
        ],
        dtype=np.float32,
    )


def _keyword_distance(text: str, candidate: dict, pattern: re.Pattern) -> float:
    """Distancia normalizada al keyword más cercano (0 = adyacente, 1 = lejos)."""
    text_len = max(len(text), 1)
    mid = (candidate["start"] + candidate["end"]) / 2
    best = text_len
    for m in pattern.finditer(text):
        km = (m.start() + m.end()) / 2
        best = min(best, abs(mid - km))
    return best / text_len


# ============================================================
# 6. CARGA DE DATOS
# ============================================================

def load_jsonl(path: str | Path) -> list[dict]:
    """Carga un fichero JSONL con campos text, label, image_id."""
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_cord_gt(split: str = "train", limit: int | None = None) -> list[dict]:
    """Carga CORD desde HuggingFace y reconstruye texto desde ground truth.

    Cada muestra devuelta tiene:
      - text: texto reconstruido desde gt_parse (sin errores OCR)
      - label: total_price del ground truth
      - image_id: índice en el dataset
    """
    from datasets import load_dataset

    dataset = load_dataset("naver-clova-ix/cord-v2", split=split, streaming=True)

    data = []
    for i, sample in enumerate(dataset):
        if limit and i >= limit:
            break
        try:
            gt = json.loads(sample["ground_truth"])
            gp = gt.get("gt_parse", {})
            total_price = gp.get("total", {}).get("total_price")
            if not total_price:
                continue
            text = _reconstruct_text(gp)
            data.append({"text": text, "label": total_price, "image_id": i})
        except Exception:
            continue
    return data


def _reconstruct_text(gt_parse: dict) -> str:
    """Reconstruye texto de factura desde la estructura gt_parse de CORD."""
    parts: list[str] = []

    # Ítems del menú
    for item in gt_parse.get("menu", []):
        for key in ("nm", "cnt", "unitprice", "price"):
            val = item.get(key)
            if val:
                parts.append(str(val))
        # Sub-ítems
        for sub in item.get("sub", []):
            for key in ("nm", "cnt", "price"):
                val = sub.get(key)
                if val:
                    parts.append(str(val))

    # Subtotal
    sub = gt_parse.get("sub_total", {})
    if sub.get("subtotal_price"):
        parts += ["Subtotal", str(sub["subtotal_price"])]
    if sub.get("tax_price"):
        parts += ["Tax", str(sub["tax_price"])]
    if sub.get("service_price"):
        parts += ["Service", str(sub["service_price"])]
    if sub.get("discount_price"):
        parts += ["Discount", str(sub["discount_price"])]

    # Total
    total = gt_parse.get("total", {})
    if total.get("total_price"):
        parts += ["Total", str(total["total_price"])]
    if total.get("cashprice"):
        parts += ["Cash", str(total["cashprice"])]
    if total.get("changeprice"):
        parts += ["Change", str(total["changeprice"])]

    return " ".join(parts)


# ============================================================
# 7. MODELOS
# ============================================================

class BaseModel:
    name: str = "Base"
    needs_training: bool = False

    def train(self, train_texts: list[str], train_labels: list[str]):
        pass

    def predict(self, text: str) -> str | None:
        raise NotImplementedError

    def predict_batch(self, texts: list[str]) -> list[str | None]:
        return [self.predict(t) for t in texts]


# -------------------------------------------------------
# 7.1  Regex Keyword  — busca keyword TOTAL y extrae nº
# -------------------------------------------------------
class RegexKeywordModel(BaseModel):
    name = "Regex Keyword"

    def predict(self, text: str) -> str | None:
        candidates = extract_number_candidates(text)
        if not candidates:
            return None

        # Buscar keyword TOTAL/DUE y devolver el candidato más cercano posterior
        for m in TOTAL_KEYWORDS.finditer(text):
            kw_end = m.end()
            best, best_dist = None, float("inf")
            for c in candidates:
                if c["start"] >= kw_end - 5:
                    dist = c["start"] - kw_end
                    if 0 <= dist < best_dist:
                        # Descartar si entre el keyword y el número hay un keyword negativo
                        between = text[kw_end : c["start"]]
                        if not NEGATIVE_KEYWORDS.search(between):
                            best = c
                            best_dist = dist
            if best:
                return best["raw"]

        # Fallback: mayor candidato sin contexto negativo
        filtered = [
            c
            for c in candidates
            if not NEGATIVE_KEYWORDS.search(c["context_before"][-30:])
            and not NEGATIVE_KEYWORDS.search(c["context_after"][:30])
        ]
        if filtered:
            return max(filtered, key=lambda c: c["value"])["raw"]
        return max(candidates, key=lambda c: c["value"])["raw"]


# -------------------------------------------------------
# 7.2  Regex Max  — mayor importe filtrado
# -------------------------------------------------------
class RegexMaxFilteredModel(BaseModel):
    name = "Regex Max Filtrado"

    def predict(self, text: str) -> str | None:
        candidates = extract_number_candidates(text)
        if not candidates:
            return None

        # Filtrar candidatos en contexto CASH / CHANGE
        filtered = []
        for c in candidates:
            ctx = c["context_before"][-40:] + " " + c["context_after"][:40]
            if not NEGATIVE_KEYWORDS.search(ctx):
                filtered.append(c)

        pool = filtered if filtered else candidates
        return max(pool, key=lambda c: c["value"])["raw"]


# -------------------------------------------------------
# 7.3 – 7.4  Candidate ML  (RF / GB / SVM)
# -------------------------------------------------------
class CandidateMLModel(BaseModel):
    needs_training = True

    def __init__(self, classifier, clf_name: str = "ML"):
        self.clf = classifier
        self.name = f"Candidate {clf_name}"
        self._scaler = None

    def train(self, train_texts: list[str], train_labels: list[str]):
        from sklearn.preprocessing import StandardScaler

        X, y = [], []
        for text, label in zip(train_texts, train_labels):
            candidates = extract_number_candidates(text)
            if not candidates:
                continue
            label_val = normalize_amount(label)
            if label_val is None:
                continue
            best_idx = _find_best_match(candidates, label_val)
            for i, c in enumerate(candidates):
                feats = candidate_features(text, candidates, i)
                X.append(feats)
                y.append(1 if i == best_idx else 0)

        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X)
        self.clf.fit(X, y)

    def predict(self, text: str) -> str | None:
        candidates = extract_number_candidates(text)
        if not candidates:
            return None

        X = np.array(
            [candidate_features(text, candidates, i) for i in range(len(candidates))],
            dtype=np.float32,
        )
        X = self._scaler.transform(X)

        if hasattr(self.clf, "predict_proba"):
            probs = self.clf.predict_proba(X)[:, 1]
            best_idx = int(np.argmax(probs))
        else:
            scores = self.clf.decision_function(X)
            best_idx = int(np.argmax(scores))

        return candidates[best_idx]["raw"]


def _find_best_match(candidates: list[dict], label_val: float) -> int:
    """Encuentra el candidato más cercano numéricamente al label."""
    best_idx, best_diff = 0, float("inf")
    for i, c in enumerate(candidates):
        diff = abs(c["value"] - label_val) / max(label_val, 1)
        if diff < best_diff:
            best_idx = i
            best_diff = diff
    return best_idx


# -------------------------------------------------------
# 7.5  TF-IDF Context + Logistic Regression
# -------------------------------------------------------
class TFIDFContextModel(BaseModel):
    name = "TF-IDF + LogReg"
    needs_training = True

    def __init__(self, window: int = 80):
        self.window = window
        self._vectorizer = None
        self._scaler = None
        self._clf = None

    def train(self, train_texts: list[str], train_labels: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from scipy.sparse import hstack, csr_matrix

        ctx_texts, num_feats, y = [], [], []

        for text, label in zip(train_texts, train_labels):
            candidates = extract_number_candidates(text)
            if not candidates:
                continue
            label_val = normalize_amount(label)
            if label_val is None:
                continue
            best_idx = _find_best_match(candidates, label_val)
            for i, c in enumerate(candidates):
                window_text = (
                    text[max(0, c["start"] - self.window) : c["end"] + self.window]
                )
                ctx_texts.append(window_text.lower())
                num_feats.append(candidate_features(text, candidates, i))
                y.append(1 if i == best_idx else 0)

        self._vectorizer = TfidfVectorizer(
            max_features=3000, ngram_range=(1, 2), sublinear_tf=True
        )
        X_tfidf = self._vectorizer.fit_transform(ctx_texts)

        num_feats = np.array(num_feats, dtype=np.float32)
        self._scaler = StandardScaler()
        num_scaled = self._scaler.fit_transform(num_feats)

        X = hstack([X_tfidf, csr_matrix(num_scaled)])
        y = np.array(y)

        self._clf = LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced", random_state=SEED
        )
        self._clf.fit(X, y)

    def predict(self, text: str) -> str | None:
        from scipy.sparse import hstack, csr_matrix

        candidates = extract_number_candidates(text)
        if not candidates:
            return None

        ctx_texts = []
        num_feats = []
        for i, c in enumerate(candidates):
            window_text = (
                text[max(0, c["start"] - self.window) : c["end"] + self.window]
            )
            ctx_texts.append(window_text.lower())
            num_feats.append(candidate_features(text, candidates, i))

        X_tfidf = self._vectorizer.transform(ctx_texts)
        num_scaled = self._scaler.transform(np.array(num_feats, dtype=np.float32))
        X = hstack([X_tfidf, csr_matrix(num_scaled)])

        probs = self._clf.predict_proba(X)[:, 1]
        return candidates[int(np.argmax(probs))]["raw"]


# -------------------------------------------------------
# 7.6  DistilBERT Candidate Reranker  (opcional)
# -------------------------------------------------------
class DistilBERTReranker(BaseModel):
    name = "DistilBERT Reranker"
    needs_training = True

    def __init__(self, epochs: int = 3, batch_size: int = 16, lr: float = 2e-5):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._model = None
        self._tokenizer = None

    def train(self, train_texts: list[str], train_labels: list[str]):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

        self._tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-multilingual-cased"
        )
        self._model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-multilingual-cased", num_labels=2
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)

        # Construir pares (contexto, label)
        inputs, labels = [], []
        for text, label in zip(train_texts, train_labels):
            candidates = extract_number_candidates(text)
            if not candidates:
                continue
            label_val = normalize_amount(label)
            if label_val is None:
                continue
            best_idx = _find_best_match(candidates, label_val)
            for i, c in enumerate(candidates):
                window = text[max(0, c["start"] - 100) : c["end"] + 100]
                pair_text = f"{window} [SEP] {c['raw']}"
                inputs.append(pair_text)
                labels.append(1 if i == best_idx else 0)

        enc = self._tokenizer(
            inputs, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        label_tensor = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], label_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.lr)
        self._model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in loader:
                ids, mask, lbl = [b.to(device) for b in batch]
                out = self._model(input_ids=ids, attention_mask=mask, labels=lbl)
                out.loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += out.loss.item()
            print(f"  DistilBERT epoch {epoch+1}/{self.epochs}  loss={total_loss/len(loader):.4f}")

        self._model.eval()

    def predict(self, text: str) -> str | None:
        import torch

        candidates = extract_number_candidates(text)
        if not candidates or self._model is None:
            return None

        device = next(self._model.parameters()).device
        inputs = []
        for c in candidates:
            window = text[max(0, c["start"] - 100) : c["end"] + 100]
            inputs.append(f"{window} [SEP] {c['raw']}")

        enc = self._tokenizer(
            inputs, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        with torch.no_grad():
            out = self._model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()

        return candidates[int(np.argmax(probs))]["raw"]


# -------------------------------------------------------
# 7.7  T5 Seq2Seq  (opcional)
# -------------------------------------------------------
class T5SeqModel(BaseModel):
    name = "T5 Seq2Seq"
    needs_training = True

    def __init__(self, epochs: int = 10, batch_size: int = 8, lr: float = 3e-4):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._model = None
        self._tokenizer = None

    def train(self, train_texts: list[str], train_labels: list[str]):
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        self._tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self._model = T5ForConditionalGeneration.from_pretrained("t5-small")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)

        class _DS(Dataset):
            def __init__(ds, texts, labels, tokenizer):
                ds.texts = texts
                ds.labels = labels
                ds.tok = tokenizer

            def __len__(ds):
                return len(ds.texts)

            def __getitem__(ds, idx):
                import torch as _torch

                src = ds.tok(
                    f"extract total: {ds.texts[idx]}",
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                )
                tgt = ds.tok(
                    str(ds.labels[idx]),
                    max_length=32,
                    padding="max_length",
                    truncation=True,
                )
                lbl = [
                    t if t != ds.tok.pad_token_id else -100
                    for t in tgt.input_ids
                ]
                return {
                    "input_ids": _torch.tensor(src.input_ids, dtype=_torch.long),
                    "attention_mask": _torch.tensor(src.attention_mask, dtype=_torch.long),
                    "labels": _torch.tensor(lbl, dtype=_torch.long),
                }

        dataset = _DS(train_texts, train_labels, self._tokenizer)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.lr)

        self._model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                lbl = batch["labels"].to(device)
                out = self._model(input_ids=ids, attention_mask=mask, labels=lbl)
                out.loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += out.loss.item()
            print(f"  T5 epoch {epoch+1}/{self.epochs}  loss={total_loss/len(loader):.4f}")

        self._model.eval()

    def predict(self, text: str) -> str | None:
        import torch

        if self._model is None:
            return None
        device = next(self._model.parameters()).device

        enc = self._tokenizer(
            f"extract total: {text}",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = self._model.generate(
                input_ids=enc.input_ids.to(device),
                attention_mask=enc.attention_mask.to(device),
                max_new_tokens=32,
            )
        decoded = self._tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return decoded if decoded else None


# ============================================================
# 8. MÉTRICAS DE EVALUACIÓN
# ============================================================

def compute_metrics(
    predictions: list[str | None], ground_truths: list[str]
) -> dict[str, float]:
    """Calcula métricas de evaluación para la extracción de totales.

    Métricas devueltas:
      - prediction_rate : fracción de muestras donde el modelo devuelve algo
      - exact_match     : coincidencia exacta tras normalizar a float
      - match_1pct      : |pred − true| / true ≤ 0.01
      - match_5pct      : |pred − true| / true ≤ 0.05
      - match_10pct     : |pred − true| / true ≤ 0.10
      - mean_rel_error  : media de |pred − true| / true  (solo sobre predicciones)
    """
    n = len(ground_truths)
    n_predicted = 0
    exact = 0
    m1 = 0
    m5 = 0
    m10 = 0
    rel_errors = []

    for pred, gt in zip(predictions, ground_truths):
        gt_val = normalize_amount(gt)
        if gt_val is None or gt_val == 0:
            n -= 1
            continue

        if pred is None:
            continue

        pred_val = normalize_amount(pred)
        if pred_val is None:
            continue

        n_predicted += 1
        rel_err = abs(pred_val - gt_val) / gt_val
        rel_errors.append(rel_err)

        if rel_err < 1e-6:
            exact += 1
        if rel_err <= 0.01:
            m1 += 1
        if rel_err <= 0.05:
            m5 += 1
        if rel_err <= 0.10:
            m10 += 1

    n = max(n, 1)
    return {
        "prediction_rate": n_predicted / n,
        "exact_match": exact / n,
        "match_1pct": m1 / n,
        "match_5pct": m5 / n,
        "match_10pct": m10 / n,
        "mean_rel_error": float(np.mean(rel_errors)) if rel_errors else 1.0,
    }


# ============================================================
# 9. EJECUCIÓN DE EXPERIMENTOS
# ============================================================

def run_experiment(
    model: BaseModel,
    train_data: list[dict],
    test_data: list[dict],
    preprocess: bool,
    source_name: str,
) -> dict:
    """Ejecuta un experimento completo: preprocesamiento → train → predict → eval."""
    exp_name = f"{model.name} | {source_name} | {'preproc' if preprocess else 'crudo'}"
    print(f"\n{'='*60}")
    print(f"  {exp_name}")
    print(f"{'='*60}")

    # Preparar textos
    if preprocess:
        train_texts = [preprocess_ocr_text(d["text"]) for d in train_data]
        test_texts = [preprocess_ocr_text(d["text"]) for d in test_data]
    else:
        train_texts = [d["text"] for d in train_data]
        test_texts = [d["text"] for d in test_data]

    train_labels = [d["label"] for d in train_data]
    test_labels = [d["label"] for d in test_data]

    # Entrenar
    t0 = time.time()
    if model.needs_training:
        print("  Entrenando...")
        model.train(train_texts, train_labels)
    train_time = time.time() - t0

    # Predecir
    t0 = time.time()
    predictions = model.predict_batch(test_texts)
    predict_time = time.time() - t0

    # Evaluar
    metrics = compute_metrics(predictions, test_labels)
    metrics["train_time_s"] = round(train_time, 2)
    metrics["predict_time_s"] = round(predict_time, 2)
    metrics["experiment"] = exp_name
    metrics["model"] = model.name
    metrics["source"] = source_name
    metrics["preprocess"] = "Sí" if preprocess else "No"

    print(f"  Exact Match:      {metrics['exact_match']:.3f}")
    print(f"  Match ±1%:        {metrics['match_1pct']:.3f}")
    print(f"  Match ±5%:        {metrics['match_5pct']:.3f}")
    print(f"  Prediction Rate:  {metrics['prediction_rate']:.3f}")
    print(f"  MRE:              {metrics['mean_rel_error']:.4f}")
    print(f"  Tiempo train:     {metrics['train_time_s']}s | predict: {metrics['predict_time_s']}s")

    return metrics


# ============================================================
# 10. COMPARACIÓN Y VISUALIZACIÓN
# ============================================================

def print_comparison_table(results: list[dict]):
    """Imprime tabla comparativa ordenada por exact_match descendente."""
    df = pd.DataFrame(results)
    df = df.sort_values("exact_match", ascending=False).reset_index(drop=True)

    print("\n")
    print("=" * 100)
    print("  COMPARATIVA DE MODELOS — EXTRACCIÓN DEL TOTAL EN FACTURAS OCR")
    print("=" * 100)

    cols = [
        ("model", "Modelo", "20s"),
        ("source", "Datos", "10s"),
        ("preprocess", "Preproc", "8s"),
        ("prediction_rate", "Pred%", ".3f"),
        ("exact_match", "EM", ".3f"),
        ("match_1pct", "±1%", ".3f"),
        ("match_5pct", "±5%", ".3f"),
        ("match_10pct", "±10%", ".3f"),
        ("mean_rel_error", "MRE", ".4f"),
        ("train_time_s", "Train(s)", ".1f"),
    ]

    header = " | ".join(f"{label:>{w[-1].replace('.','').replace('f','').replace('s','')}s}"
                        if 's' in w else f"{label:>8s}"
                        for _, label, w in cols)
    # Imprimir cabecera limpia
    header_parts = []
    for _, label, _ in cols:
        header_parts.append(f"{label:>10s}")
    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))

    for _, row in df.iterrows():
        parts = []
        for col, _, fmt in cols:
            val = row[col]
            if isinstance(val, float):
                parts.append(f"{val:>10{fmt}}")
            else:
                parts.append(f"{str(val):>10s}")
        print(" | ".join(parts))

    print("=" * len(header))
    print()

    # Mejor modelo
    best = df.iloc[0]
    print(f">>> Mejor modelo: {best['model']} | {best['source']} | preproc={best['preprocess']}")
    print(f"    Exact Match = {best['exact_match']:.3f}  |  ±1% = {best['match_1pct']:.3f}  |  MRE = {best['mean_rel_error']:.4f}")
    print()

    return df


def plot_comparison(results: list[dict], save_path: Path | None = None):
    """Genera gráficas de comparación."""
    df = pd.DataFrame(results)
    df["label"] = df["model"] + "\n" + df["source"] + "\n" + df["preprocess"]
    df = df.sort_values("exact_match", ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, max(6, len(df) * 0.4)))

    # 1. Exact Match
    ax = axes[0]
    colors = ["#2196F3" if s == "EasyOCR" else "#4CAF50" for s in df["source"]]
    bars = ax.barh(df["label"], df["exact_match"], color=colors, edgecolor="white")
    ax.set_xlabel("Exact Match")
    ax.set_title("Exact Match (mayor = mejor)")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, df["exact_match"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    # 2. Match ±5%
    ax = axes[1]
    bars = ax.barh(df["label"], df["match_5pct"], color=colors, edgecolor="white")
    ax.set_xlabel("Match ±5%")
    ax.set_title("Accuracy ±5% (mayor = mejor)")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, df["match_5pct"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    # 3. Mean Relative Error
    ax = axes[2]
    bars = ax.barh(df["label"], df["mean_rel_error"], color=colors, edgecolor="white")
    ax.set_xlabel("Mean Relative Error")
    ax.set_title("MRE (menor = mejor)")
    for bar, val in zip(bars, df["mean_rel_error"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="EasyOCR"),
        Patch(facecolor="#4CAF50", label="CORD GT"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfica guardada en: {save_path}")
    plt.show()


def plot_preprocessing_impact(results: list[dict], save_path: Path | None = None):
    """Gráfica del impacto del preprocesamiento por modelo."""
    df = pd.DataFrame(results)
    df_ocr = df[df["source"] == "EasyOCR"].copy()

    models = df_ocr["model"].unique()
    raw_em = []
    pre_em = []
    model_names = []

    for m in models:
        sub = df_ocr[df_ocr["model"] == m]
        r = sub[sub["preprocess"] == "No"]["exact_match"].values
        p = sub[sub["preprocess"] == "Sí"]["exact_match"].values
        if len(r) > 0 and len(p) > 0:
            model_names.append(m)
            raw_em.append(r[0])
            pre_em.append(p[0])

    if not model_names:
        return

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, raw_em, width, label="Sin preprocesamiento", color="#FF7043")
    ax.bar(x + width / 2, pre_em, width, label="Con preprocesamiento", color="#42A5F5")
    ax.set_ylabel("Exact Match")
    ax.set_title("Impacto del preprocesamiento OCR por modelo")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfica guardada en: {save_path}")
    plt.show()


# ============================================================
# 11. MAIN
# ============================================================

def build_classical_models():
    """Instancia todos los modelos clásicos (no requieren GPU)."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import LinearSVC

    return [
        RegexKeywordModel(),
        RegexMaxFilteredModel(),
        CandidateMLModel(
            RandomForestClassifier(n_estimators=200, max_depth=10, random_state=SEED),
            "Random Forest",
        ),
        CandidateMLModel(
            GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=SEED),
            "Gradient Boosting",
        ),
        CandidateMLModel(
            LinearSVC(C=1.0, max_iter=5000, random_state=SEED),
            "SVM",
        ),
        TFIDFContextModel(),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Comparativa de modelos para extracción de totales en facturas OCR"
    )
    parser.add_argument(
        "--skip-transformers",
        action="store_true",
        help="Omitir modelos DistilBERT y T5 (más rápido)",
    )
    parser.add_argument(
        "--skip-cord-gt",
        action="store_true",
        help="Omitir carga de datos CORD ground truth (sin internet)",
    )
    parser.add_argument(
        "--cord-limit",
        type=int,
        default=None,
        help="Límite de muestras CORD GT a cargar (por defecto: todas)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directorio para guardar resultados y gráficas",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Carga de datos ---
    print("\n>>> Cargando datos JSONL (EasyOCR)...")
    train_jsonl = load_jsonl(DATA_DIR / "cord_training_data.jsonl")
    test_jsonl = load_jsonl(DATA_DIR / "cord_test_data.jsonl")
    print(f"    Train: {len(train_jsonl)} muestras  |  Test: {len(test_jsonl)} muestras")

    cord_gt_train, cord_gt_test = None, None
    if not args.skip_cord_gt:
        try:
            print("\n>>> Cargando datos CORD ground truth desde HuggingFace...")
            cord_gt_train = load_cord_gt("train", limit=args.cord_limit)
            cord_gt_test = load_cord_gt("test", limit=args.cord_limit)
            print(f"    GT Train: {len(cord_gt_train)} muestras  |  GT Test: {len(cord_gt_test)} muestras")
        except Exception as e:
            print(f"    No se pudieron cargar datos CORD GT: {e}")
            print("    Continuando solo con datos JSONL...")

    # --- Construir modelos ---
    all_results: list[dict] = []

    # Modelos clásicos
    classical_models = build_classical_models()

    # Modelos transformer (opcionales)
    transformer_models: list[BaseModel] = []
    if not args.skip_transformers:
        transformer_models = [
            DistilBERTReranker(epochs=3, batch_size=16),
            T5SeqModel(epochs=10, batch_size=8),
        ]

    all_models = classical_models + transformer_models

    # --- Experimentos con datos EasyOCR ---
    for preprocess in [False, True]:
        for model in all_models:
            # Crear instancia fresca para modelos con estado
            if model.needs_training:
                model = _fresh_model(model)
            result = run_experiment(
                model, train_jsonl, test_jsonl, preprocess, source_name="EasyOCR"
            )
            all_results.append(result)

    # --- Experimentos con CORD Ground Truth ---
    if cord_gt_train and cord_gt_test:
        # Solo modelos clásicos con datos GT (el preprocesamiento no aplica)
        gt_models = build_classical_models()
        for model in gt_models:
            result = run_experiment(
                model, cord_gt_train, cord_gt_test, preprocess=False, source_name="CORD GT"
            )
            all_results.append(result)

    # --- Resultados ---
    df_results = print_comparison_table(all_results)

    # Guardar CSV
    csv_path = output_dir / "comparativa_modelos.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Resultados guardados en: {csv_path}")

    # Gráficas
    plot_comparison(all_results, save_path=output_dir / "comparativa_general.png")
    plot_preprocessing_impact(all_results, save_path=output_dir / "impacto_preprocesamiento.png")

    # --- Resumen por categoría ---
    print_summary(all_results)


def _fresh_model(model: BaseModel) -> BaseModel:
    """Crea una instancia fresca de un modelo con estado (para evitar contaminación)."""
    if isinstance(model, CandidateMLModel):
        from sklearn.base import clone
        return CandidateMLModel(clone(model.clf), model.name.replace("Candidate ", ""))
    if isinstance(model, TFIDFContextModel):
        return TFIDFContextModel(window=model.window)
    if isinstance(model, DistilBERTReranker):
        return DistilBERTReranker(epochs=model.epochs, batch_size=model.batch_size, lr=model.lr)
    if isinstance(model, T5SeqModel):
        return T5SeqModel(epochs=model.epochs, batch_size=model.batch_size, lr=model.lr)
    return model


def print_summary(results: list[dict]):
    """Imprime un resumen analítico de los resultados."""
    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("  RESUMEN ANALÍTICO")
    print("=" * 80)

    # 1. Impacto del preprocesamiento
    df_ocr = df[df["source"] == "EasyOCR"]
    print("\n--- Impacto del preprocesamiento (datos EasyOCR) ---")
    for model_name in df_ocr["model"].unique():
        sub = df_ocr[df_ocr["model"] == model_name]
        raw = sub[sub["preprocess"] == "No"]["exact_match"].values
        pre = sub[sub["preprocess"] == "Sí"]["exact_match"].values
        if len(raw) > 0 and len(pre) > 0:
            delta = pre[0] - raw[0]
            arrow = "+" if delta >= 0 else ""
            print(f"  {model_name:30s}  crudo={raw[0]:.3f}  preproc={pre[0]:.3f}  ({arrow}{delta:.3f})")

    # 2. Impacto de la fuente de datos
    df_gt = df[df["source"] == "CORD GT"]
    if len(df_gt) > 0:
        print("\n--- Impacto del ground truth vs EasyOCR (sin preproc) ---")
        df_ocr_raw = df_ocr[df_ocr["preprocess"] == "No"]
        for model_name in df_gt["model"].unique():
            gt_em = df_gt[df_gt["model"] == model_name]["exact_match"].values
            ocr_em = df_ocr_raw[df_ocr_raw["model"] == model_name]["exact_match"].values
            if len(gt_em) > 0 and len(ocr_em) > 0:
                delta = gt_em[0] - ocr_em[0]
                print(f"  {model_name:30s}  OCR={ocr_em[0]:.3f}  GT={gt_em[0]:.3f}  (+{delta:.3f})")

    # 3. Ranking por tipo de modelo
    print("\n--- Ranking de modelos (mejor EM por modelo, cualquier config) ---")
    best_per_model = df.groupby("model")["exact_match"].max().sort_values(ascending=False)
    for i, (name, em) in enumerate(best_per_model.items(), 1):
        print(f"  {i}. {name:30s}  EM = {em:.3f}")

    print()


if __name__ == "__main__":
    main()

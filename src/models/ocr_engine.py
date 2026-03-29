"""
Motor OCR para extracción de totales en facturas.

Pipeline: PaddleOCR → preprocesamiento texto → Gradient Boosting.
Si PaddleOCR no está disponible, usa EasyOCR como fallback.
"""

import os
import re
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
import joblib

from src.features.region_detector import SimpleRegionDetector
from src.utils.image_processing import preprocess_image_for_ocr

logger = logging.getLogger(__name__)

# Ruta al modelo GB (relativa a la raíz del proyecto)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MODEL_PATHS = [
    os.path.join(_PROJECT_ROOT, "models", "mobile_pipeline_gb_np1.joblib"),
    os.path.join(_PROJECT_ROOT, "models", "mobile_pipeline_gb.joblib"),
]

# ── Funciones de extracción (importadas de compare_extraction_models) ──
# Se reimplementan aquí para no depender de scripts/ en producción.

_TOTAL_KEYWORDS = re.compile(
    r"\b(grand\s*total|total|totai|tota[l1i]|gnd\s*tota[l1i]|"
    r"due|amount|amt|tl|rounding)\b",
    re.IGNORECASE,
)
_NEGATIVE_KEYWORDS = re.compile(
    r"\b(cash|change|changed|kembal[i1]|tunai|debit|credit|card|"
    r"bca|tendered|bayar|paid|pago|efectivo)\b",
    re.IGNORECASE,
)
_SUBTOTAL_KEYWORDS = re.compile(
    r"\b(sub\s*-?\s*total|subtotal|subttl|sub\s*ttl|svc|service|"
    r"tax|pajak|pb[1i]|disc|discount)\b",
    re.IGNORECASE,
)

_OCR_DIGIT = r"[\dOoQq]"
_CANDIDATE_PATTERNS = [
    rf"{_OCR_DIGIT}{{1,3}}(?:[.,]\s?{_OCR_DIGIT}{{3}})+(?:[.,]{_OCR_DIGIT}{{1,2}})?",
    r"\d{1,3}(?:\s\d{3})+",
    rf"{_OCR_DIGIT}{{4,10}}",
]
_CANDIDATE_RE = re.compile("(" + "|".join(_CANDIDATE_PATTERNS) + ")")


def _normalize_amount(s: str) -> Optional[float]:
    s = re.sub(r"[^\d.,]", "", s)
    if not s or not re.search(r"\d", s):
        return None
    if "." in s and "," in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) <= 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _preprocess_ocr_text(text: str) -> str:
    text = re.sub(r"(?<=\d)[OoQq]+", lambda m: "0" * len(m.group()), text)
    text = re.sub(r"[OoQq]+(?=\d)", lambda m: "0" * len(m.group()), text)
    text = re.sub(
        r"(?<=[\d,.])[OoQq]{2,3}(?=[\s,.\-;:)\]|$])",
        lambda m: "0" * len(m.group()), text,
    )
    text = re.sub(r"(?<=\d)[Dd](?=[\dOo0])", "0", text)
    text = re.sub(r"(?<=\d)[Il](?=[\d])", "1", text)
    text = re.sub(r"[Il](?=\d{2,})", "1", text)
    text = text.lower()
    text = re.sub(r"[{}\[\]<>~`|\\^\"']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_candidates(text: str) -> list:
    candidates = []
    for m in _CANDIDATE_RE.finditer(text):
        raw = m.group(1)
        if not re.search(r"\d", raw):
            continue
        cleaned = re.sub(r"[OoQq]", "0", raw).replace(" ", "")
        value = _normalize_amount(cleaned)
        if value is None or value < 100:
            continue
        start, end = m.start(1), m.end(1)
        candidates.append(dict(
            raw=raw, value=value, start=start, end=end,
            context_before=text[max(0, start - 60):start],
            context_after=text[end:end + 60],
        ))
    return candidates


def _keyword_distance(text, candidate, pattern):
    text_len = max(len(text), 1)
    mid = (candidate["start"] + candidate["end"]) / 2
    best = text_len
    for m in pattern.finditer(text):
        km = (m.start() + m.end()) / 2
        best = min(best, abs(mid - km))
    return best / text_len


def _candidate_features(text, candidates, idx):
    c = candidates[idx]
    text_len = max(len(text), 1)
    all_values = [cc["value"] for cc in candidates]
    max_val = max(all_values) if all_values else 1.0
    ctx = c["context_before"] + " " + c["context_after"]
    neg_match = _NEGATIVE_KEYWORDS.search(text[c["end"]:])
    is_last_before_neg = 0.0
    if neg_match:
        between = text[c["end"]:c["end"] + neg_match.start()]
        if not _CANDIDATE_RE.search(between):
            is_last_before_neg = 1.0
    return np.array([
        c["end"] / text_len,
        np.log1p(c["value"]),
        c["value"] / max_val if max_val > 0 else 0.0,
        float(c["value"] == max_val),
        _keyword_distance(text, c, _TOTAL_KEYWORDS),
        _keyword_distance(text, c, _NEGATIVE_KEYWORDS),
        _keyword_distance(text, c, _SUBTOTAL_KEYWORDS),
        float(bool(_TOTAL_KEYWORDS.search(ctx))),
        float(bool(_NEGATIVE_KEYWORDS.search(ctx))),
        float(bool(_SUBTOTAL_KEYWORDS.search(ctx))),
        len(candidates),
        is_last_before_neg,
    ], dtype=np.float32)


# ===================================================================

class OptimizedOCREngine:
    """Motor OCR: PaddleOCR (o EasyOCR) + Gradient Boosting."""

    def __init__(self, use_gpu: bool = False):
        self._ocr = None
        self._ocr_type = None
        self._gb_clf = None
        self._scaler = None

        # 1 — Intentar PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            self._ocr_type = "paddle"
            logger.info("PaddleOCR inicializado.")
        except Exception as e:
            logger.warning(f"PaddleOCR no disponible ({e}), usando EasyOCR.")
            try:
                import easyocr
                self._ocr = easyocr.Reader(["es", "en"], gpu=use_gpu)
                self._ocr_type = "easyocr"
                logger.info("EasyOCR inicializado.")
            except Exception as e2:
                logger.error(f"No se pudo inicializar ningun OCR: {e2}")

        # 2 — Cargar modelo GB
        self._load_gb_model()

    def _load_gb_model(self):
        for path in _MODEL_PATHS:
            if os.path.isfile(path):
                try:
                    data = joblib.load(path)
                    self._gb_clf = data["gb_clf"]
                    self._scaler = data["scaler"]
                    logger.info(f"Modelo GB cargado: {path}")
                    return
                except Exception as e:
                    logger.warning(f"No se pudo cargar {path}: {e}")

        # Entrenar desde JSONL si existe
        jsonl_path = os.path.join(_PROJECT_ROOT, "data", "processed", "cord_training_data.jsonl")
        if os.path.isfile(jsonl_path):
            logger.info("Entrenando GB desde JSONL...")
            self._train_gb_from_jsonl(jsonl_path)
            return

        # Entrenar desde CORD directamente
        logger.info("Entrenando GB desde CORD (HuggingFace)...")
        self._train_gb_from_cord()

    def _train_gb_from_jsonl(self, jsonl_path: str):
        import json
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler

        data = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))

        X, y = self._build_training_data(data)
        if len(X) == 0:
            logger.error("No se generaron features de entrenamiento.")
            return

        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X)
        self._gb_clf = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, random_state=42
        )
        self._gb_clf.fit(X, y)
        self._save_model()

    def _train_gb_from_cord(self):
        from datasets import load_dataset
        import json
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler

        ds = load_dataset("naver-clova-ix/cord-v2", split="train")
        data = []
        for i in range(len(ds)):
            try:
                sample = ds[i]
                gt = json.loads(sample["ground_truth"])
                tp = gt.get("gt_parse", {}).get("total", {}).get("total_price")
                if not tp:
                    continue
                img = np.array(sample["image"])
                text = self._run_ocr_on_image(img)
                data.append({"text": text, "label": tp})
                if len(data) % 50 == 0:
                    logger.info(f"  Entrenamiento: {len(data)} muestras...")
            except Exception:
                continue

        X, y = self._build_training_data(data)
        if len(X) == 0:
            logger.error("No se generaron features de entrenamiento.")
            return

        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X)
        self._gb_clf = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, random_state=42
        )
        self._gb_clf.fit(X, y)
        self._save_model()

    def _build_training_data(self, data: list):
        X, y = [], []
        for d in data:
            lv = _normalize_amount(str(d["label"]))
            if lv is None or lv == 0:
                continue
            text = _preprocess_ocr_text(d["text"])
            cands = _extract_candidates(text)
            if not cands:
                continue
            best_idx, best_diff = 0, float("inf")
            for i, c in enumerate(cands):
                diff = abs(c["value"] - lv) / max(lv, 1)
                if diff < best_diff:
                    best_idx, best_diff = i, diff
            for i in range(len(cands)):
                X.append(_candidate_features(text, cands, i))
                y.append(1 if i == best_idx else 0)
        return np.array(X, dtype=np.float32) if X else np.array([]), np.array(y)

    def _save_model(self):
        path = _MODEL_PATHS[0]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"gb_clf": self._gb_clf, "scaler": self._scaler}, path)
        logger.info(f"Modelo GB guardado: {path}")

    # ── OCR ──

    def _run_ocr_on_image(self, image: np.ndarray) -> str:
        """Ejecuta OCR sobre una imagen y devuelve texto plano."""
        if self._ocr_type == "paddle":
            result = self._ocr.ocr(image, cls=True)
            lines = result[0] if result and result[0] else []
            return " ".join(line[1][0] for line in lines)
        elif self._ocr_type == "easyocr":
            results = self._ocr.readtext(image, detail=0)
            return " ".join(results)
        return ""

    def process_and_extract(
        self, original_img: np.ndarray
    ) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, np.ndarray]]:
        """Segmenta, preprocesa y ejecuta OCR (compatibilidad con pipeline anterior)."""
        detector = SimpleRegionDetector()
        crops = detector.crop_regions(original_img)
        extracted_data = {}
        raw_lists = {}
        preprocessed_crops = {}

        for region_name, crop in crops.items():
            clean_crop = preprocess_image_for_ocr(crop)
            preprocessed_crops[region_name] = clean_crop
            text = self._run_ocr_on_image(clean_crop)
            raw_lists[region_name] = text.split() if text else []
            extracted_data[region_name] = text

        return extracted_data, raw_lists, preprocessed_crops

    # ── Extracción de total ──

    def find_total_value(self, raw_text_input) -> Optional[float]:
        """Extrae el total de una factura usando GB (o regex como fallback).

        Args:
            raw_text_input: str con el texto OCR, o List[str] (compatibilidad).
        """
        if isinstance(raw_text_input, list):
            text = " ".join(raw_text_input)
        else:
            text = raw_text_input

        if not text or not text.strip():
            return None

        preprocessed = _preprocess_ocr_text(text)
        candidates = _extract_candidates(preprocessed)

        if not candidates:
            return None

        # GB scoring
        if self._gb_clf is not None and self._scaler is not None:
            X = np.array(
                [_candidate_features(preprocessed, candidates, i)
                 for i in range(len(candidates))],
                dtype=np.float32,
            )
            X = self._scaler.transform(X)
            probs = self._gb_clf.predict_proba(X)
            scores = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            best = candidates[int(np.argmax(scores))]
            return best["value"]

        # Fallback: regex keyword
        for m in _TOTAL_KEYWORDS.finditer(preprocessed):
            kw_end = m.end()
            for c in candidates:
                if c["start"] >= kw_end - 5:
                    return c["value"]

        return max(candidates, key=lambda c: c["value"])["value"] if candidates else None

    def extract_total(self, image: np.ndarray) -> Optional[float]:
        """Pipeline completo: imagen → OCR → preprocesamiento → GB → total."""
        text = self._run_ocr_on_image(image)
        return self.find_total_value(text)

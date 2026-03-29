"""
Test de rendimiento: PaddleOCR + Gradient Boosting.

Mide el tiempo total desde que se recibe la imagen hasta la predicción final:
  imagen → PaddleOCR → preprocesamiento → GB → resultado

Uso (desde P3_AP-IA/):
  .venv_paddle/Scripts/python tests/test_paddleocr_gb_timing.py
  .venv_paddle/Scripts/python tests/test_paddleocr_gb_timing.py --num-samples 20
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.compare_extraction_models import (
    normalize_amount,
    preprocess_ocr_text,
    extract_number_candidates,
    candidate_features,
    _find_best_match,
    load_jsonl,
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

MODEL_PATH = os.path.join("models", "mobile_pipeline_gb.joblib")
COMPAT_MODEL_PATH = os.path.join("models", "mobile_pipeline_gb_np1.joblib")


def _train_gb_from_jsonl():
    """Entrena GB desde JSONL (rápido, sin OCR)."""
    train_data = load_jsonl(os.path.join("data", "processed", "cord_training_data.jsonl"))
    X, y = [], []
    for d in train_data:
        label_val = normalize_amount(str(d["label"]))
        if label_val is None or label_val == 0:
            continue
        text = preprocess_ocr_text(d["text"])
        cands = extract_number_candidates(text)
        if not cands:
            continue
        best_idx = _find_best_match(cands, label_val)
        for i in range(len(cands)):
            X.append(candidate_features(text, cands, i))
            y.append(1 if i == best_idx else 0)
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
    clf.fit(X, y)
    return clf, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--model-path", default=MODEL_PATH)
    args = parser.parse_args()

    # ── 1. Cargar modelo (intentar compat primero, luego original, luego entrenar) ──
    gb_clf, scaler = None, None
    for path in [COMPAT_MODEL_PATH, args.model_path]:
        if os.path.isfile(path):
            try:
                data = joblib.load(path)
                gb_clf, scaler = data["gb_clf"], data["scaler"]
                print(f"Modelo cargado: {path}\n")
                break
            except Exception as e:
                print(f"No se pudo cargar {path}: {e}")

    if gb_clf is None:
        print("Re-entrenando GB desde JSONL...")
        gb_clf, scaler = _train_gb_from_jsonl()
        os.makedirs(os.path.dirname(COMPAT_MODEL_PATH) or ".", exist_ok=True)
        joblib.dump({"gb_clf": gb_clf, "scaler": scaler}, COMPAT_MODEL_PATH)
        print(f"Modelo guardado: {COMPAT_MODEL_PATH}\n")

    # ── 2. Inicializar PaddleOCR ──
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    print("PaddleOCR inicializado\n")

    # ── 3. Cargar imágenes CORD test (sin streaming — evita dependencia de torch) ──
    from datasets import load_dataset
    ds = load_dataset("naver-clova-ix/cord-v2", split="test")
    samples = []
    for i in range(len(ds)):
        if len(samples) >= args.num_samples:
            break
        sample = ds[i]
        gt = json.loads(sample["ground_truth"])
        tp = gt.get("gt_parse", {}).get("total", {}).get("total_price")
        if not tp:
            continue
        samples.append(dict(image=np.array(sample["image"]), label=tp, id=i))
    print(f"{len(samples)} muestras cargadas\n")

    # ── 4. Ejecutar pipeline completo por muestra ──
    print(f"{'#':>3}  {'GT':>12}  {'Pred':>12}  {'OCR':>8}  {'Prep':>7}  {'GB':>7}  {'Total':>8}  OK")
    print("-" * 82)

    times_ocr, times_prep, times_gb, times_total = [], [], [], []
    n_exact, n_approx, n_pred = 0, 0, 0

    for s in samples:
        t_total_start = time.perf_counter()

        # OCR
        t = time.perf_counter()
        result = ocr.ocr(s["image"], cls=True)
        t_ocr = time.perf_counter() - t
        lines = result[0] if result and result[0] else []
        raw_text = " ".join(line[1][0] for line in lines)

        # Preprocesamiento
        t = time.perf_counter()
        text = preprocess_ocr_text(raw_text)
        t_prep = time.perf_counter() - t

        # GB
        t = time.perf_counter()
        candidates = extract_number_candidates(text)
        pred = None
        if candidates:
            X = np.array(
                [candidate_features(text, candidates, i) for i in range(len(candidates))],
                dtype=np.float32,
            )
            X = scaler.transform(X)
            probs = gb_clf.predict_proba(X)
            scores = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            pred = candidates[int(np.argmax(scores))]["raw"]
        t_gb = time.perf_counter() - t

        t_total = time.perf_counter() - t_total_start

        # Métricas
        pred_val = normalize_amount(str(pred)) if pred else None
        gt_val = normalize_amount(str(s["label"]))
        exact = False
        if pred_val is not None:
            n_pred += 1
        if pred_val and gt_val and gt_val != 0:
            rel = abs(pred_val - gt_val) / gt_val
            exact = rel < 1e-6
            if exact:
                n_exact += 1
            if rel <= 0.05:
                n_approx += 1

        mark = "OK" if exact else " X"
        print(
            f"{s['id']:>3}  {s['label']:>12}  {str(pred or '-'):>12}  "
            f"{t_ocr*1000:>7.0f}ms  {t_prep*1000:>5.1f}ms  {t_gb*1000:>5.1f}ms  "
            f"{t_total*1000:>7.0f}ms  {mark}"
        )

        times_ocr.append(t_ocr * 1000)
        times_prep.append(t_prep * 1000)
        times_gb.append(t_gb * 1000)
        times_total.append(t_total * 1000)

    # ── 5. Resumen ──
    n = len(samples)
    print("\n" + "=" * 50)
    print("  RESUMEN")
    print("=" * 50)
    print(f"\n  Accuracy:")
    print(f"    Exact Match:   {n_exact}/{n}  ({n_exact/n*100:.1f}%)")
    print(f"    Match +-5%%:    {n_approx}/{n}  ({n_approx/n*100:.1f}%)")
    print(f"    Pred. Rate:    {n_pred}/{n}  ({n_pred/n*100:.1f}%)")
    print(f"\n  Tiempos medios:")
    print(f"    PaddleOCR:     {np.mean(times_ocr):>8.0f} ms")
    print(f"    Preproc:       {np.mean(times_prep):>8.1f} ms")
    print(f"    GB:            {np.mean(times_gb):>8.1f} ms")
    print(f"    TOTAL:         {np.mean(times_total):>8.0f} ms")
    print(f"\n  Tiempos min/max (total):")
    print(f"    Min:           {np.min(times_total):>8.0f} ms")
    print(f"    Max:           {np.max(times_total):>8.0f} ms")
    print(f"    Mediana:       {np.median(times_total):>8.0f} ms")
    print("=" * 50)


if __name__ == "__main__":
    main()

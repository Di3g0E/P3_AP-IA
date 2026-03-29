"""
Test comparativo: PaddleOCR vs EasyOCR sobre CORD test.

Para cada muestra ejecuta ambos OCRs y evalúa dos métodos de extracción:
  - Baseline (Regex): primer número tras keyword "total"
  - Gradient Boosting: scoring de candidatos con modelo entrenado

Requiere el entorno .venv_paddle (Python 3.12 + paddleocr).

Uso:
  # Desde P3_AP-IA/:
  .venv_paddle/Scripts/python tests/test_paddleocr_vs_easyocr.py --num-samples 10
  .venv_paddle/Scripts/python tests/test_paddleocr_vs_easyocr.py --num-samples 1000
"""

import argparse
import json
import os
import sys
import time

# Fix torch DLL loading en rutas OneDrive con espacios (Windows)
try:
    import importlib.util as _ilu
    _ts = _ilu.find_spec("torch")
    if _ts and _ts.origin:
        _tl = os.path.join(os.path.dirname(_ts.origin), "lib")
        if os.path.isdir(_tl):
            os.add_dll_directory(os.path.abspath(_tl))
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from scripts.compare_extraction_models import (
    normalize_amount,
    preprocess_ocr_text,
    extract_number_candidates,
    candidate_features,
    _find_best_match,
    TOTAL_KEYWORDS,
)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

DEFAULT_MODEL_PATH = os.path.join("models", "mobile_pipeline_gb.joblib")


# ===================================================================
#  OCR ENGINES
# ===================================================================

def init_paddleocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


def init_easyocr():
    import easyocr
    return easyocr.Reader(["es", "en"], gpu=False)


def run_paddleocr(ocr, image: np.ndarray) -> tuple[str, float]:
    """Devuelve (texto, tiempo_ms)."""
    t = time.perf_counter()
    result = ocr.ocr(image, cls=True)
    ms = (time.perf_counter() - t) * 1000
    lines = result[0] if result and result[0] else []
    text = " ".join(line[1][0] for line in lines)
    return text, ms


def run_easyocr(reader, image: np.ndarray) -> tuple[str, float]:
    """Devuelve (texto, tiempo_ms)."""
    t = time.perf_counter()
    results = reader.readtext(image)
    ms = (time.perf_counter() - t) * 1000
    text = " ".join(r[1] for r in results)
    return text, ms


# ===================================================================
#  PREDICCIÓN
# ===================================================================

def baseline_predict(text: str) -> str | None:
    """Primer candidato numérico tras keyword 'total'."""
    preprocessed = preprocess_ocr_text(text)
    candidates = extract_number_candidates(preprocessed)
    if not candidates:
        return None

    last_kw_end = -1
    for m in TOTAL_KEYWORDS.finditer(preprocessed):
        last_kw_end = m.end()

    if last_kw_end == -1:
        return None

    for c in candidates:
        if c["start"] >= last_kw_end - 5:
            return c["raw"]
    return None


def gb_predict(gb_clf, scaler, text: str) -> str | None:
    """Gradient Boosting solo sobre texto preprocesado."""
    preprocessed = preprocess_ocr_text(text)
    candidates = extract_number_candidates(preprocessed)
    if not candidates:
        return None

    X = np.array(
        [candidate_features(preprocessed, candidates, i) for i in range(len(candidates))],
        dtype=np.float32,
    )
    X = scaler.transform(X)
    probs = gb_clf.predict_proba(X)
    scores = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    best = int(np.argmax(scores))
    return candidates[best]["raw"]


# ===================================================================
#  CARGA DE DATOS
# ===================================================================

def load_cord_test(num_samples: int = 10):
    """Descarga imágenes + ground truth de CORD test."""
    from datasets import load_dataset

    ds = load_dataset("naver-clova-ix/cord-v2", split="test", streaming=True)
    samples = []
    for i, sample in enumerate(ds):
        if len(samples) >= num_samples:
            break
        try:
            gt = json.loads(sample["ground_truth"])
            tp = gt.get("gt_parse", {}).get("total", {}).get("total_price")
            if not tp:
                continue
            samples.append(dict(
                image=np.array(sample["image"]),
                label=tp,
                image_id=i,
            ))
        except Exception:
            continue
    return samples


# ===================================================================
#  MÉTRICAS
# ===================================================================

def _eval(pred_raw, gt_label):
    pred_val = normalize_amount(str(pred_raw)) if pred_raw else None
    gt_val = normalize_amount(str(gt_label))
    rel_err = None
    exact = False
    if pred_val is not None and gt_val and gt_val != 0:
        rel_err = abs(pred_val - gt_val) / gt_val
        exact = rel_err < 1e-6
    return dict(
        prediction=pred_raw, rel_error=rel_err,
        exact=exact, has_pred=pred_raw is not None,
    )


def _print_summary(evals, name):
    n = len(evals)
    n_pred = sum(e["has_pred"] for e in evals)
    n_exact = sum(e["exact"] for e in evals)
    n_5 = sum(1 for e in evals if e["rel_error"] is not None and e["rel_error"] <= 0.05)
    n_10 = sum(1 for e in evals if e["rel_error"] is not None and e["rel_error"] <= 0.10)
    errors = [e["rel_error"] for e in evals if e["rel_error"] is not None]
    mre = np.mean(errors) if errors else float("nan")

    print(f"  {name}:")
    print(f"    Pred Rate:   {n_pred}/{n}  ({n_pred/n*100:.1f}%)")
    print(f"    Exact Match: {n_exact}/{n}  ({n_exact/n*100:.1f}%)")
    print(f"    Match +-5%%:  {n_5}/{n}  ({n_5/n*100:.1f}%)")
    print(f"    Match +-10%%: {n_10}/{n}  ({n_10/n*100:.1f}%)")
    print(f"    MRE:         {mre:.4f}")

    return dict(
        name=name, pred_rate=n_pred / n, exact=n_exact / n,
        match_5=n_5 / n, match_10=n_10 / n, mre=mre,
    )


# ===================================================================
#  VISUALIZACIÓN
# ===================================================================

def plot_results(stats: dict, ocr_times: dict, output_dir: str) -> str:
    """Gráfico comparativo 4 métodos (2 OCR x 2 extracción)."""
    methods = list(stats.keys())
    metrics = ["pred_rate", "exact", "match_5", "match_10"]
    metric_labels = ["Prediction\nRate", "Exact\nMatch", "Match\n+-5%", "Match\n+-10%"]

    colors = {
        "PaddleOCR + Regex":  "#42a5f5",
        "PaddleOCR + GB":     "#1565c0",
        "EasyOCR + Regex":    "#ffa726",
        "EasyOCR + GB":       "#e65100",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # --- Accuracy ---
    ax = axes[0]
    x = np.arange(len(metrics))
    n_methods = len(methods)
    w = 0.8 / n_methods

    for j, method in enumerate(methods):
        vals = [stats[method][m] for m in metrics]
        offset = (j - n_methods / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=method,
                      color=colors.get(method, "#90a4ae"), edgecolor="white")
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2, h + 0.01,
                        f"{h:.0%}", ha="center", fontsize=7.5, fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Ratio")
    ax.set_title("Accuracy por metodo", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    # --- Tiempos OCR ---
    ax = axes[1]
    ocr_names = list(ocr_times.keys())
    ocr_vals = [ocr_times[k] for k in ocr_names]
    ocr_colors = ["#42a5f5", "#ffa726"]
    bars = ax.bar(ocr_names, ocr_vals, color=ocr_colors, edgecolor="white", width=0.4)
    for b, v in zip(bars, ocr_vals):
        label = f"{v:.0f} ms" if v < 1000 else f"{v/1000:.1f} s"
        ax.text(b.get_x() + b.get_width() / 2, v + max(ocr_vals) * 0.03,
                label, ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("ms")
    ax.set_title("Tiempo medio OCR por imagen", fontweight="bold")

    fig.suptitle("PaddleOCR vs EasyOCR — CORD test", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "paddleocr_vs_easyocr.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ===================================================================
#  MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PaddleOCR vs EasyOCR sobre CORD test",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output-dir", default="results/paddleocr_vs_easyocr")
    args = parser.parse_args()

    print("=" * 70)
    print("  PaddleOCR vs EasyOCR — CORD test")
    print("  Metodos: Baseline (Regex) y Gradient Boosting")
    print("=" * 70)

    # ── 1. Cargar o entrenar modelo GB ──
    COMPAT_MODEL_PATH = os.path.join("models", "mobile_pipeline_gb_np1.joblib")
    gb_clf, scaler = None, None

    # Intentar cargar modelo compatible con numpy<2
    if os.path.isfile(COMPAT_MODEL_PATH):
        try:
            data = joblib.load(COMPAT_MODEL_PATH)
            gb_clf = data["gb_clf"]
            scaler = data["scaler"]
            print(f"\n  Modelo GB cargado desde {COMPAT_MODEL_PATH}")
        except Exception as e:
            print(f"\n  AVISO: no se pudo cargar {COMPAT_MODEL_PATH} ({e})")

    # Si no hay modelo compatible, intentar el original
    if gb_clf is None and os.path.isfile(args.model_path):
        try:
            data = joblib.load(args.model_path)
            gb_clf = data["gb_clf"]
            scaler = data["scaler"]
            print(f"\n  Modelo GB cargado desde {args.model_path}")
        except Exception as e:
            print(f"\n  AVISO: no se pudo cargar {args.model_path} ({e})")
            print("  (probable incompatibilidad numpy 2.x vs 1.x)")

    # Si ninguno cargó, re-entrenar y guardar con nombre separado
    if gb_clf is None:
        print("\n>>> Re-entrenando GB desde CORD train (EasyOCR)...")
        print(">>> Inicializando EasyOCR para entrenamiento...")
        _reader = init_easyocr()
        from datasets import load_dataset as _ld
        _ds = _ld("naver-clova-ix/cord-v2", split="train", streaming=True)
        _texts, _labels = [], []
        for _i, _sample in enumerate(_ds):
            try:
                _gt = json.loads(_sample["ground_truth"])
                _tp = _gt.get("gt_parse", {}).get("total", {}).get("total_price")
                if not _tp:
                    continue
                _img = np.array(_sample["image"])
                _res = _reader.readtext(_img)
                _texts.append(" ".join(r[1] for r in _res))
                _labels.append(_tp)
                if len(_texts) % 50 == 0:
                    print(f"      {len(_texts)} muestras...")
            except Exception:
                continue
        print(f"    Train: {len(_texts)} muestras")

        X, y = [], []
        for _text, _label in zip(_texts, _labels):
            _lv = normalize_amount(str(_label))
            if _lv is None or _lv == 0:
                continue
            _prep = preprocess_ocr_text(_text)
            _cands = extract_number_candidates(_prep)
            if not _cands:
                continue
            _best = _find_best_match(_cands, _lv)
            for _j in range(len(_cands)):
                X.append(candidate_features(_prep, _cands, _j))
                y.append(1 if _j == _best else 0)

        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        gb_clf = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
        gb_clf.fit(X, y)
        print("    GB entrenado.")

        os.makedirs(os.path.dirname(COMPAT_MODEL_PATH) or ".", exist_ok=True)
        joblib.dump({"gb_clf": gb_clf, "scaler": scaler}, COMPAT_MODEL_PATH)
        print(f"    Modelo guardado en {COMPAT_MODEL_PATH}")

    # ── 2. Inicializar OCRs ──
    print(">>> Inicializando PaddleOCR...")
    paddle_ocr = init_paddleocr()

    print(">>> Inicializando EasyOCR...")
    easy_reader = init_easyocr()

    # ── 3. Cargar datos ──
    print(f"\n>>> Cargando {args.num_samples} muestras de CORD test...")
    samples = load_cord_test(num_samples=args.num_samples)
    print(f"    {len(samples)} muestras cargadas\n")

    # ── 4. Evaluar ──
    results = {
        "paddle_regex": [], "paddle_gb": [],
        "easy_regex": [], "easy_gb": [],
    }
    paddle_times = []
    easy_times = []

    header = (f"{'#':>4}  {'GT':>12}  "
              f"{'P-Regex':>12} {'P-GB':>12}  "
              f"{'E-Regex':>12} {'E-GB':>12}  "
              f"{'Paddle':>7} {'Easy':>7}")
    print(header)
    print("-" * len(header))

    for s in samples:
        img = s["image"]
        gt = s["label"]

        # OCR
        p_text, p_ms = run_paddleocr(paddle_ocr, img)
        e_text, e_ms = run_easyocr(easy_reader, img)
        paddle_times.append(p_ms)
        easy_times.append(e_ms)

        # Baseline (Regex)
        p_regex = baseline_predict(p_text)
        e_regex = baseline_predict(e_text)
        results["paddle_regex"].append(_eval(p_regex, gt))
        results["easy_regex"].append(_eval(e_regex, gt))

        # Gradient Boosting
        p_gb = gb_predict(gb_clf, scaler, p_text) if gb_clf else None
        e_gb = gb_predict(gb_clf, scaler, e_text) if gb_clf else None
        results["paddle_gb"].append(_eval(p_gb, gt))
        results["easy_gb"].append(_eval(e_gb, gt))

        # Marcas
        def mark(ev):
            if ev["exact"]:
                return "OK"
            if ev["rel_error"] is not None and ev["rel_error"] <= 0.05:
                return " ~"
            return " X"

        pr_m = mark(results["paddle_regex"][-1])
        pg_m = mark(results["paddle_gb"][-1])
        er_m = mark(results["easy_regex"][-1])
        eg_m = mark(results["easy_gb"][-1])

        print(
            f"{s['image_id']:>4}  {gt:>12}  "
            f"{str(p_regex or '-'):>10}{pr_m} "
            f"{str(p_gb or '-'):>10}{pg_m}  "
            f"{str(e_regex or '-'):>10}{er_m} "
            f"{str(e_gb or '-'):>10}{eg_m}  "
            f"{p_ms:>6.0f}ms {e_ms:>6.0f}ms"
        )

    # ── 5. Resumen ──
    print("\n" + "=" * 70)
    print("  RESUMEN")
    print("=" * 70)

    all_stats = {}
    for key, label in [
        ("paddle_regex", "PaddleOCR + Regex"),
        ("paddle_gb",    "PaddleOCR + GB"),
        ("easy_regex",   "EasyOCR + Regex"),
        ("easy_gb",      "EasyOCR + GB"),
    ]:
        print()
        all_stats[label] = _print_summary(results[key], label)

    print(f"\n  Tiempos OCR medios:")
    print(f"    PaddleOCR:  {np.mean(paddle_times):.0f} ms")
    print(f"    EasyOCR:    {np.mean(easy_times):.0f} ms")

    # ── 6. Gráfico ──
    ocr_times = {"PaddleOCR": np.mean(paddle_times), "EasyOCR": np.mean(easy_times)}
    out = plot_results(all_stats, ocr_times, args.output_dir)
    print(f"\n  Grafico guardado en: {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()

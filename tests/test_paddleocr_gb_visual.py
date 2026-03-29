"""
Test visual: PaddleOCR + Gradient Boosting.

Genera una imagen por muestra mostrando cada paso del pipeline:
  [Imagen original] [OCR texto] [Preprocesamiento] [Candidatos + GB] [Resultado]

Uso (desde P3_AP-IA/):
  .venv_paddle/Scripts/python tests/test_paddleocr_gb_visual.py
  .venv_paddle/Scripts/python tests/test_paddleocr_gb_visual.py --num-samples 20
"""

import argparse
import json
import os
import sys
import textwrap
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
OUTPUT_DIR = os.path.join("results", "paddleocr_gb_visual")


# ===================================================================
#  Modelo
# ===================================================================

def _train_gb_from_jsonl():
    train_data = load_jsonl(os.path.join("data", "processed", "cord_training_data.jsonl"))
    X, y = [], []
    for d in train_data:
        lv = normalize_amount(str(d["label"]))
        if lv is None or lv == 0:
            continue
        text = preprocess_ocr_text(d["text"])
        cands = extract_number_candidates(text)
        if not cands:
            continue
        best = _find_best_match(cands, lv)
        for i in range(len(cands)):
            X.append(candidate_features(text, cands, i))
            y.append(1 if i == best else 0)
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
    clf.fit(X, y)
    return clf, sc


def load_model(model_path):
    gb_clf, scaler = None, None
    for path in [COMPAT_MODEL_PATH, model_path]:
        if os.path.isfile(path):
            try:
                data = joblib.load(path)
                gb_clf, scaler = data["gb_clf"], data["scaler"]
                print(f"Modelo cargado: {path}")
                return gb_clf, scaler
            except Exception as e:
                print(f"No se pudo cargar {path}: {e}")
    print("Re-entrenando GB desde JSONL...")
    gb_clf, scaler = _train_gb_from_jsonl()
    os.makedirs(os.path.dirname(COMPAT_MODEL_PATH) or ".", exist_ok=True)
    joblib.dump({"gb_clf": gb_clf, "scaler": scaler}, COMPAT_MODEL_PATH)
    print(f"Modelo guardado: {COMPAT_MODEL_PATH}")
    return gb_clf, scaler


# ===================================================================
#  Pipeline (devuelve info detallada de cada paso)
# ===================================================================

def run_pipeline(ocr, gb_clf, scaler, image):
    info = {}

    # 1 — PaddleOCR
    t = time.perf_counter()
    result = ocr.ocr(image, cls=True)
    info["t_ocr"] = time.perf_counter() - t
    lines = result[0] if result and result[0] else []
    info["ocr_lines"] = lines
    info["ocr_text"] = " ".join(line[1][0] for line in lines)

    # 2 — Preprocesamiento
    t = time.perf_counter()
    info["prep_text"] = preprocess_ocr_text(info["ocr_text"])
    info["t_prep"] = time.perf_counter() - t

    # 3 — Candidatos + GB
    t = time.perf_counter()
    cands = extract_number_candidates(info["prep_text"])
    info["candidates"] = cands
    info["pred"] = None
    info["scores"] = []
    info["best_idx"] = None

    if cands:
        X = np.array(
            [candidate_features(info["prep_text"], cands, i) for i in range(len(cands))],
            dtype=np.float32,
        )
        X = scaler.transform(X)
        probs = gb_clf.predict_proba(X)
        scores = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        info["scores"] = scores.tolist()
        info["best_idx"] = int(np.argmax(scores))
        info["pred"] = cands[info["best_idx"]]["raw"]

    info["t_gb"] = time.perf_counter() - t
    info["t_total"] = info["t_ocr"] + info["t_prep"] + info["t_gb"]
    return info


# ===================================================================
#  Visualización
# ===================================================================

def _wrap(text, width=58):
    return "\n".join(textwrap.wrap(text, width)) if text else ""


def _highlight(text, value):
    """Recorta el texto alrededor del valor encontrado."""
    if not value or not text:
        return text[:250] + ("..." if len(text) > 250 else "")
    idx = text.find(value)
    if idx == -1:
        return text[:250] + ("..." if len(text) > 250 else "")
    start = max(0, idx - 60)
    end = min(len(text), idx + len(value) + 60)
    snippet = text[start:end]
    snippet = snippet.replace(value, f">>>{value}<<<", 1)
    return ("..." if start > 0 else "") + snippet + ("..." if end < len(text) else "")


def visualize(image, info, ground_truth, sample_id, output_dir):
    fig = plt.figure(figsize=(22, 9))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1.3, 1, 1, 1],
                  hspace=0.30, wspace=0.25)

    # ── Col 0: Imagen original (2 filas) ──
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(image)
    ax.set_title(f"Imagen Original\nMuestra {sample_id}", fontsize=11, fontweight="bold")
    ax.axis("off")

    # ── Col 1 fila 0: Texto OCR ──
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ocr_display = _wrap(info["ocr_text"][:350] + ("..." if len(info["ocr_text"]) > 350 else ""))
    ax.text(0.03, 0.97, ocr_display, transform=ax.transAxes,
            fontsize=6.5, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9c4", alpha=0.85))
    ax.set_title(f"Paso 1 - PaddleOCR ({info['t_ocr']*1000:.0f} ms)",
                 fontsize=10, fontweight="bold")

    # ── Col 1 fila 1: Texto preprocesado ──
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    prep_display = _wrap(info["prep_text"][:350] + ("..." if len(info["prep_text"]) > 350 else ""))
    ax.text(0.03, 0.97, prep_display, transform=ax.transAxes,
            fontsize=6.5, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#e1f5fe", alpha=0.85))
    ax.set_title(f"Paso 2 - Preprocesamiento ({info['t_prep']*1000:.2f} ms)",
                 fontsize=10, fontweight="bold")

    # ── Col 2 (2 filas): Candidatos + GB scoring ──
    ax = fig.add_subplot(gs[:, 2])
    ax.axis("off")
    cands = info["candidates"]
    scores = info["scores"]
    best_idx = info["best_idx"]

    cand_text = f"Candidatos encontrados: {len(cands)}\n{'─'*40}\n"
    for j, c in enumerate(cands[:15]):
        score_str = f"  score={scores[j]:.3f}" if j < len(scores) else ""
        marker = "  <<< ELEGIDO" if j == best_idx else ""
        cand_text += f"  {c['raw']:>14}  (val={c['value']:>10,.0f}){score_str}{marker}\n"
    if len(cands) > 15:
        cand_text += f"  ... y {len(cands)-15} mas\n"

    gb_color = "#c8e6c9" if info["pred"] else "#ffcdd2"
    ax.text(0.03, 0.97, cand_text, transform=ax.transAxes,
            fontsize=7, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=gb_color, alpha=0.7))
    ax.set_title(f"Paso 3 - Gradient Boosting ({info['t_gb']*1000:.2f} ms)",
                 fontsize=10, fontweight="bold")

    # ── Col 3 (2 filas): Resultado final ──
    ax = fig.add_subplot(gs[:, 3])
    ax.axis("off")

    pred = info["pred"]
    gt_line = ""
    box_color = "#90caf9"

    if ground_truth is not None:
        gt_val = normalize_amount(str(ground_truth))
        pred_val = normalize_amount(str(pred)) if pred else None
        if gt_val and pred_val and gt_val != 0:
            rel = abs(pred_val - gt_val) / gt_val
            if rel < 1e-6:
                verdict, box_color = "EXACTO", "#a5d6a7"
            elif rel <= 0.05:
                verdict, box_color = f"APROX ({rel*100:.1f}%)", "#fff176"
            else:
                verdict, box_color = f"ERROR ({rel*100:.1f}%)", "#ef9a9a"
        else:
            verdict, box_color = "NO COMPARABLE", "#bdbdbd"
        gt_line = f"\nGround Truth:  {ground_truth}\nEvaluacion:    {verdict}"

    context = _highlight(info["prep_text"], pred)

    result_text = (
        f"PREDICCION:  {pred or 'Sin resultado'}\n"
        f"{gt_line}\n"
        f"{'─'*40}\n"
        f"Contexto:\n{_wrap(context, 38)}\n"
        f"{'─'*40}\n"
        f"OCR:     {info['t_ocr']*1000:>7.0f} ms\n"
        f"Preproc: {info['t_prep']*1000:>7.2f} ms\n"
        f"GB:      {info['t_gb']*1000:>7.2f} ms\n"
        f"TOTAL:   {info['t_total']*1000:>7.0f} ms"
    )
    ax.text(0.5, 0.55, result_text, transform=ax.transAxes,
            fontsize=9, va="center", ha="center", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=box_color, alpha=0.6))
    ax.set_title("Resultado Final", fontsize=11, fontweight="bold")

    fig.suptitle(f"PaddleOCR + GB - Muestra {sample_id}", fontsize=14, fontweight="bold", y=0.99)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"sample_{sample_id:03d}.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ===================================================================
#  Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    print("=" * 60)
    print("  PaddleOCR + GB — Test Visual")
    print("=" * 60)

    gb_clf, scaler = load_model(args.model_path)

    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    print("PaddleOCR inicializado\n")

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

    n_exact = 0

    for idx, s in enumerate(samples):
        print(f"  [{idx+1}/{len(samples)}] Muestra {s['id']}...", end=" ")

        info = run_pipeline(ocr, gb_clf, scaler, s["image"])

        pred_val = normalize_amount(str(info["pred"])) if info["pred"] else None
        gt_val = normalize_amount(str(s["label"]))
        exact = False
        if pred_val and gt_val and gt_val != 0:
            exact = abs(pred_val - gt_val) / gt_val < 1e-6
        if exact:
            n_exact += 1

        out = visualize(s["image"], info, s["label"], s["id"], args.output_dir)
        mark = "OK" if exact else "X"
        print(f"GT={s['label']:>12}  Pred={str(info['pred'] or '-'):>12}  "
              f"{info['t_total']*1000:.0f}ms  {mark}  -> {out}")

    n = len(samples)
    print(f"\n  Exact Match: {n_exact}/{n} ({n_exact/n*100:.1f}%)")
    print(f"  Imagenes en: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

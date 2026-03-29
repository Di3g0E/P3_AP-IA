"""
Comparación: baseline naive (primer número tras "total") vs Gradient Boosting solo.

El baseline:
  1. Preprocesa el texto OCR
  2. Busca la palabra "total"
  3. Devuelve el primer candidato numérico que aparece después — sin filtros.

Gradient Boosting solo:
  1. Preprocesa el texto OCR
  2. Extrae todos los candidatos numéricos
  3. GB puntúa cada candidato por sus features (posición, contexto, magnitud...)
  4. Devuelve el candidato con mayor score — sin Regex previa.

Uso:
  python tests/test_baseline_vs_gb.py --num-samples 10
  python tests/test_baseline_vs_gb.py --num-samples 1000   # todo el test set
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.compare_extraction_models import (
    normalize_amount,
    preprocess_ocr_text,
    extract_number_candidates,
    candidate_features,
    TOTAL_KEYWORDS,
)
from test_mobile_pipeline import (
    MobilePipeline,
    _init_easyocr,
    load_cord_test_images,
)

DEFAULT_MODEL_PATH = os.path.join("models", "mobile_pipeline_gb.joblib")


# ===================================================================
#  BASELINE: primer candidato numérico tras "total"
# ===================================================================

def baseline_predict(text: str) -> str | None:
    """Preprocesa + devuelve el primer candidato tras la keyword 'total'."""
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


# ===================================================================
#  GB SOLO: predicción directa con Gradient Boosting
# ===================================================================

def gb_only_predict(pipeline: MobilePipeline, text: str) -> str | None:
    """Preprocesa + usa solo GB para elegir candidato (sin Regex)."""
    preprocessed = preprocess_ocr_text(text)
    pred, _score, _cands = pipeline._predict_gb(preprocessed)
    return pred


# ===================================================================
#  MÉTRICAS
# ===================================================================

def _eval(pred_raw: str | None, gt_label: str) -> dict:
    pred_val = normalize_amount(str(pred_raw)) if pred_raw else None
    gt_val = normalize_amount(str(gt_label))
    rel_err = None
    exact = False
    if pred_val is not None and gt_val and gt_val != 0:
        rel_err = abs(pred_val - gt_val) / gt_val
        exact = rel_err < 1e-6
    return dict(
        prediction=pred_raw,
        pred_val=pred_val,
        gt_val=gt_val,
        rel_error=rel_err,
        exact=exact,
        has_pred=pred_raw is not None,
    )


def _summary(evals: list[dict], name: str) -> dict:
    n = len(evals)
    n_pred = sum(e["has_pred"] for e in evals)
    n_exact = sum(e["exact"] for e in evals)
    n_5pct = sum(1 for e in evals if e["rel_error"] is not None and e["rel_error"] <= 0.05)
    n_10pct = sum(1 for e in evals if e["rel_error"] is not None and e["rel_error"] <= 0.10)
    errors = [e["rel_error"] for e in evals if e["rel_error"] is not None]
    mre = np.mean(errors) if errors else float("nan")

    print(f"\n  {name}:")
    print(f"    Prediction Rate:   {n_pred}/{n}  ({n_pred/n*100:.1f}%)")
    print(f"    Exact Match:       {n_exact}/{n}  ({n_exact/n*100:.1f}%)")
    print(f"    Match +-5%%:        {n_5pct}/{n}  ({n_5pct/n*100:.1f}%)")
    print(f"    Match +-10%%:       {n_10pct}/{n}  ({n_10pct/n*100:.1f}%)")
    print(f"    MRE:               {mre:.4f}")

    return dict(
        name=name, n=n,
        pred_rate=n_pred / n,
        exact=n_exact / n,
        match_5=n_5pct / n,
        match_10=n_10pct / n,
        mre=mre,
    )


# ===================================================================
#  VISUALIZACIÓN
# ===================================================================

def plot_comparison(bl: dict, gb: dict, output_dir: str) -> str:
    metrics = ["pred_rate", "exact", "match_5", "match_10"]
    labels = ["Prediction\nRate", "Exact\nMatch", "Match\n+-5%", "Match\n+-10%"]

    bl_vals = [bl[m] for m in metrics]
    gb_vals = [gb[m] for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_bl = ax.bar(x - w / 2, bl_vals, w,
                     label="Baseline (numero tras 'total')",
                     color="#ef9a9a", edgecolor="white")
    bars_gb = ax.bar(x + w / 2, gb_vals, w,
                     label="Gradient Boosting (solo)",
                     color="#42a5f5", edgecolor="white")

    for bars in (bars_bl, bars_gb):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{b.get_height():.1%}", ha="center", fontsize=10,
                    fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Ratio", fontsize=11)
    ax.set_title("Baseline vs Gradient Boosting (solo) — CORD test",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "baseline_vs_gb.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ===================================================================
#  MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comparacion: baseline (numero tras 'total') vs GB solo",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output-dir", default="results/baseline_vs_gb")
    args = parser.parse_args()

    print("=" * 60)
    print("  BASELINE vs GRADIENT BOOSTING (solo)")
    print("=" * 60)

    # ── 1. Cargar modelo GB ──
    pipeline = MobilePipeline()
    if os.path.isfile(args.model_path):
        pipeline.load(args.model_path)
    else:
        print(f"\n  ERROR: modelo no encontrado en {args.model_path}")
        print("  Ejecuta primero: python tests/test_mobile_pipeline.py")
        return

    # ── 2. Cargar datos CORD test ──
    print("\n>>> Inicializando EasyOCR...")
    reader = _init_easyocr()

    print(f">>> Cargando {args.num_samples} muestras de CORD test + EasyOCR...")
    samples = load_cord_test_images(reader, num_samples=args.num_samples)
    print(f"    {len(samples)} muestras cargadas\n")

    # ── 3. Evaluar ambos métodos ──
    bl_evals = []
    gb_evals = []

    print(f"{'#':>4}  {'GT':>15}  {'Baseline':>15}  {'GB solo':>15}  BL  GB")
    print("-" * 78)

    for s in samples:
        text = s["ocr_text"]
        gt = s["label"]

        bl_pred = baseline_predict(text)
        bl_ev = _eval(bl_pred, gt)
        bl_evals.append(bl_ev)

        gb_pred = gb_only_predict(pipeline, text)
        gb_ev = _eval(gb_pred, gt)
        gb_evals.append(gb_ev)

        bl_mark = "OK" if bl_ev["exact"] else ("~" if bl_ev["rel_error"] and bl_ev["rel_error"] <= 0.05 else "X")
        gb_mark = "OK" if gb_ev["exact"] else ("~" if gb_ev["rel_error"] and gb_ev["rel_error"] <= 0.05 else "X")

        print(
            f"{s['image_id']:>4}  "
            f"{gt:>15}  "
            f"{str(bl_pred or '-'):>15}  "
            f"{str(gb_pred or '-'):>15}  "
            f"{bl_mark:>2}  {gb_mark:>2}"
        )

    # ── 4. Resumen ──
    print("\n" + "=" * 60)
    print("  RESUMEN")
    print("=" * 60)

    bl_stats = _summary(bl_evals, "Baseline (primer numero tras 'total')")
    gb_stats = _summary(gb_evals, "Gradient Boosting (solo, sin Regex)")

    print(f"\n  Diferencia (GB - Baseline):")
    print(f"    Exact Match:  {(gb_stats['exact'] - bl_stats['exact'])*100:+.1f} pp")
    print(f"    Match +-5%%:   {(gb_stats['match_5'] - bl_stats['match_5'])*100:+.1f} pp")
    print(f"    Pred. Rate:   {(gb_stats['pred_rate'] - bl_stats['pred_rate'])*100:+.1f} pp")

    # ── 5. Gráfico ──
    out = plot_comparison(bl_stats, gb_stats, args.output_dir)
    print(f"\n  Grafico guardado en: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()

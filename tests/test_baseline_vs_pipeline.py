"""
Comparación: baseline naive (primer número tras "total") vs pipeline móvil (Regex+GB).

El baseline hace exactamente esto:
  1. Preprocesa el texto OCR
  2. Busca la palabra "total" en el texto
  3. Devuelve el primer candidato numérico (de extract_number_candidates) que
     aparezca DESPUÉS de esa palabra — sin filtros, sin ML, sin fallback.

El pipeline móvil:
  1. Preprocesa el texto OCR
  2. Regex Keyword (busca "total" + filtra negativos + fallback a max sin contexto negativo)
  3. Si Regex falla → Gradient Boosting (scoring de candidatos por features)

Uso:
  python tests/test_baseline_vs_pipeline.py --num-samples 10
  python tests/test_baseline_vs_pipeline.py --num-samples 1000   # todo el test set
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.compare_extraction_models import (
    normalize_amount,
    preprocess_ocr_text,
    extract_number_candidates,
    candidate_features,
    _find_best_match,
    RegexKeywordModel,
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
    """Devuelve el primer candidato de extract_number_candidates() que
    aparece después de la keyword 'total'. Sin filtros ni fallback."""
    preprocessed = preprocess_ocr_text(text)
    candidates = extract_number_candidates(preprocessed)
    if not candidates:
        return None

    # Buscar la última ocurrencia de keyword "total"
    last_kw_end = -1
    for m in TOTAL_KEYWORDS.finditer(preprocessed):
        last_kw_end = m.end()

    if last_kw_end == -1:
        return None  # no hay keyword "total" → no devolvemos nada

    # Primer candidato que empieza después (o muy cerca) de la keyword
    for c in candidates:
        if c["start"] >= last_kw_end - 5:
            return c["raw"]

    return None


# ===================================================================
#  MÉTRICAS
# ===================================================================

def _eval(pred_raw: str | None, gt_label: str) -> dict:
    """Compara predicción con ground truth."""
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


def _summary(evals: list[dict], name: str):
    """Imprime resumen de un método."""
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
#  VISUALIZACIÓN COMPARATIVA
# ===================================================================

def plot_comparison(baseline_stats: dict, pipeline_stats: dict, output_dir: str) -> str:
    """Gráfico de barras lado a lado: baseline vs pipeline."""
    metrics = ["pred_rate", "exact", "match_5", "match_10"]
    labels = ["Prediction\nRate", "Exact\nMatch", "Match\n+-5%", "Match\n+-10%"]

    bl_vals = [baseline_stats[m] for m in metrics]
    pl_vals = [pipeline_stats[m] for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_bl = ax.bar(x - w / 2, bl_vals, w, label="Baseline (numero tras 'total')",
                     color="#ef9a9a", edgecolor="white")
    bars_pl = ax.bar(x + w / 2, pl_vals, w, label="Pipeline (Regex + GB)",
                     color="#81c784", edgecolor="white")

    for bars in (bars_bl, bars_pl):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{b.get_height():.1%}", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Ratio", fontsize=11)
    ax.set_title("Baseline vs Pipeline Movil — CORD test", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "baseline_vs_pipeline.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ===================================================================
#  MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comparacion: baseline (numero tras 'total') vs pipeline movil",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output-dir", default="results/baseline_vs_pipeline")
    args = parser.parse_args()

    print("=" * 60)
    print("  BASELINE vs PIPELINE MOVIL")
    print("=" * 60)

    # ── 1. Cargar pipeline ──
    pipeline = MobilePipeline()
    if os.path.isfile(args.model_path):
        pipeline.load(args.model_path)
    else:
        print(f"\n  AVISO: modelo no encontrado en {args.model_path}")
        print("  Ejecuta primero: python tests/test_mobile_pipeline.py")
        print("  El pipeline funcionara solo con Regex (sin GB).\n")

    # ── 2. Cargar datos CORD test ──
    print("\n>>> Inicializando EasyOCR...")
    reader = _init_easyocr()

    print(f">>> Cargando {args.num_samples} muestras de CORD test + EasyOCR...")
    samples = load_cord_test_images(reader, num_samples=args.num_samples)
    print(f"    {len(samples)} muestras cargadas\n")

    # ── 3. Evaluar ambos métodos ──
    baseline_evals = []
    pipeline_evals = []

    print(f"{'#':>4}  {'GT':>15}  {'Baseline':>15}  {'Pipeline':>15}  BL  PL")
    print("-" * 78)

    for s in samples:
        text = s["ocr_text"]
        gt = s["label"]

        # Baseline
        bl_pred = baseline_predict(text)
        bl_eval = _eval(bl_pred, gt)
        baseline_evals.append(bl_eval)

        # Pipeline
        info = pipeline.run(text)
        pl_pred = info["final_result"]
        pl_eval = _eval(pl_pred, gt)
        pipeline_evals.append(pl_eval)

        bl_mark = "OK" if bl_eval["exact"] else ("~" if bl_eval["rel_error"] and bl_eval["rel_error"] <= 0.05 else "X")
        pl_mark = "OK" if pl_eval["exact"] else ("~" if pl_eval["rel_error"] and pl_eval["rel_error"] <= 0.05 else "X")

        print(
            f"{s['image_id']:>4}  "
            f"{gt:>15}  "
            f"{str(bl_pred or '-'):>15}  "
            f"{str(pl_pred or '-'):>15}  "
            f"{bl_mark:>2}  {pl_mark:>2}"
        )

    # ── 4. Resumen ──
    print("\n" + "=" * 60)
    print("  RESUMEN")
    print("=" * 60)

    bl_stats = _summary(baseline_evals, "Baseline (primer numero tras 'total')")
    pl_stats = _summary(pipeline_evals, "Pipeline movil (Regex + Gradient Boosting)")

    # Diferencias
    print(f"\n  Diferencia (Pipeline - Baseline):")
    print(f"    Exact Match:  {(pl_stats['exact'] - bl_stats['exact'])*100:+.1f} pp")
    print(f"    Match +-5%%:   {(pl_stats['match_5'] - bl_stats['match_5'])*100:+.1f} pp")
    print(f"    Pred. Rate:   {(pl_stats['pred_rate'] - bl_stats['pred_rate'])*100:+.1f} pp")

    # ── 5. Gráfico ──
    out = plot_comparison(bl_stats, pl_stats, args.output_dir)
    print(f"\n  Grafico guardado en: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()

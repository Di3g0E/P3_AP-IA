"""
Test del pipeline móvil recomendado para extracción de totales en facturas.

Pipeline híbrido (2 pasos):
  Paso 1 — Regex Keyword   (~0.01 ms,  ~0 KB)
  Paso 2 — Gradient Boosting si Regex no encuentra resultado (~0.2 ms, ~1-5 MB)
  Ambos pasos reciben texto con preprocesamiento OCR previo.

Flujo:
  1. Busca el modelo en --model-path (por defecto models/mobile_pipeline_gb.joblib).
  2. Si NO existe: descarga CORD train + validation de HuggingFace, ejecuta
     EasyOCR sobre las imágenes, entrena el Gradient Boosting, evalúa sobre
     validation y guarda el modelo.
  3. Carga --num-samples muestras del split test de CORD, ejecuta EasyOCR +
     pipeline y genera una visualización por muestra + resumen agregado.

Uso:
  # Test básico (10 muestras, entrena si no existe modelo)
  python tests/test_mobile_pipeline.py

  # Más muestras
  python tests/test_mobile_pipeline.py --num-samples 30

  # Forzar re-entrenamiento
  python tests/test_mobile_pipeline.py --force-retrain

  # Usar un modelo concreto
  python tests/test_mobile_pipeline.py --model-path models/mi_modelo.joblib
"""

import argparse
import json
import os
import re
import sys
import textwrap
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import joblib

# ---------------------------------------------------------------------------
# Imports del proyecto
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from scripts.compare_extraction_models import (
    normalize_amount,
    preprocess_ocr_text,
    extract_number_candidates,
    candidate_features,
    _find_best_match,
    RegexKeywordModel,
    TOTAL_KEYWORDS,
    NEGATIVE_KEYWORDS,
)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


# ===================================================================
# 1.  PIPELINE MÓVIL
# ===================================================================

class MobilePipeline:
    """Pipeline híbrido Regex → Gradient Boosting + preprocesamiento OCR."""

    def __init__(self):
        self.regex = RegexKeywordModel()
        self.gb_clf = None
        self.scaler = None
        self._trained = False

    # ----- persistencia -----
    def save(self, path: str):
        """Guarda clasificador + scaler en un fichero joblib."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"gb_clf": self.gb_clf, "scaler": self.scaler}, path)
        print(f"    Modelo guardado en {path}")

    def load(self, path: str):
        """Carga clasificador + scaler desde fichero joblib."""
        data = joblib.load(path)
        self.gb_clf = data["gb_clf"]
        self.scaler = data["scaler"]
        self._trained = True
        print(f"    Modelo cargado desde {path}")

    # ----- entrenamiento GB -----
    def train(self, texts: list[str], labels: list[str]):
        X, y = [], []
        for text, label in zip(texts, labels):
            label_val = normalize_amount(str(label))
            if label_val is None or label_val == 0:
                continue
            cands = extract_number_candidates(text)
            if not cands:
                continue
            best_idx = _find_best_match(cands, label_val)
            for i in range(len(cands)):
                X.append(candidate_features(text, cands, i))
                y.append(1 if i == best_idx else 0)

        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        self.gb_clf = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, random_state=42
        )
        self.gb_clf.fit(X, y)
        self._trained = True

    # ----- predicción GB -----
    def _predict_gb(self, text: str):
        """Devuelve (raw, score, candidates) o (None, None, [])."""
        if not self._trained:
            return None, None, []
        cands = extract_number_candidates(text)
        if not cands:
            return None, None, []
        X = np.array(
            [candidate_features(text, cands, i) for i in range(len(cands))],
            dtype=np.float32,
        )
        X = self.scaler.transform(X)
        probs = self.gb_clf.predict_proba(X)
        scores = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        best = int(np.argmax(scores))
        return cands[best]["raw"], float(scores[best]), cands

    # ----- pipeline completo -----
    def run(self, raw_text: str) -> dict:
        """Ejecuta el pipeline completo y devuelve detalles de cada paso."""
        info: dict = dict(
            text_raw=raw_text,
            text_preproc=None,
            # regex
            regex_result=None,
            # gradient boosting
            gb_result=None,
            gb_score=None,
            gb_candidates=[],
            # final
            final_result=None,
            step_used=None,
            # tiempos
            t_preproc=0.0,
            t_regex=0.0,
            t_gb=0.0,
            t_total=0.0,
        )

        t0 = time.perf_counter()

        # 1 — Preprocesamiento
        t = time.perf_counter()
        preprocessed = preprocess_ocr_text(raw_text)
        info["text_preproc"] = preprocessed
        info["t_preproc"] = time.perf_counter() - t

        # 2 — Regex Keyword
        t = time.perf_counter()
        regex_pred = self.regex.predict(preprocessed)
        info["regex_result"] = regex_pred
        info["t_regex"] = time.perf_counter() - t

        if regex_pred is not None:
            info["final_result"] = regex_pred
            info["step_used"] = "Regex Keyword"

        # 3 — Gradient Boosting (siempre se ejecuta para la comparación visual)
        t = time.perf_counter()
        gb_pred, gb_score, gb_cands = self._predict_gb(preprocessed)
        info["gb_result"] = gb_pred
        info["gb_score"] = gb_score
        info["gb_candidates"] = gb_cands
        info["t_gb"] = time.perf_counter() - t

        if info["final_result"] is None and gb_pred is not None:
            info["final_result"] = gb_pred
            info["step_used"] = "Gradient Boosting"

        if info["final_result"] is None:
            info["step_used"] = "Sin resultado"

        info["t_total"] = time.perf_counter() - t0
        return info


# ===================================================================
# 2.  CARGA DE DATOS DESDE CORD (HuggingFace)
# ===================================================================

def _init_easyocr():
    """Inicializa EasyOCR una sola vez."""
    import easyocr
    return easyocr.Reader(["es", "en"], gpu=False)


def load_cord_ocr(reader, split: str = "train", limit: int | None = None):
    """Descarga CORD, ejecuta EasyOCR y devuelve [{text, label, image_id}].

    Se usa para entrenamiento/validación (no guarda imágenes en memoria).
    """
    from datasets import load_dataset

    ds = load_dataset("naver-clova-ix/cord-v2", split=split, streaming=True)
    data: list[dict] = []
    for i, sample in enumerate(ds):
        if limit is not None and len(data) >= limit:
            break
        try:
            gt = json.loads(sample["ground_truth"])
            tp = gt.get("gt_parse", {}).get("total", {}).get("total_price")
            if not tp:
                continue
            img = np.array(sample["image"])
            results = reader.readtext(img)
            text = " ".join(r[1] for r in results)
            data.append(dict(text=text, label=tp, image_id=i))
            if len(data) % 25 == 0:
                print(f"      {split}: {len(data)} muestras procesadas...")
        except Exception:
            continue
    return data


def load_cord_test_images(reader, num_samples: int = 10):
    """Descarga CORD test, ejecuta EasyOCR y devuelve muestras con imagen."""
    from datasets import load_dataset

    ds = load_dataset("naver-clova-ix/cord-v2", split="test", streaming=True)
    samples: list[dict] = []
    for i, sample in enumerate(ds):
        if len(samples) >= num_samples:
            break
        try:
            gt = json.loads(sample["ground_truth"])
            tp = gt.get("gt_parse", {}).get("total", {}).get("total_price")
            if not tp:
                continue
            img = np.array(sample["image"])
            t = time.perf_counter()
            results = reader.readtext(img)
            ocr_ms = (time.perf_counter() - t) * 1000
            text = " ".join(r[1] for r in results)
            samples.append(dict(
                image=img, label=tp, ocr_text=text,
                ocr_time_ms=ocr_ms, image_id=i, source="CORD",
            ))
        except Exception:
            continue
    return samples


# ===================================================================
# 3.  VISUALIZACIÓN
# ===================================================================

def _wrap(text: str, width: int = 62) -> str:
    """Ajusta texto largo a ancho fijo para mostrar en gráficos."""
    return "\n".join(textwrap.wrap(text, width)) if text else ""


def _highlight_in_text(text: str, value_raw: str | None) -> str:
    """Marca el valor encontrado con >>> <<< dentro del texto truncado."""
    if value_raw is None or not text:
        return text[:300] + ("..." if len(text) > 300 else "")
    idx = text.find(value_raw)
    if idx == -1:
        return text[:300] + ("..." if len(text) > 300 else "")
    start = max(0, idx - 80)
    end = min(len(text), idx + len(value_raw) + 80)
    snippet = text[start:end]
    snippet = snippet.replace(value_raw, f">>>{value_raw}<<<", 1)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + snippet + suffix


def visualize_sample(image, info: dict, ground_truth: str | None,
                     sample_id: int, source: str, output_dir: str) -> str:
    """Genera una figura con la imagen original y los resultados de cada paso."""

    fig = plt.figure(figsize=(20, 11))
    gs = GridSpec(3, 3, figure=fig, width_ratios=[1.3, 1, 1],
                  hspace=0.35, wspace=0.30)

    # ── Panel izquierdo: imagen original (ocupa 3 filas) ──
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(image)
    ax_img.set_title(f"Imagen Original\nMuestra {sample_id} — {source}",
                     fontsize=11, fontweight="bold")
    ax_img.axis("off")

    # ── (0,1)  Texto OCR crudo ──
    ax_raw = fig.add_subplot(gs[0, 1])
    ax_raw.axis("off")
    raw_display = _wrap(info["text_raw"][:400] + ("..." if len(info["text_raw"]) > 400 else ""))
    ax_raw.text(
        0.03, 0.97, raw_display, transform=ax_raw.transAxes,
        fontsize=6.5, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9c4", alpha=0.85),
    )
    ax_raw.set_title("Paso 0 — Texto OCR (EasyOCR)", fontsize=10, fontweight="bold")

    # ── (0,2)  Texto preprocesado ──
    ax_prep = fig.add_subplot(gs[0, 2])
    ax_prep.axis("off")
    prep_display = _wrap(info["text_preproc"][:400] + ("..." if len(info["text_preproc"]) > 400 else ""))
    ax_prep.text(
        0.03, 0.97, prep_display, transform=ax_prep.transAxes,
        fontsize=6.5, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e1f5fe", alpha=0.85),
    )
    ax_prep.set_title("Paso 1 — Preprocesamiento OCR", fontsize=10, fontweight="bold")

    # ── (1,1)  Regex Keyword ──
    ax_regex = fig.add_subplot(gs[1, 1])
    ax_regex.axis("off")

    regex_val = info["regex_result"]
    regex_ok = regex_val is not None
    regex_color = "#c8e6c9" if regex_ok else "#ffcdd2"

    regex_snippet = _highlight_in_text(info["text_preproc"], regex_val)
    regex_lines = (
        f"Resultado:  {regex_val or 'None'}\n"
        f"Estado:     {'ENCONTRADO' if regex_ok else 'NO ENCONTRADO'}\n"
        f"Tiempo:     {info['t_regex']*1000:.2f} ms\n"
        f"{'─'*50}\n"
        f"Contexto:\n{_wrap(regex_snippet, 55)}"
    )
    ax_regex.text(
        0.03, 0.97, regex_lines, transform=ax_regex.transAxes,
        fontsize=7, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=regex_color, alpha=0.7),
    )
    ax_regex.set_title("Paso 2 — Regex Keyword", fontsize=10, fontweight="bold")

    # ── (1,2)  Gradient Boosting ──
    ax_gb = fig.add_subplot(gs[1, 2])
    ax_gb.axis("off")

    gb_val = info["gb_result"]
    gb_score = info["gb_score"]
    gb_cands = info["gb_candidates"]
    gb_ok = gb_val is not None
    gb_color = "#c8e6c9" if gb_ok else "#ffcdd2"

    cand_lines = ""
    if gb_cands:
        for c in gb_cands[:10]:
            marker = " <<" if c["raw"] == gb_val else ""
            cand_lines += f"  {c['raw']:>15}  (val={c['value']:,.0f}){marker}\n"

    score_txt = f"{gb_score:.4f}" if gb_score is not None else "—"
    gb_lines = (
        f"Candidatos:  {len(gb_cands)}\n"
        f"{cand_lines}"
        f"Seleccionado: {gb_val or 'None'}  (score={score_txt})\n"
        f"Tiempo:       {info['t_gb']*1000:.2f} ms"
    )
    ax_gb.text(
        0.03, 0.97, gb_lines, transform=ax_gb.transAxes,
        fontsize=7, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=gb_color, alpha=0.7),
    )
    ax_gb.set_title("Paso 3 — Gradient Boosting", fontsize=10, fontweight="bold")

    # ── (2,1:2)  Resultado final (ocupa 2 columnas) ──
    ax_final = fig.add_subplot(gs[2, 1:])
    ax_final.axis("off")

    final = info["final_result"]
    step = info["step_used"]

    # Evaluación contra ground truth
    gt_line = ""
    if ground_truth is not None:
        gt_val = normalize_amount(str(ground_truth))
        pred_val = normalize_amount(str(final)) if final else None
        if gt_val and pred_val and gt_val != 0:
            rel = abs(pred_val - gt_val) / gt_val
            if rel < 1e-6:
                verdict = "EXACTO"
                box_color = "#a5d6a7"
            elif rel <= 0.05:
                verdict = f"APROX (error {rel*100:.1f}%)"
                box_color = "#fff176"
            else:
                verdict = f"ERROR (error {rel*100:.1f}%)"
                box_color = "#ef9a9a"
        else:
            verdict = "NO COMPARABLE"
            box_color = "#bdbdbd"
        gt_line = f"Ground Truth:    {ground_truth}\nEvaluacion:      {verdict}\n"
    else:
        box_color = "#90caf9"

    final_lines = (
        f"RESULTADO FINAL:  {final or 'Sin resultado'}\n"
        f"Paso utilizado:   {step}\n"
        f"{gt_line}"
        f"{'─'*50}\n"
        f"Tiempos:  preproc={info['t_preproc']*1000:.2f} ms  |  "
        f"regex={info['t_regex']*1000:.2f} ms  |  "
        f"GB={info['t_gb']*1000:.2f} ms  |  "
        f"total={info['t_total']*1000:.2f} ms"
    )
    ax_final.text(
        0.5, 0.55, final_lines, transform=ax_final.transAxes,
        fontsize=10, verticalalignment="center", horizontalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor=box_color, alpha=0.65),
    )
    ax_final.set_title("Resultado Final (Pipeline Hibrido)", fontsize=11, fontweight="bold")

    fig.suptitle(
        f"Pipeline Movil — Muestra {sample_id}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"pipeline_sample_{sample_id:03d}.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def visualize_summary(results: list[dict], output_dir: str) -> str:
    """Genera un gráfico resumen con las métricas agregadas."""

    n = len(results)
    n_gt = [r for r in results if r["ground_truth"] is not None]

    if not n_gt:
        return ""

    exact = sum(1 for r in n_gt if r["exact"])
    approx = sum(1 for r in n_gt if r["rel_error"] is not None and r["rel_error"] <= 0.05)
    regex_used = sum(1 for r in results if r["step"] == "Regex Keyword")
    gb_used = sum(1 for r in results if r["step"] == "Gradient Boosting")
    no_res = sum(1 for r in results if r["step"] == "Sin resultado")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Accuracy ---
    ax = axes[0]
    labels_bar = ["Exact Match", "Match +-5%", "Prediccion"]
    pred_rate = sum(1 for r in n_gt if r["prediction"] is not None) / len(n_gt)
    vals = [exact / len(n_gt), approx / len(n_gt), pred_rate]
    colors = ["#4caf50", "#ff9800", "#2196f3"]
    bars = ax.bar(labels_bar, vals, color=colors, edgecolor="white", width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1%}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_title("Precision sobre muestras test", fontweight="bold")
    ax.set_ylabel("Ratio")

    # --- Paso utilizado ---
    ax = axes[1]
    step_labels = ["Regex\nKeyword", "Gradient\nBoosting", "Sin\nresultado"]
    step_vals = [regex_used, gb_used, no_res]
    step_colors = ["#66bb6a", "#42a5f5", "#ef5350"]
    bars = ax.bar(step_labels, step_vals, color=step_colors, edgecolor="white", width=0.5)
    for b, v in zip(bars, step_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, str(v),
                ha="center", fontsize=12, fontweight="bold")
    ax.set_title("Paso utilizado (N muestras)", fontweight="bold")
    ax.set_ylabel("Muestras")

    # --- Tiempos ---
    ax = axes[2]
    t_ocr = [r["t_ocr_ms"] for r in results if r["t_ocr_ms"] is not None]
    t_pipe = [r["t_pipe_ms"] for r in results]
    time_labels = []
    time_vals = []
    time_colors_plot = []
    if t_ocr:
        time_labels.append("OCR\n(EasyOCR)")
        time_vals.append(np.mean(t_ocr))
        time_colors_plot.append("#ff7043")
    time_labels += ["Preproc", "Regex", "GB", "Pipeline\ntotal"]
    t_preproc = np.mean([r["t_preproc_ms"] for r in results])
    t_regex = np.mean([r["t_regex_ms"] for r in results])
    t_gb = np.mean([r["t_gb_ms"] for r in results])
    t_total = np.mean([r["t_pipe_ms"] for r in results])
    time_vals += [t_preproc, t_regex, t_gb, t_total]
    time_colors_plot += ["#e1f5fe", "#66bb6a", "#42a5f5", "#ab47bc"]

    bars = ax.bar(time_labels, time_vals, color=time_colors_plot, edgecolor="white", width=0.5)
    for b, v in zip(bars, time_vals):
        label = f"{v:.1f} ms" if v < 1000 else f"{v/1000:.2f} s"
        ax.text(b.get_x() + b.get_width() / 2, v + max(time_vals) * 0.03, label,
                ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Tiempos medios por paso", fontweight="bold")
    ax.set_ylabel("ms")

    fig.suptitle(
        f"Pipeline Movil — Resumen ({n} muestras)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "pipeline_resumen.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ===================================================================
# 5.  EVALUACIÓN SOBRE VALIDACIÓN
# ===================================================================

def evaluate_on_split(pipeline: MobilePipeline, data: list[dict], split_name: str):
    """Evalúa el pipeline sobre un split y muestra métricas (sin imágenes)."""
    n_total = len(data)
    n_exact = 0
    n_approx = 0
    n_pred = 0

    for d in data:
        info = pipeline.run(d["text"])
        pred_val = normalize_amount(str(info["final_result"])) if info["final_result"] else None
        gt_val = normalize_amount(str(d["label"]))
        if pred_val is not None:
            n_pred += 1
        if pred_val is not None and gt_val and gt_val != 0:
            rel = abs(pred_val - gt_val) / gt_val
            if rel < 1e-6:
                n_exact += 1
            if rel <= 0.05:
                n_approx += 1

    print(f"\n    Evaluacion {split_name} ({n_total} muestras):")
    print(f"      Prediction Rate: {n_pred / n_total:.1%}")
    print(f"      Exact Match:     {n_exact / n_total:.1%}")
    print(f"      Match +-5%%:      {n_approx / n_total:.1%}")


# ===================================================================
# 6.  MAIN
# ===================================================================

DEFAULT_MODEL_PATH = os.path.join("models", "mobile_pipeline_gb.joblib")


def main():
    parser = argparse.ArgumentParser(
        description="Test del pipeline movil: Regex + Gradient Boosting",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help="Ruta al modelo joblib (default: models/mobile_pipeline_gb.joblib)")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Fuerza re-entrenamiento aunque exista el modelo")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Numero de muestras de CORD test a evaluar")
    parser.add_argument("--output-dir", default="results/pipeline_test",
                        help="Directorio de salida para visualizaciones")
    args = parser.parse_args()

    print("=" * 60)
    print("  PIPELINE MOVIL — TEST COMPLETO")
    print("  Regex Keyword  ->  Gradient Boosting  +  Preprocesamiento")
    print("=" * 60)

    pipeline = MobilePipeline()
    reader = None  # se inicializa bajo demanda

    # ── 1. Cargar o entrenar modelo ──
    model_exists = os.path.isfile(args.model_path) and not args.force_retrain

    if model_exists:
        print(f"\n>>> Modelo encontrado en {args.model_path}")
        pipeline.load(args.model_path)
    else:
        if args.force_retrain:
            print(f"\n>>> Forzando re-entrenamiento (--force-retrain)")
        else:
            print(f"\n>>> Modelo NO encontrado en {args.model_path}")

        print(">>> Inicializando EasyOCR...")
        reader = _init_easyocr()

        # Train
        print(">>> Cargando CORD train + ejecutando EasyOCR...")
        train_data = load_cord_ocr(reader, split="train")
        print(f"    Train: {len(train_data)} muestras con OCR")

        # Validation
        print(">>> Cargando CORD validation + ejecutando EasyOCR...")
        val_data = load_cord_ocr(reader, split="validation")
        print(f"    Validation: {len(val_data)} muestras con OCR")

        # Entrenar
        print(">>> Entrenando Gradient Boosting...")
        t0 = time.perf_counter()
        pipeline.train(
            [d["text"] for d in train_data],
            [d["label"] for d in train_data],
        )
        print(f"    Entrenado en {time.perf_counter() - t0:.2f}s")

        # Evaluar sobre validation
        evaluate_on_split(pipeline, val_data, "validation")

        # Guardar
        pipeline.save(args.model_path)

    # ── 2. Inicializar EasyOCR para test (reutilizar si ya existe) ──
    if reader is None:
        print("\n>>> Inicializando EasyOCR...")
        reader = _init_easyocr()

    # ── 3. Cargar test de CORD ──
    print(f">>> Cargando {args.num_samples} muestras de CORD test + EasyOCR...")
    samples = load_cord_test_images(reader, num_samples=args.num_samples)
    print(f"    {len(samples)} muestras cargadas")

    # ── 4. Ejecutar pipeline muestra a muestra ──
    print(f"\n>>> Ejecutando pipeline sobre {len(samples)} muestras...\n")
    summary: list[dict] = []

    for idx, s in enumerate(samples):
        sid = s["image_id"]
        print(f"  [{idx+1}/{len(samples)}] Muestra {sid} ({s['source']})")

        # Pipeline (el OCR ya se ejecutó en load_cord_test_images)
        info = pipeline.run(s["ocr_text"])

        # Visualización
        out_path = visualize_sample(
            image=s["image"],
            info=info,
            ground_truth=s.get("label"),
            sample_id=sid,
            source=s["source"],
            output_dir=args.output_dir,
        )

        # Métricas
        pred_val = normalize_amount(str(info["final_result"])) if info["final_result"] else None
        gt_val = normalize_amount(str(s["label"])) if s.get("label") else None
        is_exact = False
        rel_err = None
        if pred_val is not None and gt_val and gt_val != 0:
            rel_err = abs(pred_val - gt_val) / gt_val
            is_exact = rel_err < 1e-6

        row = dict(
            sample_id=sid,
            ground_truth=s.get("label"),
            prediction=info["final_result"],
            step=info["step_used"],
            exact=is_exact,
            rel_error=rel_err,
            t_ocr_ms=s["ocr_time_ms"],
            t_preproc_ms=info["t_preproc"] * 1000,
            t_regex_ms=info["t_regex"] * 1000,
            t_gb_ms=info["t_gb"] * 1000,
            t_pipe_ms=info["t_total"] * 1000,
        )
        summary.append(row)

        mark = "OK" if is_exact else ("~" if rel_err and rel_err <= 0.05 else "X")
        print(
            f"    GT={str(s.get('label','N/A')):>15}  |  "
            f"Pred={str(info['final_result']):>15}  |  "
            f"Paso={info['step_used']:>18}  |  {mark}  |  "
            f"OCR={s['ocr_time_ms']:.0f}ms "
            f"Pipeline={info['t_total']*1000:.1f}ms"
        )
        print(f"    -> {out_path}")

    # ── 5. Resumen ──
    print("\n" + "=" * 60)
    print("  RESUMEN")
    print("=" * 60)

    with_gt = [r for r in summary if r["ground_truth"] is not None]
    if with_gt:
        n_exact = sum(r["exact"] for r in with_gt)
        n_approx = sum(1 for r in with_gt if r["rel_error"] is not None and r["rel_error"] <= 0.05)
        print(f"\n  Muestras con GT:     {len(with_gt)}")
        print(f"  Exact Match:         {n_exact}/{len(with_gt)}  "
              f"({n_exact/len(with_gt)*100:.1f}%)")
        print(f"  Match +-5%:          {n_approx}/{len(with_gt)}  "
              f"({n_approx/len(with_gt)*100:.1f}%)")

    n_regex = sum(1 for r in summary if r["step"] == "Regex Keyword")
    n_gb = sum(1 for r in summary if r["step"] == "Gradient Boosting")
    n_none = sum(1 for r in summary if r["step"] == "Sin resultado")
    print(f"\n  Paso utilizado:")
    print(f"    Regex Keyword:     {n_regex}/{len(summary)}")
    print(f"    Gradient Boosting: {n_gb}/{len(summary)}")
    print(f"    Sin resultado:     {n_none}/{len(summary)}")

    ocr_times = [r["t_ocr_ms"] for r in summary if r["t_ocr_ms"] is not None]
    pipe_times = [r["t_pipe_ms"] for r in summary]
    print(f"\n  Tiempos promedio:")
    if ocr_times:
        print(f"    OCR:               {np.mean(ocr_times):.0f} ms")
    print(f"    Pipeline:          {np.mean(pipe_times):.1f} ms")
    if ocr_times:
        print(f"    Total (OCR+pipe):  {np.mean(ocr_times) + np.mean(pipe_times):.0f} ms")

    # Gráfico resumen
    summary_path = visualize_summary(summary, args.output_dir)
    if summary_path:
        print(f"\n  Resumen visual:      {summary_path}")
    print(f"  Resultados en:       {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

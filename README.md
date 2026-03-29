# P3_AP-IA — Extraccion de Totales en Facturas

Sistema de extraccion automatica del importe total en facturas y recibos mediante OCR + Machine Learning.

## Pipeline final

```
Imagen de factura
    |
    v
PaddleOCR (deteccion + reconocimiento de texto)    ~875 ms
    |
    v
Preprocesamiento OCR (correccion errores: O->0, l->1, etc.)    ~0.2 ms
    |
    v
Gradient Boosting (scoring de candidatos numericos)    ~2 ms
    |
    v
Total extraido
```

**Accuracy (CORD test, 95 muestras):** 83.2% Exact Match | 86.3% Match +-5%

## Uso

Requiere uno de los tres modos de entrada (son mutuamente excluyentes):

### Imagen individual

```bash
python main.py --image data/raw/factura.png
python main.py --image C:\Users\fotos\ticket.jpg
```

Carga la imagen, ejecuta el pipeline y muestra el total por terminal.

### Directorio de imagenes

```bash
python main.py --dir data/raw
python main.py --dir C:\Users\fotos\facturas
```

Procesa todas las imagenes del directorio (png, jpg, jpeg, bmp, tiff, webp) y muestra el total de cada una.

### Camara en vivo

```bash
python main.py --camera
```

Abre la camara del dispositivo. Controles:
- **ESPACIO** — captura la imagen actual y extrae el total
- **ESC** — cierra la camara y sale

Se pueden realizar multiples capturas sin reiniciar el programa.

### Ejemplos de salida

```
==================================================
  Extraccion de Totales — PaddleOCR + GB
==================================================

                  factura_01.png  ->  Total: 1,591,600.00
                  factura_02.png  ->  Total: 580,965.00
                  factura_03.png  ->  Total: no encontrado
```

## Entornos virtuales

El proyecto tiene dos entornos:

| Entorno | Python | OCR | Uso |
|---------|--------|-----|-----|
| `.venv` | 3.13 | EasyOCR | Entrenamiento, comparativas, tests originales |
| `.venv_paddle` | 3.12 | PaddleOCR + EasyOCR | Pipeline final, tests de rendimiento |

```bash
# Activar entorno principal (EasyOCR)
.venv\Scripts\activate

# Activar entorno PaddleOCR
.venv_paddle\Scripts\activate
```

## Estructura del proyecto

```
P3_AP-IA/
  main.py                  # Punto de entrada (imagen, directorio o camara)
  src/
    models/ocr_engine.py   # Motor OCR: PaddleOCR + GB (fallback a EasyOCR)
    features/preprocess.py  # Preprocesamiento de imagen
    features/region_detector.py  # Segmentacion Header/Body/Footer
    data/data_loader.py     # Carga de imagenes y datasets
    data/data_download.py   # Descarga de CORD desde HuggingFace
    utils/image_processing.py  # Recorte y mejora de imagen
    evaluation/evaluate.py  # Visualizacion de resultados
  scripts/
    compare_extraction_models.py  # Comparativa de 8 modelos de extraccion
    generate_training_data.py     # Genera JSONL con EasyOCR sobre CORD
  tests/
    test_mobile_pipeline.py       # Test pipeline movil (Regex + GB)
    test_baseline_vs_gb.py        # Baseline vs GB solo
    test_baseline_vs_pipeline.py  # Baseline vs pipeline hibrido
    test_paddleocr_vs_easyocr.py  # PaddleOCR vs EasyOCR (4 combinaciones)
    test_paddleocr_gb_timing.py   # Tiempos PaddleOCR + GB
    test_paddleocr_gb_visual.py   # Visualizacion paso a paso
  models/
    mobile_pipeline_gb.joblib       # Modelo GB (numpy 2.x)
    mobile_pipeline_gb_np1.joblib   # Modelo GB (numpy 1.x, compat paddle)
  data/
    raw/                    # Imagenes de facturas
    processed/              # Datos JSONL (OCR preprocesado)
  results/                  # Graficos y CSVs de comparativas
```

## Datos

- **Dataset:** [CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (Naver-Clova) — recibos de restaurantes coreanos e indonesios
- **Train:** ~800 muestras | **Validation:** ~100 | **Test:** 95 muestras con ground truth
- **Formatos:** multiples divisas, con/sin decimales, diferentes layouts

## Documentacion adicional

Ver [EXPERIMENTACION.md](EXPERIMENTACION.md) para el detalle completo de todas las pruebas, modelos comparados y decisiones tomadas hasta llegar al pipeline final.

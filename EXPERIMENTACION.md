# Experimentacion — Extraccion de Totales en Facturas

Documento que recoge todas las pruebas realizadas, resultados obtenidos y decisiones tomadas hasta llegar al pipeline final de produccion.

## 1. Contexto del problema

Dada una imagen de factura o recibo, extraer automaticamente el importe total. Los datos provienen del dataset CORD (recibos reales con diferentes formatos, divisas y layouts). El texto se obtiene mediante OCR, que introduce errores de lectura (O en vez de 0, l en vez de 1, etc.).

## 2. Fase 1 — Comparativa de modelos de extraccion

### 2.1 Modelos evaluados

Se implementaron y compararon **8 modelos** de extraccion sobre los mismos datos:

| Modelo | Tipo | Descripcion |
|--------|------|-------------|
| Regex Keyword | Reglas | Busca keyword "total" y devuelve el numero adyacente |
| Regex Max Filtrado | Reglas | Devuelve el mayor numero filtrando contextos negativos (cash, change) |
| Candidate Random Forest | ML clasico | Clasifica candidatos numericos con 12 features + RF (200 arboles) |
| Candidate Gradient Boosting | ML clasico | Idem con GradientBoosting (150 arboles, depth=5) |
| Candidate SVM | ML clasico | Idem con LinearSVC |
| TF-IDF + LogReg | NLP clasico | Contexto TF-IDF (3000 features, 1-2 gramas) + features numericas |
| DistilBERT Reranker | Deep Learning | distilbert-base-multilingual-cased fine-tuned como clasificador binario |
| T5 Seq2Seq | Deep Learning | t5-small fine-tuned para generacion directa del total |

### 2.2 Datos de entrenamiento

- **Fuente:** CORD v2 (HuggingFace), split train
- **OCR:** EasyOCR (es + en) sobre las imagenes originales
- **Formato:** JSONL con campos `text` (OCR completo), `label` (total_price del ground truth)
- **Muestras:** 779 train, 95 test

### 2.3 Preprocesamiento de texto OCR

Se implemento una funcion de correccion de errores tipicos de OCR:
- `O`, `o`, `Q` adyacentes a digitos se convierten a `0`
- `D` adyacente a digitos se convierte a `0`
- `l`, `I` adyacentes a digitos se convierten a `1`
- Eliminacion de caracteres basura (`{}[]<>~`)
- Normalizacion de espacios
- Conversion a minusculas

### 2.4 Resultados con EasyOCR (datos con ruido OCR)

| Modelo | Preproc | Exact Match | +-5% | Pred Rate | Predict (s) |
|--------|:-------:|:-----------:|:----:|:---------:|:-----------:|
| T5 Seq2Seq | No | 66.3% | 72.6% | 96.8% | 13.72 |
| T5 Seq2Seq | Si | 66.3% | 71.6% | 97.9% | 14.22 |
| Cand. Random Forest | Si | 54.7% | 61.1% | 97.9% | 3.81 |
| DistilBERT Reranker | Si | 54.7% | 57.9% | 97.9% | 3.46 |
| Cand. Gradient Boosting | Si | 52.6% | 58.9% | 98.9% | 0.18 |
| Cand. Gradient Boosting | No | 49.5% | 53.7% | 87.4% | 0.15 |
| Cand. Random Forest | No | 48.4% | 52.6% | 86.3% | 3.65 |
| DistilBERT Reranker | No | 46.3% | 48.4% | 86.3% | 2.58 |
| Regex Keyword | Si | 42.1% | 48.4% | 97.9% | 0.01 |
| TF-IDF + LogReg | Si | 38.9% | 43.2% | 98.9% | 0.27 |
| Cand. SVM | Si | 37.9% | 43.2% | 98.9% | 0.12 |
| Regex Keyword | No | 36.8% | 41.1% | 88.4% | 0.02 |
| Cand. SVM | No | 32.6% | 34.7% | 82.1% | 0.12 |
| TF-IDF + LogReg | No | 32.6% | 33.7% | 81.1% | 0.25 |
| Regex Max Filtrado | Si | 25.3% | 29.5% | 97.9% | 0.01 |
| Regex Max Filtrado | No | 23.2% | 24.2% | 85.3% | 0.02 |

### 2.5 Resultados con Ground Truth (texto limpio, sin OCR)

| Modelo | Exact Match | +-5% | Pred Rate |
|--------|:-----------:|:----:|:---------:|
| Regex Keyword | 100% | 100% | 100% |
| Cand. Gradient Boosting | 100% | 100% | 100% |
| Cand. Random Forest | 100% | 100% | 100% |
| Cand. SVM | 100% | 100% | 100% |
| TF-IDF + LogReg | 96.4% | 96.4% | 100% |
| Regex Max Filtrado | 47.3% | 49.1% | 100% |

### 2.6 Conclusiones de la Fase 1

1. **El preprocesamiento de texto OCR mejora todos los modelos** (+4-8 pp en Exact Match). Debe usarse siempre.
2. **T5 tiene la mejor precision** (66.3%) pero es inviable para movil: 900 MB, 14s por prediccion.
3. **DistilBERT no justifica su coste:** misma precision que Random Forest pero 250 MB.
4. **Gradient Boosting es el mejor equilibrio** calidad/coste: 52.6% Exact Match, 0.18s, ~2 MB.
5. **El cuello de botella es la calidad del OCR**, no el modelo de extraccion (100% con texto limpio).

## 3. Fase 2 — Pipeline movil: Regex + Gradient Boosting

### 3.1 Diseno del pipeline hibrido

Se propuso un pipeline de 2 pasos para dispositivos moviles:
1. **Regex Keyword** (0.01 ms, 0 KB) — si encuentra "total" + numero, devuelve
2. **Gradient Boosting** (2 ms, 2 MB) — fallback si Regex falla

### 3.2 Resultados del pipeline hibrido (95 muestras CORD test)

- **Exact Match:** 40.0%
- **Match +-5%:** 46.3%
- **Prediction Rate:** 97.9%
- **Paso utilizado:** Regex en 93/95 (98%), GB en 0/95, sin resultado en 2/95

### 3.3 Problema detectado

Regex siempre devuelve algo (tiene fallback al numero maximo), por lo que **GB nunca se ejecuta** como fallback. La precision del pipeline hibrido es identica a la de Regex solo.

### 3.4 Test Baseline vs GB solo

Se comparo el baseline naive (primer numero tras "total") contra GB usado como metodo principal (no como fallback):

- **Baseline:** Exact Match variable segun OCR
- **GB solo:** +10.5 pp sobre Regex con EasyOCR

**Decision:** usar GB como metodo principal, no como fallback.

## 4. Fase 3 — Cambio de OCR: PaddleOCR vs EasyOCR

### 4.1 Motivacion

El analisis de tiempos mostro que el 99.95% del tiempo se consume en el OCR (~4.7s EasyOCR), no en la extraccion (~2 ms). Se evaluo PaddleOCR como alternativa mas rapida.

### 4.2 Compatibilidad

- PaddleOCR v3.4 (paddlepaddle 3.x) tiene un bug con OneDNN en Windows
- Se uso PaddleOCR v2.8.1 + paddlepaddle 2.6.2 (combinacion estable)
- Requiere Python 3.12 (no soporta 3.13), se creo entorno `.venv_paddle` con `uv`

### 4.3 Resultados comparativos (95 muestras CORD test)

| Pipeline | Exact Match | +-5% | Pred Rate | MRE |
|----------|:-----------:|:----:|:---------:|:---:|
| **PaddleOCR + GB** | **83.2%** | **86.3%** | **100%** | **0.12** |
| PaddleOCR + Regex | 57.9% | 61.1% | 78.9% | 0.19 |
| EasyOCR + GB | 52.6% | 58.9% | 97.9% | 0.44 |
| EasyOCR + Regex | 37.9% | 41.1% | 58.9% | 0.73 |

### 4.4 Tiempos

| OCR | Tiempo medio/imagen |
|-----|:-------------------:|
| PaddleOCR | 875 ms |
| EasyOCR | 16,417 ms |

PaddleOCR es **18.8x mas rapido** y produce texto de **mejor calidad** (+30 pp Exact Match).

### 4.5 Memoria en disco

| Pipeline | Tamano estimado |
|----------|:--------------:|
| PaddleOCR + GB | ~350 MB |
| EasyOCR + GB | ~900 MB |

### 4.6 Tiempos por paso (PaddleOCR + GB)

| Paso | Tiempo medio |
|------|:------------:|
| PaddleOCR | 875 ms |
| Preprocesamiento texto | 0.2 ms |
| Gradient Boosting | 2 ms |
| **Total pipeline** | **~877 ms** |

## 5. Pipeline final de produccion

### Arquitectura

```
Imagen (cualquier formato)
    |
    v
PaddleOCR v2.8 (deteccion + reconocimiento, CPU)
    |
    v
Preprocesamiento texto (correccion errores OCR)
    |
    v
Extraccion de candidatos numericos (regex)
    |
    v
Feature engineering (12 features por candidato:
    posicion relativa, magnitud, proximidad a keywords,
    contexto positivo/negativo, etc.)
    |
    v
Gradient Boosting (150 arboles, depth=5)
    - scoring de cada candidato
    - selecciona el de mayor probabilidad
    |
    v
Total extraido (float)
```

### Metricas finales

| Metrica | Valor |
|---------|:-----:|
| Exact Match | 83.2% |
| Match +-5% | 86.3% |
| Match +-10% | 88.4% |
| Prediction Rate | 100% |
| MRE | 0.12 |
| Tiempo total/imagen | ~877 ms |
| Tamano modelo GB | ~2 MB |
| Tamano total pipeline | ~350 MB |

### Modos de uso

```bash
python main.py --image factura.png     # Imagen individual
python main.py --dir data/raw          # Directorio
python main.py --camera                # Camara en vivo
```

## 6. Tests disponibles

| Test | Entorno | Descripcion |
|------|---------|-------------|
| `test_mobile_pipeline.py` | .venv | Pipeline hibrido Regex+GB con EasyOCR, visualizacion por muestra |
| `test_baseline_vs_pipeline.py` | .venv | Baseline naive vs pipeline hibrido completo |
| `test_baseline_vs_gb.py` | .venv | Baseline naive vs GB solo |
| `test_paddleocr_vs_easyocr.py` | .venv_paddle | 4 combinaciones (2 OCR x 2 extraccion), grafico comparativo |
| `test_paddleocr_gb_timing.py` | .venv_paddle | Tiempos desglosados PaddleOCR + GB |
| `test_paddleocr_gb_visual.py` | .venv_paddle | Visualizacion paso a paso del pipeline final |
| `compare_extraction_models.py` | .venv | Comparativa completa de 8 modelos x 2 datos x 2 preproc |

## 7. Decisiones clave

1. **Candidate ranking > Seq2Seq**: En vez de generar el numero, el modelo elige entre candidatos extraidos del texto. Mas robusto y rapido.
2. **GB > Random Forest**: Misma precision, 20x mas rapido en prediccion (0.18s vs 3.8s).
3. **PaddleOCR > EasyOCR**: 18.8x mas rapido, +30 pp en accuracy, 2.5x mas ligero.
4. **GB como metodo principal**: Usar GB directamente en vez de como fallback de Regex aprovecha su ventaja de +25 pp.
5. **Preprocesamiento de texto siempre activo**: +4-8 pp gratis en todos los modelos.

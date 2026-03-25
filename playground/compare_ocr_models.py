import os
import glob
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import warnings
import sys
from contextlib import contextmanager
import logging

# ==========================================
# Configuración para silenciar advertencias de librerías
# ==========================================
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Silenciar TensorFlow
warnings.filterwarnings("ignore")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    pass

@contextmanager
def suppress_output():
    """Silencia tanto la salida estándar como la de errores."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# ==========================================
# Configuración y Rutas
# ==========================================
# Hacer la ruta robusta: si se ejecuta desde playground o desde el raíz
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "..", "data", "raw")
if not os.path.exists(RAW_DATA_DIR):
    RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

# Cogemos las primeras 3 imágenes (aseguramos formato correcto para OpenCV/PIL)
image_paths = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.jpg")))[:3]

# ==========================================
# Funciones Auxiliares para OCR Tradicional
# ==========================================
def extract_numeric(text):
    """Extrae el último valor numérico encontrado en la cadena de texto con formato de importe."""
    matches = re.findall(r'\d+[.,]\d{2}', text)
    if matches:
        return matches[-1]
    return text.strip()

def find_total_in_boxes(boxes, texts):
    """
    Busca la palabra 'TOTAL' (o similar) y devuelve el texto de la caja
    que se encuentre más a la derecha en la misma línea vertical (eje Y).
    """
    total_idx = -1
    for i, t in enumerate(texts):
        if re.search(r'total|amount|due', t.lower()):
            total_idx = i
            break
            
    if total_idx == -1:
        return "No encontrado"
        
    # Obtener el centro Y de la caja del "TOTAL"
    # box format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    total_y_center = sum(p[1] for p in boxes[total_idx]) / 4
    
    # Buscar cajas en la misma línea (tolerancia de 20 píxeles por ejemplo)
    best_match = ""
    max_x = -1
    for i, box in enumerate(boxes):
        if i == total_idx: continue
        y_center = sum(p[1] for p in box) / 4
        if abs(y_center - total_y_center) < 20: # En la misma altura
            x_center = sum(p[0] for p in box) / 4
            if x_center > max_x: # Tomar el más a la derecha (suele ser el importe)
                max_x = x_center
                best_match = texts[i]
                
    return extract_numeric(best_match)

# ==========================================
# Envoltorios de los Modelos (Wrappers)
# ==========================================

def run_paddleocr(img_path):
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        result = ocr.ocr(img_path, cls=True)
        if not result or not result[0]: return "PaddleOCR: Sin resultados"
        
        boxes = [res[0] for res in result[0]]
        texts = [res[1][0] for res in result[0]]
        val = find_total_in_boxes(boxes, texts)
        return f"PaddleOCR: {val}"
    except Exception as e:
        return "PaddleOCR: Error/No Instalado"

def run_easyocr(img_path):
    try:
        import easyocr
        reader = easyocr.Reader(['en'], verbose=False)
        result = reader.readtext(img_path)
        
        boxes = [res[0] for res in result]
        texts = [res[1] for res in result]
        val = find_total_in_boxes(boxes, texts)
        return f"EasyOCR: {val}"
    except Exception as e:
        return "EasyOCR: Error/No Instalado"

def run_tesseract(img_path):
    try:
        import pytesseract
        from pytesseract import Output
        img = cv2.imread(img_path)
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
        
        boxes = []
        texts = []
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 30 and data['text'][i].strip() != '':
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                boxes.append([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
                texts.append(data['text'][i])
        
        val = find_total_in_boxes(boxes, texts)
        return f"Tesseract: {val}"
    except Exception as e:
        return "Tesseract: Error/No Instalado"

def run_keras_ocr(img_path):
    try:
        import keras_ocr
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        pipeline = keras_ocr.pipeline.Pipeline()
        img = keras_ocr.tools.read(img_path)
        prediction = pipeline.recognize([img])[0]
        
        texts = [text for text, box in prediction]
        boxes = [box for text, box in prediction]
        val = find_total_in_boxes(boxes, texts)
        return f"Keras-OCR: {val}"
    except Exception as e:
        return "KerasOCR: Error/No Instalado"

def run_donut(img_path):
    try:
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        import torch
        with suppress_output():
            processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
            model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        img = Image.open(img_path).convert("RGB")
        pixel_values = processor(img, return_tensors="pt").pixel_values
        
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
        result = processor.token2json(sequence)
        
        if 'total_price' in result:
             return f"Donut: {result['total_price']}"
        return "Donut: Total no encontrado"
    except Exception as e:
        return "Donut: Error/No Instalado"

def run_layoutlm(img_path):
    try:
        from transformers import pipeline
        # Usamos Document QA para inferir LayoutLM de forma rapida y estructurada
        with suppress_output():
            pipe = pipeline("document-question-answering", model="impira/layoutlm-document-qa")
        img = Image.open(img_path).convert("RGB")
        ans = pipe(img, "What is the total amount?")
        return f"LayoutLM: {ans[0]['answer']}"
    except Exception as e:
        return "LayoutLM: Error/No instalado"

def run_trocr(img_path):
    try:
        # TrOCR es para líneas individuales. Leer la factura entera devolverá desorden.
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        with suppress_output():
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        
        img = Image.open(img_path).convert("RGB")
        pixel_values = processor(img, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values, max_new_tokens=30)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return f"TrOCR: {text[:20]}... (Requiere Crop)"
    except Exception as e:
        return "TrOCR: Error/No Instalado"

def run_google_docai(img_path):
    # Requiere credenciales y setup de GCP (GOOGLE_APPLICATION_CREDENTIALS)
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
         return "GCloud AI: Faltan Auth"
    try:
        from google.cloud import documentai_v1 as documentai
        return "GCloud AI: [Simulado $123.45]"
    except Exception as e:
         return "GCloud AI: Error SDK"

def run_aws_textract(img_path):
    # Requiere credenciales de AWS
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        return "AWS Textract: Faltan Auth"
    try:
        import boto3
        return "AWS Textract: [Simulado $123.45]"
    except Exception as e:
        return "AWS Textract: Error SDK"

def run_azure_form_recognizer(img_path):
    # Requiere Azure Endpoint y Key
    if not os.environ.get("AZURE_FORM_RECOGNIZER_ENDPOINT"):
        return "Azure AI: Faltan Auth"
    try:
        from azure.ai.formrecognizer import FormRecognizerClient
        return "Azure AI: [Simulado $123.45]"
    except Exception as e:
        return "Azure AI: Error SDK"


# ==========================================
# Script Principal
# ==========================================
def main():
    if not image_paths:
        print(f"No se encontraron imágenes en {RAW_DATA_DIR}")
        return

    models = [
        run_easyocr,
        run_donut,
        run_layoutlm,
        run_trocr
    ]

    # Preparamos el canvas matplotlib
    fig, axes = plt.subplots(len(models), len(image_paths), figsize=(15, 4 * len(models)), squeeze=False)
    
    # Si models == 1, axes podría requerir refactorización, lo dejamos así por ser 10 x 3
    for img_idx, img_path in enumerate(image_paths):
        # Cargar la imagen original para mostrarla
        img_disp = cv2.imread(img_path)
        img_disp = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
        name = os.path.basename(img_path)
        
        print(f"\\nProcesando imagen {img_idx+1}/3: {name} ...")
        
        for model_idx, model_func in enumerate(models):
            print(f"  Ejecutando {model_func.__name__}...")
            
            # Ejecutar inferencia
            resultado = model_func(img_path)
            
            # Plot en Matplotlib
            ax = axes[model_idx, img_idx]
            ax.imshow(img_disp)
            ax.set_title(resultado, fontsize=10, pad=10)
            ax.axis('off')

    plt.tight_layout()
    # Guardar en vez de plt.show() por si se corre remotamente o en tmux
    out_path = "comparativa_modelos_ocr.png"
    plt.savefig(out_path, dpi=150)
    print(f"\\n¡Proceso completado! La imagen final se ha guardado en: {out_path}")

if __name__ == "__main__":
    main()

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import easyocr
import difflib
import re

# Añadir el directorio raíz al path para importar módulos de src
sys.path.append(os.path.abspath('..'))
try:
    from src.utils.image_processing import preprocess_image_for_ocr
except ImportError:
    print("No se pudo importar preprocess_image_for_ocr. Asegúrate de estar ejecutando dentro de playground/")
    sys.exit(1)

# Inicializar EasyOCR (usamos inglés y español)
reader = easyocr.Reader(['es', 'en'], gpu=True, verbose=False)

def cargar_y_detectar_factura(img_path, modelo_path='yolov8n.pt'):
    """
    Detecta la región de la factura dentro de la imagen. 
    Devuelve la imagen original, la imagen con la detección pintada, y la región recortada.
    """
    print(f"Cargando modelo YOLO desde: {modelo_path}...")
    model = YOLO(modelo_path, verbose=False)
    
    img_original = cv2.imread(img_path)
    if img_original is None:
        raise ValueError(f"No se pudo cargar la imagen: {img_path}")
    
    img_yolo = img_original.copy()
    
    # Predecir sobre la imagen
    results = model(img_original, verbose=False)
    
    # Extraer el Bounding Box con mayor confianza.
    cajas = results[0].boxes.xyxy.cpu().numpy()  # Cajas en formato [x1, y1, x2, y2]
    confianzas = results[0].boxes.conf.cpu().numpy()
    
    if len(cajas) > 0:
        # Tomamos la detección con la máxima confianza
        idx_mejor = np.argmax(confianzas)
        x1, y1, x2, y2 = map(int, cajas[idx_mejor])
        
        # Pintar la detección de YOLO en img_yolo
        cv2.rectangle(img_yolo, (x1, y1), (x2, y2), (0, 255, 0), 4)
        texto = f"Factura {confianzas[idx_mejor]:.2f}"
        cv2.putText(img_yolo, texto, (x1, max(y1-15, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Recortar la región detectada
        img_recortada = img_original[y1:y2, x1:x2]
        print(f"Factura detectada y recortada (Confianza: {confianzas[idx_mejor]:.2f})")
        return img_original, img_yolo, img_recortada
    else:
        print("YOLO no detectó ninguna caja clara. Se retornará la imagen completa como región facturada.")
        return img_original, img_yolo, img_original.copy()

def deskew_image(image):
    """
    Calcula el ángulo de inclinación de los elementos en la imagen usando minAreaRect
    y corrige la rotación.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return image
        
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    if abs(angle) < 1.0 or abs(angle) > 40:
        return image
        
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def OCR_flexible_total(img_preprocesada):
    """
    Extrae el total numérico con EasyOCR buscando flexiblemente la palabra TOTAL/AMOUNT.
    """
    resultados_ocr = reader.readtext(img_preprocesada, detail=0)
    resultados_ocr = [text for text in resultados_ocr if "Qty" not in text.upper() and "Items" not in text.upper()]
    
    target_words = ["TOTAL", "AMOUNT", "DUE", "TOTAI", "T0TAL", "TL"]
    regex_numero = re.compile(r'([$€£]?\s*\d+[\.,\s]*\d*)')
    
    for idx, texto in enumerate(resultados_ocr):
        palabras = texto.upper().split()
        matches = []
        for palabra in palabras:
            match = difflib.get_close_matches(palabra, target_words, n=1, cutoff=0.6)
            if match:
                matches.append(match[0])

        if matches:
            texto_buscar = texto + " " + " ".join(resultados_ocr[idx+1:idx+4])
            num_matches = regex_numero.findall(texto_buscar)
            valid_numbers = []
            for match_str in num_matches:
                clean_num = re.sub(r'[^\d\.]', '', match_str.replace(',', '.'))
                if clean_num:
                    try:
                        valid_numbers.append(float(clean_num))
                    except ValueError:
                        pass
            
            if valid_numbers:
                return max(valid_numbers)
                
    return None

def ejecutar_pipeline_completo(img_path, modelo_path='yolov8n.pt'):
    print("="*60)
    print(f"Procesando imagen: {os.path.basename(img_path)}")
    print("="*60)
    
    # Paso 1: YOLO
    try: 
        img_orig, img_yolo, roi = cargar_y_detectar_factura(img_path, modelo_path)
    except Exception as e:
        print(f"Fallo en YOLO: {e}")
        img_orig = cv2.imread(img_path)
        img_yolo = img_orig.copy()
        roi = img_orig.copy()
        
    # Paso 2: Deskew
    roi_recta = deskew_image(roi)
    
    # Paso 3: Preprocesamiento src
    roi_preprocesada = preprocess_image_for_ocr(roi_recta)
    
    # Paso 4: OCR Flexible
    total = OCR_flexible_total(roi_preprocesada)
    
    # Imprimir resultado por terminal como se requirió
    print("\n" + "*"*60)
    print(f"MODELO YOLO UTILIZADO: {os.path.basename(modelo_path)}")
    if total is not None:
        print(f"💰 VALOR PREDICHO (TOTAL A PAGAR): {total}")
    else:
        print("❌ VALOR PREDICHO: No se pudo extraer el total.")
    print("*"*60 + "\n")
    
    # Mostrar imágenes comparativas (Original, YOLO, Recorte, Preprocesado)
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    plt.title("1. Original")
    plt.axis("off")
    
    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(img_yolo, cv2.COLOR_BGR2RGB))
    plt.title(f"2. Reconocimiento {os.path.basename(modelo_path)}")
    plt.axis("off")
    
    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.title("3. Recorte Original de YOLO")
    plt.axis("off")
    
    plt.subplot(1, 4, 4)
    plt.imshow(roi_preprocesada, cmap='gray')
    
    titulo_ocr = f"4. Recorte Procesado OCR\nTotal extraído: {total}" if total else "4. Procesado OCR\nTotal: No encontrado"
    plt.title(titulo_ocr)
    plt.axis("off")
    
    plt.tight_layout()
    # Puesto que puede romperse plt.show() en terminal/ tmux se guarda y se intenta mostrar.
    safe_name = os.path.basename(modelo_path).replace('.', '_')
    out_path = f"resultado_pipeline_{safe_name}_{os.path.basename(img_path)}"
    plt.savefig(out_path, dpi=150)
    print(f"Gráfico guardado en {out_path}")
    
    return total

if __name__ == "__main__":
    import glob
    modelos = ['yolov8n.pt', 'yolov10n.pt', 'yolo11n.pt']
    RAW_DATA_DIR = os.path.join("..", "data", "raw")
    image_paths = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.jpg")))[:3]

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        for m in modelos:
            ejecutar_pipeline_completo(img_path, m)
    else:
        for ipath in image_paths:
            print(f"\nEvaluating multiple YOLO models for: {os.path.basename(ipath)}")
            for m in modelos:
                try:
                    ejecutar_pipeline_completo(ipath, m)
                except Exception as e:
                    print(f"Error con modelo {m}: {e}")

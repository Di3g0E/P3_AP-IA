import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from src.utils.image_processing import crop_document

def evaluate_performance_and_accuracy(img_bgr, title, ocr_engine):
    """
    Evalúa el tiempo de procesamiento y de extracción de OCR dependiendo 
    de si la imagen es reescalada al superar 1200px de ancho.
    """
    img_bgr = crop_document(img_bgr)

    h_orig, w_orig = img_bgr.shape[:2]
    MAX_WIDTH = 1200
    if w_orig > MAX_WIDTH:
        ratio = MAX_WIDTH / float(w_orig)
        img_downscaled = cv2.resize(img_bgr, (MAX_WIDTH, int(h_orig * ratio)))
    else:
        img_downscaled = img_bgr

    h_down, w_down = img_downscaled.shape[:2]

    _, raw_lists_down, prep_crops_down = ocr_engine.process_and_extract(img_downscaled)

    val_down = ocr_engine.find_total_value(raw_lists_down.get('Footer', []))
    if not val_down: val_down = ocr_engine.find_total_value(raw_lists_down.get('Body', []))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"{title} | Extracción Final: {val_down}", fontweight='bold', fontsize=15)

    axes[0].imshow(cv2.cvtColor(img_downscaled, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Factura Recortada Auto-Encuadrada (Width: {w_down})")
    axes[0].axis('off')

    reconst_down = np.zeros((h_down, w_down), dtype=np.uint8)
    reconst_down[0:int(0.25*h_down), 0:w_down] = prep_crops_down['Header']
    reconst_down[int(0.25*h_down):int(0.80*h_down), 0:w_down] = prep_crops_down['Body']
    reconst_down[int(0.80*h_down):h_down, 0:w_down] = prep_crops_down['Footer']

    axes[1].imshow(reconst_down, cmap='gray')
    axes[1].set_title(f"Extremos Filtrados (Sin Fondo Oscuro)")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

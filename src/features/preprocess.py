import cv2
import numpy as np


def preprocess_image(img, ocr_engine):
    """Preprocesa una imagen para el OCR."""
    h_orig, w_orig = img.shape[:2]
    MAX_WIDTH = 1200
    if w_orig > MAX_WIDTH:
        ratio = MAX_WIDTH / float(w_orig)
        img_downscaled = cv2.resize(img, (MAX_WIDTH, int(h_orig * ratio)))
    else:
        img_downscaled = img

    h_down, w_down = img_downscaled.shape[:2]

    _, raw_lists_down, prep_crops_down = ocr_engine.process_and_extract(img_downscaled)

    val_down = ocr_engine.find_total_value(raw_lists_down.get('Footer', []))
    if not val_down: val_down = ocr_engine.find_total_value(raw_lists_down.get('Body', []))

    reconst_down = np.zeros((h_down, w_down), dtype=np.uint8)
    reconst_down[0:int(0.25*h_down), 0:w_down] = prep_crops_down['Header']
    reconst_down[int(0.25*h_down):int(0.80*h_down), 0:w_down] = prep_crops_down['Body']
    reconst_down[int(0.80*h_down):h_down, 0:w_down] = prep_crops_down['Footer']

    return val_down, reconst_down
    
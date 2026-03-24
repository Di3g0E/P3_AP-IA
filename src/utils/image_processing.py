import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def crop_document(img_bgr: np.ndarray) -> np.ndarray:
    """
    Aísla el papel de la factura analizando bordes y contornos.
    
    Args:
        img_bgr (np.ndarray): Imagen original en formato BGR.
        
    Returns:
        np.ndarray: Imagen recortada con el área de la factura o la imagen original si falla.
    """
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        
        # Dilatamos un poco los bordes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edged, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Filtramos contornos muy pequeños que puedan ser ruido
            valid_contours = [c for c in contours if cv2.contourArea(c) > 100]

            if valid_contours:
                # Concatenamos todos los contornos válidos
                all_points = np.vstack(valid_contours)
                
                # Obtenemos el rectángulo que enmarca a todos los contornos
                x, y, w, h = cv2.boundingRect(all_points)
                
                # Devolvemos el recorte con un pequeño margen de seguridad si es posible
                margin = 10
                y1 = max(0, y - margin)
                y2 = min(img_bgr.shape[0], y + h + margin)
                x1 = max(0, x - margin)
                x2 = min(img_bgr.shape[1], x + w + margin)
                
                return img_bgr[y1:y2, x1:x2]
        
        return img_bgr
    except Exception as e:
        logger.error(f"Error procesando el recorte del documento: {e}")
        return img_bgr

def preprocess_image_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Mejora la imagen original usando dilatación y umbralización OTSU 
    para preparar el texto para el modelo OCR.
    
    Args:
        img_bgr (np.ndarray): Imagen en formato BGR o su recorte principal.
        
    Returns:
        np.ndarray: Imagen lista y umbralizada para el motor de OCR.
    """
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        dilated_img = cv2.dilate(gray, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(gray, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        _, thresh = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return thresh
    except Exception as e:
        logger.error(f"Error en el preprocesamiento de la imagen para el OCR: {e}")
        return img_bgr

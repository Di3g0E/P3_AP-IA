import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def crop_document(img_bgr: np.ndarray) -> np.ndarray:
    """
    Aísla el papel de la factura usando técnicas avanzadas para fondos complejos.
    
    Args:
        img_bgr (np.ndarray): Imagen original en formato BGR.
        
    Returns:
        np.ndarray: Imagen recortada con el área de la factura o la imagen original si falla.
    """
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape
        
        # Técnica 1: Detección basada en diferencias de color/textura
        # Convertimos a espacio de color HSV para mejor separación fondo-documento
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Detectamos áreas claras (típico de papel)
        light_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        
        # Detectamos áreas con baja saturación (típico de documento)
        low_sat_mask = cv2.inRange(hsv, (0, 0, 50), (180, 40, 255))
        
        # Combinamos máscaras
        paper_mask = cv2.bitwise_or(light_mask, low_sat_mask)
        
        # Limpiamos la máscara
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, kernel)
        paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN, kernel)
        
        # Técnica 2: Detección de bordes tradicional como fallback
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Combinamos ambas técnicas
        combined = cv2.bitwise_or(paper_mask, edges)
        
        # Encontramos contornos
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            valid_contours = []
            
            for c in contours:
                area = cv2.contourArea(c)
                
                # Eliminar contornos muy pequeños
                if area > 5000:
                    x, y, w, h = cv2.boundingRect(c)
                    
                    # Solo rectángulos con proporción razonable
                    aspect_ratio = w / h
                    if 0.5 <= aspect_ratio <= 2.0: 
                        coverage = area / (h_img * w_img)
                        if coverage > 0.03:  # Al menos 3%
                            roi = gray[y:y+h, x:x+w]
                            mean_brightness = np.mean(roi)
                            if mean_brightness > 100:  # Área relativamente clara
                                valid_contours.append(c)

            if valid_contours:
                # Elegimos el contorno más grande y mejor proporcionado
                best_contour = max(valid_contours, key=lambda c: cv2.contourArea(c))
                x, y, w, h = cv2.boundingRect(best_contour)
                
                # Margen ajustado
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
    Mejora la imagen original usando técnicas adaptativas según las características
    de la imagen para optimizar el reconocimiento de texto.
    
    Args:
        img_bgr (np.ndarray): Imagen en formato BGR o su recorte principal.
        
    Returns:
        np.ndarray: Imagen lista y umbralizada para el motor de OCR.
    """
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Análisis de la imagen para decidir la mejor técnica
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Técnica 1: Para imágenes con bajo contraste o ruido
        if std_brightness < 30 or mean_brightness < 100:
            # Mejorar contraste agresivamente
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Reducción de ruido
            denoised = cv2.medianBlur(enhanced, 3)
            
            # Técnica original con imagen mejorada
            dilated_img = cv2.dilate(denoised, np.ones((5, 5), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 15)
            diff_img = 255 - cv2.absdiff(denoised, bg_img)
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            _, thresh = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
        # Técnica 2: Para imágenes normales
        else:
            # Técnica original pero más suave
            dilated_img = cv2.dilate(gray, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(gray, bg_img)
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            _, thresh = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Post-procesamiento: limpiar pequeños artefactos
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return thresh
    except Exception as e:
        logger.error(f"Error en el preprocesamiento de la imagen para el OCR: {e}")
        return img_bgr

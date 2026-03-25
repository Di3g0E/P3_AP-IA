import easyocr
import re
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

from src.features.region_detector import SimpleRegionDetector
from src.utils.image_processing import preprocess_image_for_ocr

logger = logging.getLogger(__name__)

class OptimizedOCREngine:
    """
    Motor OCR que coordina la detección de regiones en una factura y 
    la limpieza y extracción de texto usando EasyOCR.
    """
    def __init__(self, use_gpu: bool = True):
        """
        Inicializa el motor OCR optimizado.
        
        Args:
            use_gpu (bool): Prioridad de uso de GPU si está disponible.
        """
        try:
            self.reader = easyocr.Reader(['es', 'en'], gpu=use_gpu) 
            logger.info("EasyOCR corriendo en GPU (o forzado a intentarlo).")
        except Exception as e:
            logger.warning(f"Error al inicializar GPU. Recurriendo a CPU: {e}")
            self.reader = easyocr.Reader(['es', 'en'], gpu=False)
            logger.info("EasyOCR corriendo en CPU.")

    def process_and_extract(self, original_img: np.ndarray) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, np.ndarray]]:
        """
        Segmenta, preprocesa y ejecuta OCR sobre la imagen original.
        
        Args:
            original_img (np.ndarray): Imagen original del recibo.
            
        Returns:
            Tuple: 
                - texto extraído por bloque
                - raw lists de resultados de OCR
                - preprocessed crops por bloque
        """
        detector = SimpleRegionDetector()
        crops = detector.crop_regions(original_img)
        extracted_data = {}
        raw_lists = {}
        preprocessed_crops = {} 

        for region_name, crop in crops.items():
            clean_crop = preprocess_image_for_ocr(crop)
            preprocessed_crops[region_name] = clean_crop
            results = self.reader.readtext(clean_crop, detail=0)
            raw_lists[region_name] = results
            extracted_data[region_name] = " ".join(results).strip()

        return extracted_data, raw_lists, preprocessed_crops

    def find_total_value(self, raw_text_list: List[str]) -> Optional[float]:
        """
        Busca maximizar el valor TOTAL en una lista de bloques horizontales extraídos 
        del recibo por el OCR, validando con regex y lógicas secuenciales horizontales.
        
        Args:
            raw_text_list (List[str]): Resultados parciales del OCR en una región (por lo general Footer).
            
        Returns:
            Optional[float]: Valor flotante total deducido o None si no hay match.
        """
        # No quiero que se tenga en cuenta cuando pone "Total Qty" o "Total Items"
        # Por lo que voy a filtrar esas palabras
        raw_text_list = [text for text in raw_text_list if "Total Qty" not in text and "Total Items" not in text]
        keyword_regex = re.compile(r'\b(TOTAL|TOTAI|TL|AMT|DUE)\b', re.IGNORECASE)
        number_regex = re.compile(r'([$€£]?\s*\d+[\.,\s]*\d*)')

        for i, text in enumerate(raw_text_list):
            if keyword_regex.search(text.upper()):
                match_kw = keyword_regex.search(text.upper())
                right_part = text[match_kw.end():]

                full_search_text = right_part + " " + " ".join(raw_text_list[i+1:i+4])
                matches = number_regex.findall(full_search_text)

                valid_numbers = []
                for m in matches:
                    clean_num = re.sub(r'[^\d\.]', '', m.replace(',', '.'))
                    if clean_num:
                        try:
                            valid_numbers.append(float(clean_num))
                        except ValueError:
                            pass

                if valid_numbers:
                    return max(valid_numbers)
        return None

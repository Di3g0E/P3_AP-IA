import numpy as np
from typing import Dict

class SimpleRegionDetector:
    """
    Clase encargada de dividir un documento fiscal en 3 zonas principales: Header, Body, y Footer
    basado en porcentajes estáticos de su altura.
    """
    def crop_regions(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Divide una imagen del recibo/factura en 3 sectores funcionales.
        
        Args:
            img (np.ndarray): Imagen BGR del documento a dividir.
            
        Returns:
            Dict[str, np.ndarray]: Diccionario con Header, Body y Footer segmentado.
        """
        h, w = img.shape[:2]
        regions = {
            'Header': img[0:int(0.25 * h), 0:w],
            'Body': img[int(0.25 * h):int(0.80 * h), 0:w],
            'Footer': img[int(0.80 * h):h, 0:w]
        }
        return regions

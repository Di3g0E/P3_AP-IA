import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
from datasets import load_dataset


def load_cord_sample() -> Optional[Tuple[np.ndarray, str]]:
    """Descarga una muestra de CORD desde HuggingFace."""
    try:
        dataset = load_dataset("naver-clova-ix/cord-v2", split="train", streaming=True)
        sample = next(iter(dataset))
        img_bgr = cv2.cvtColor(np.array(sample['image']), cv2.COLOR_RGB2BGR)
        return img_bgr, "CORD (Naver-Clova)"
    except Exception as e:
        print(f"\nError cargando muestra de CORD: {e}\n")
        return None

def load_cord_training_data():
    """Carga los datos de entrenamiento de CORD."""
    try:
        dataset = load_dataset("naver-clova-ix/cord-v2", split="train", streaming=True)
        return dataset, "CORD (Naver-Clova)"
    except Exception as e:
        print(f"\nError cargando datos de CORD: {e}\n")
        return None

def load_cord_test_data():
    """Carga los datos de test de CORD."""
    try:
        dataset = load_dataset("naver-clova-ix/cord-v2", split="test", streaming=True)
        return dataset, "CORD (Naver-Clova)"
    except Exception as e:
        print(f"\nError cargando datos de CORD: {e}\n")
        return None

def load_voxel51_sample() -> Optional[Tuple[np.ndarray, str]]:
    """Descarga una muestra de Voxel51 desde HuggingFace."""
    try:
        dataset = load_dataset("Voxel51/high-quality-invoice-images-for-ocr", split="train", streaming=True)
        sample = next(iter(dataset))
        img_bgr = cv2.cvtColor(np.array(sample['image']), cv2.COLOR_RGB2BGR)
        return img_bgr, "Voxel51 High-Res Invoices"
    except Exception as e:
        print(f"\nError cargando muestra de Voxel51: {e}\n")
        return None

def load_images(path="data/raw"):
    """Carga todas las imágenes del directorio data/raw."""
    base_path = Path(__file__).resolve().parent.parent
    images_path = base_path.parent / path
    
    images = []
    for image_path in images_path.iterdir():
        img = cv2.imread(str(image_path))
        images.append(img)
    return images

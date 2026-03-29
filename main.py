"""
Extracción de totales en facturas: PaddleOCR + Gradient Boosting.

Uso:
  # Imagen individual
  python main.py --image data/raw/factura.png

  # Directorio de imágenes
  python main.py --dir data/raw

  # Cámara (captura una foto al pulsar espacio)
  python main.py --camera
"""

import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path

from src.models.ocr_engine import OptimizedOCREngine


def process_image(engine: OptimizedOCREngine, image: np.ndarray, name: str):
    """Ejecuta el pipeline sobre una imagen y muestra el resultado."""
    total = engine.extract_total(image)
    if total is not None:
        print(f"  {name:>30}  ->  Total: {total:,.2f}")
    else:
        print(f"  {name:>30}  ->  Total: no encontrado")
    return total


def run_on_image(engine: OptimizedOCREngine, path: str):
    """Carga y procesa una imagen individual."""
    img = cv2.imread(path)
    if img is None:
        print(f"Error: no se pudo leer '{path}'")
        return
    process_image(engine, img, os.path.basename(path))


def run_on_directory(engine: OptimizedOCREngine, dir_path: str):
    """Procesa todas las imágenes de un directorio."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    files = sorted(
        f for f in Path(dir_path).iterdir()
        if f.suffix.lower() in exts
    )
    if not files:
        print(f"No se encontraron imagenes en '{dir_path}'")
        return

    print(f"Procesando {len(files)} imagenes de '{dir_path}':\n")
    for f in files:
        img = cv2.imread(str(f))
        if img is not None:
            process_image(engine, img, f.name)


def run_camera(engine: OptimizedOCREngine):
    """Captura una imagen de la cámara y extrae el total."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: no se pudo abrir la camara")
        return

    print("Camara abierta. Pulsa ESPACIO para capturar, ESC para salir.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camara - ESPACIO para capturar, ESC para salir", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 32:  # ESPACIO
            print("Captura realizada, procesando...")
            process_image(engine, frame, "camara")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Extraccion de totales en facturas (PaddleOCR + GB)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Ruta a una imagen")
    group.add_argument("--dir", type=str, help="Directorio con imagenes")
    group.add_argument("--camera", action="store_true", help="Usar camara")
    args = parser.parse_args()

    print("=" * 50)
    print("  Extraccion de Totales — PaddleOCR + GB")
    print("=" * 50)
    print()

    engine = OptimizedOCREngine()
    print()

    if args.image:
        run_on_image(engine, args.image)
    elif args.dir:
        run_on_directory(engine, args.dir)
    elif args.camera:
        run_camera(engine)


if __name__ == "__main__":
    main()

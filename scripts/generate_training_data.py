import easyocr
import json
import os
import sys
from datasets import load_dataset
import numpy as np
import argparse

# Añadir el separador de directorios correcto y la ruta del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features.preprocess import preprocess_image

def generate_training_data(limit=None, preprocess=False, dataset_type="train"):
    """
    Genera datos de entrenamiento a partir del dataset CORD.
    Extrae el texto completo mediante EasyOCR y el total de la factura desde el ground truth.
    """
    print("Inicializando EasyOCR...")
    reader = easyocr.Reader(['es', 'en'])

    print("Cargando dataset CORD...")
    dataset = load_dataset("naver-clova-ix/cord-v2", split=dataset_type, streaming=True)

    if preprocess:
        output_path = os.path.join("data", "processed", f"cord_{dataset_type}_data_preprocessed.jsonl")
    else:
        output_path = os.path.join("data", "processed", f"cord_{dataset_type}_data.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Procesando imágenes y guardando en {output_path}...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(dataset):
            if limit and i >= limit:
                break
            
            try:
                # 1. Extraer etiqueta real (total_price)
                gt = json.loads(sample["ground_truth"])
                total_price = gt["gt_parse"]["total"]["total_price"]
                
                # Si el total es None o no existe, saltamos la muestra
                if total_price is None:
                    continue

                # 2. Extraer texto mediante OCR
                img_array = np.array(sample['image'])
                if preprocess:
                    img_array = preprocess_image(img_array, reader)
                results = reader.readtext(img_array)
                full_text = " ".join([r[1] for r in results])

                # 3. Guardar en formato JSONL
                data_point = {
                    "text": full_text,
                    "label": total_price,
                    "image_id": i
                }
                f.write(json.dumps(data_point, ensure_ascii=False) + "\n")
                f.flush()
                
                if (i + 1) % 10 == 0:
                    print(f"Procesadas {i + 1} imágenes...")

            except Exception as e:
                print(f"Error procesando muestra {i}: {e}")
                continue

    print(f"Proceso finalizado. Datos guardados en {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera datos de entrenamiento a partir del dataset CORD.")
    parser.add_argument("--limit", type=int, default=None, help="Límite de imágenes a procesar.")
    parser.add_argument("--preprocess", default=False, action="store_true", help="Preprocesa las imágenes antes de generar los datos.")
    parser.add_argument("--dataset_type", type=str, default="train", choices=["train", "test", "validation"], help="Tipo de dataset a procesar (train, test, validation).")
    args = parser.parse_args()
    # Por defecto procesamos las primeras 100 para probar, o elimina el límite para todo el dataset
    generate_training_data(limit=args.limit, preprocess=args.preprocess, dataset_type=args.dataset_type)

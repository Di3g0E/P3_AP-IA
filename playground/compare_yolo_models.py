import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==========================================
# Configuración y Rutas
# ==========================================
RAW_DATA_DIR = os.path.join("..", "data", "raw")
image_paths = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.jpg")))[:3]
modelos_nombres = ['yolov8n.pt', 'yolov10n.pt', 'yolo11n.pt']

def compare_yolo_detections():
    if not image_paths:
        print(f"No se encontraron imágenes en {RAW_DATA_DIR}")
        return

    # Preparamos el canvas matplotlib (3 modelos x 3 imágenes)
    fig, axes = plt.subplots(len(modelos_nombres), len(image_paths), 
                             figsize=(18, 5 * len(modelos_nombres)), squeeze=False)
    
    for model_idx, model_name in enumerate(modelos_nombres):
        print(f"Caragando y evaluando {model_name}...")
        try:
            model = YOLO(model_name, verbose=False)
        except Exception as e:
            print(f"Error cargando {model_name}: {e}")
            continue

        for img_idx, img_path in enumerate(image_paths):
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Inferencia
            results = model(img, verbose=False)
            
            # Dibujar detecciones
            img_draw = img_rgb.copy()
            cajas = results[0].boxes.xyxy.cpu().numpy()
            confianzas = results[0].boxes.conf.cpu().numpy()
            
            if len(cajas) > 0:
                for i, (box, conf) in enumerate(zip(cajas, confianzas)):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f"Inv {conf:.2f}"
                    cv2.putText(img_draw, label, (x1, max(y1-10, 0)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            ax = axes[model_idx, img_idx]
            ax.imshow(img_draw)
            ax.set_title(f"{model_name}\nImg: {os.path.basename(img_path)}", fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    out_path = "comparativa_yolo_invoice_detection.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n¡Comparativa completada! Guardada en: {out_path}")

if __name__ == "__main__":
    compare_yolo_detections()

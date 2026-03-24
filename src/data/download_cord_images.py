import os
import cv2
import numpy as np
from datasets import load_dataset

def download_cord_samples(num_images=20):
    # Ruta absoluta apuntando a data/raw
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw'))
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Descargando {num_images} imágenes de CORD a: {save_dir} ...")
    
    try:
        # Cargar dataset de streaming
        dataset = load_dataset("naver-clova-ix/cord-v2", split="train", streaming=True)
        count = 0
        
        for sample in dataset:
            img_bgr = cv2.cvtColor(np.array(sample['image']), cv2.COLOR_RGB2BGR)
            filename = os.path.join(save_dir, f"cord_receipt_{count:03d}.jpg")
            cv2.imwrite(filename, img_bgr)
            
            print(f"[{count+1}/{num_images}] Guardada: {filename}")
            count += 1
            if count >= num_images:
                break
                
        print("\n¡Descarga y guardado completados con éxito!")
        
    except Exception as e:
        print(f"Error durante la descarga: {e}")

if __name__ == "__main__":
    download_cord_samples()

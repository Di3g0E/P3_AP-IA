from src.data.data_loader import load_images
from src.features.preprocess import preprocess_image
from src.models.ocr_engine import OptimizedOCREngine
from src.evaluation.evaluate import evaluate_performance_and_accuracy

def main_1():
    ocr_engine = OptimizedOCREngine()
    images = load_images()
    for i, img in enumerate(images[3:8]):
        evaluate_performance_and_accuracy(img, f"Imagen {i+4}", ocr_engine)


def main():
    images = load_images()
    ocr_engine = OptimizedOCREngine()
    
    for i, img in enumerate(images[3:8]):
        val, prep_img = preprocess_image(img, ocr_engine)
        print(f"Imagen {i+1}: {val}")

#TODO: ver si dejamos o no la palabra subtotal -> Confusiones
if __name__ == '__main__':
    main_1()

from src.data.data_loader import load_images
from src.data.data_download import save_image
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
    
    for i, img in enumerate(images[:]):
        raw_lists, prep_img = preprocess_image(img, ocr_engine)
        val_down = ocr_engine.find_total_value(raw_lists.get('Footer', []))

        if not val_down: val_down = ocr_engine.find_total_value(raw_lists.get('Body', []))
        print(f"Imagen {i}: {val_down}")


if __name__ == '__main__':
    main()

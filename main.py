from src.data.data_loader import load_images
from src.features.preprocess import preprocess_image
from src.models.ocr_engine import OptimizedOCREngine


def main_1():
    ocr_engine = OptimizedOCREngine()
    images = load_images()
    for img in images[:3]:
        evaluate_performance_and_accuracy(img, "Raw Image", ocr_engine)


def main():
    images = load_images()
    ocr_engine = OptimizedOCREngine()
    
    for i, img in enumerate(images[:3]):
        val, prep_img = preprocess_image(img, ocr_engine)
        print(f"Imagen {i+1}: {val}")


if __name__ == '__main__':
    main()

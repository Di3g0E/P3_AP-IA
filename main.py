import matplotlib.pyplot as plt

from src.data.data_loader import load_images
from src.features.preprocess import preprocess_image
from src.models.ocr_engine import OptimizedOCREngine

from src.evaluation.evaluate import evaluate_performance_and_accuracy


def main_1():
    ocr_engine = OptimizedOCREngine()

    # loaders = [
    #     load_cord_sample,
    #     load_voxel51_sample
    # ]

    # for i, loader_fn in enumerate(loaders, 1):
    #     result = loader_fn()
    #     if result:
    #         img_bgr, title = result
    #         evaluate_performance_and_accuracy(img_bgr, title, ocr_engine)
    #     else:
    #         print(f"Dataset omitido debido a un error previo.")

    images = load_images()
    for img in images[:3]:
        evaluate_performance_and_accuracy(img, "Raw Image", ocr_engine)

def main():
    images = load_images()
    ocr_engine = OptimizedOCREngine()
    for img in images[:3]:
        val, prep_img = preprocess_image(img, ocr_engine)
        print(val)

        plt.imshow(prep_img, cmap='gray')
        plt.title(f"Preprocesada | Total: {val}")
        plt.axis('off')
        plt.show()



if __name__ == '__main__':
    main()
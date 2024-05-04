import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import my_utils


def load_image(image_path: str):
    # load image as grayscale
    return cv2.imread(image_path, 0)


def adaptive_mean(image: cv2.typing.MatLike, image_path: str = ""):
    image = get_image(image, image_path)
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5
    )


def adaptive_mean_fix1(image: cv2.typing.MatLike, image_path: str = ""):
    image = get_image(image, image_path)
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 111, 13
    )
    # ทำ morphological closing เพื่อลบช่องว่างในเส้นของมือ
    kernel = np.ones((3, 3), np.uint8)  # เปลี่ยน kernel เป็น (3, 3)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # ทำ morphological opening เพื่อลบ noise เล็กๆ
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image


def adaptive_gaussian(image: cv2.typing.MatLike, image_path: str = ""):
    image = get_image(image, image_path)
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5
    )


def get_image(image: cv2.typing.MatLike, image_path: str) -> ...:
    # Read as grayscale
    return (
        cv2.imread(image_path, 0)
        if image_path
        else image if image is not None else None
    )


def equalize_histogram(image: cv2.typing.MatLike, image_path: str = ""):
    image = get_image(image, image_path)
    # Perform histogram equalization on the reference image
    return cv2.equalizeHist(image)


def denoise(image: cv2.typing.MatLike, image_path: str = ""):
    image = get_image(image, image_path)
    return cv2.fastNlMeansDenoising(image, None)


def denoise_morphology(image: cv2.typing.MatLike, image_path: str = ""):
    image = get_image(image, image_path)
    kernel = np.ones((5, 5), np.uint8)

    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    return image


def plot(image1: cv2.typing.MatLike):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Reference image and its histogram
    axs[0, 0].imshow(image1, cmap="gray")
    axs[0, 0].set_title("Reference Image")
    # axs[0, 1].hist(image1.flatten(), bins=256, color='black', alpha=0.7)
    # axs[0, 1].set_title('Histogram')
    # axs[0, 1].set_xlabel('Pixel Value')
    # axs[0, 1].set_ylabel('Frequency')

    # plt.tight_layout()
    plt.show()


def plot_datas(datas):
    num_rows = len(datas)
    print(f"data count: {num_rows}")
    fig, axs = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

    for i, row in enumerate(datas):
        img1, img2, img3 = row

        # Plot image1
        axs[i, 0].imshow(img1, cmap="gray")
        axs[i, 0].set_title("Original")

        # Plot image2
        axs[i, 1].imshow(img2, cmap="gray")
        axs[i, 1].set_title("Process 1")

        # Plot image3
        axs[i, 2].imshow(img3, cmap="gray")
        axs[i, 2].set_title("Process 2")

        # Plot someGraph
        # axs[i, 2].plot(graph_data)
        # axs[i, 2].set_title('Some Graph')

    plt.tight_layout()
    plt.show()


def export_image(
    image: cv2.typing.MatLike, file_name: str, folder_path: str = "output/"
):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name)
    cv2.imwrite(file_path, image)


def export_masking_dataset(output_path: str = "data/Palm/output/"):
    dataset_path = my_utils.dataset_path
    images = os.listdir(dataset_path)
    print(f"Export image: {output_path}")
    for image_file in images:
        if image_file.endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")
        ):
            image = load_image(os.path.join(dataset_path, image_file))
            image_adapt = adaptive_mean_fix1(image=image)

            export_image(image_adapt, file_name=image_file, folder_path=output_path)
            print(f"export image: {image_file}")

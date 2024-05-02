import cv2
import numpy as np
import os


class ImageMaskingBuilder:
    def __init__(self, image_path: str):
        self.kernel_size = None
        self.image_path: str = image_path
        self.image: np.ndarray = self.__load_image_grayscale()

        self.set_kernel_size()
        self.convert_image_grayscale()

    # TODO: draft
    def __load_image(self) -> np.ndarray:
        return cv2.imread(self.image_path)

    def __load_image_grayscale(self) -> np.ndarray:
        return cv2.imread(self.image_path, 0)

    # TODO: Draft
    def convert_image_grayscale(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def set_kernel_size(self, size: int = 3):
        self.kernel_size: np.ndarray = np.ones((size, size), np.uint8)

    def do_adaptive_mean(self, block_size: int = 199, constant: int = 5):
        self.image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, block_size, constant)
        return self

    def do_adaptive_gaussian(self, block_size: int = 199, constant: int = 5):
        self.image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block_size, constant)
        return self

    def do_adaptive_mean_fix1(self):
        return self.do_adaptive_mean(111, 13).do_denoise_morphology_close().do_denoise_morphology_open()

    def do_equalize_histogram(self):
        self.image = cv2.equalizeHist(self.image)
        return self

    def do_denoise_morphology_close(self):
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, self.kernel_size)
        return self

    def do_denoise_morphology_open(self):
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, self.kernel_size)
        return self

    def do_denoise(self):
        self.image = cv2.fastNlMeansDenoising(self.image, None)
        return self

    def do_denoise_morphology_combined(self):
        return self.do_denoise_morphology_close().do_denoise_morphology_open()

    def do_sobel_combined(self):
        sobel_x = cv2.Sobel(self.image, cv2.CV_64F, 1, 0)
        sobel_y = cv2.Sobel(self.image, cv2.CV_64F, 0, 1)
        self.image = cv2.bitwise_or(sobel_x, sobel_y)
        return self

    def do_sobel_x(self):
        self.image = cv2.Sobel(self.image, cv2.CV_64F, 1, 0)
        return self

    def do_sobel_y(self):
        self.image = cv2.Sobel(self.image, cv2.CV_64F, 0, 1)
        return self

    def do_canny(self, min_threshold: float = 10, max_threshold: float = 200):
        self.image = cv2.Canny(self.image, min_threshold, max_threshold)
        return self

    def do_laplacian(self):
        temp_transform = cv2.Laplacian(self.image, cv2.CV_64F, ksize=5)
        # self.image = np.uint8(np.absolute(temp_transform))
        self.image = np.absolute(temp_transform).astype(np.uint8)
        return self

    def export_image(self, file_name: str, folder_path: str = "../output/"):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, file_name)
        cv2.imwrite(file_path, self.image)

    def reset_image(self):
        self.image = self.__load_image_grayscale()

import cv2
import numpy as np

from utility.file_utils import *
from utility.project import *
from .image_utils import *
from .image import *


class ImageBuilder:
    def __init__(self, image_path: str = ""):
        self.__temp_image: Image = Image(image_path)
        self.__processed_image: cv2.typing.MatLike = self.__temp_image.image
        self.set_kernel_size()
        self.kernel_size: np.ndarray = np.zeros((3, 3), np.uint8)

    def set_kernel_size(self, size: int = 3):
        self.kernel_size: np.ndarray = np.ones((size, size), np.uint8)

    def do_adaptive_mean(self, block_size: int = 199, constant: int = 5):
        self.__processed_image = cv2.adaptiveThreshold(
            self.__processed_image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            constant,
        )
        return self

    def do_adaptive_gaussian(self, block_size: int = 199, constant: int = 5):
        self.__processed_image = cv2.adaptiveThreshold(
            self.__processed_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            constant,
        )
        return self

    def do_adaptive_mean_fix1(self):
        return (
            self.do_adaptive_mean(111, 13)
            .do_denoise_morphology_close()
            .do_denoise_morphology_open()
        )

    def do_equalize_histogram(self):
        self.__processed_image = cv2.equalizeHist(self.__processed_image)
        return self

    def do_denoise_morphology_close(self):
        self.__processed_image = cv2.morphologyEx(self.__processed_image, cv2.MORPH_CLOSE, self.kernel_size)
        return self

    def do_denoise_morphology_open(self):
        self.__processed_image = cv2.morphologyEx(self.__processed_image, cv2.MORPH_OPEN, self.kernel_size)
        return self

    def do_denoise(self):
        self.__processed_image = cv2.fastNlMeansDenoising(self.__processed_image, None)
        return self

    def do_denoise_morphology_combined(self):
        return self.do_denoise_morphology_close().do_denoise_morphology_open()

    # TODO: Fix
    def do_sobel_combined(self):
        sobel_x = cv2.Sobel(self.__processed_image, cv2.CV_64F, 1, 0)
        sobel_y = cv2.Sobel(self.__processed_image, cv2.CV_64F, 0, 1)
        self.__processed_image = cv2.bitwise_or(sobel_x, sobel_y)
        return self

    def do_sobel_x(self):
        self.__processed_image = cv2.Sobel(self.__processed_image, cv2.CV_64F, 1, 0)
        return self

    def do_sobel_y(self):
        self.__processed_image = cv2.Sobel(self.__processed_image, cv2.CV_64F, 0, 1)
        return self

    def do_canny(self, min_threshold: float = 10, max_threshold: float = 200):
        self.__processed_image = cv2.Canny(self.__processed_image, min_threshold, max_threshold)
        return self

    def do_laplacian(self):
        temp_transform = cv2.Laplacian(self.__processed_image, cv2.CV_64F, ksize=5)
        # self.image = np.uint8(np.absolute(temp_transform))
        self.__processed_image = np.absolute(temp_transform).astype(np.uint8)
        return self

    def flip_image(self, flip_code: int = 0):
        self.__processed_image = cv2.flip(self.__processed_image, flip_code)
        return self

    def build(self) -> Image:
        return self.__temp_image

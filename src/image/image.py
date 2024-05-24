import cv2
from .image_utils import *


class Image:
    def __init__(self, image_path: str, load_type: ImageLoadType = ImageLoadType.COLOR):
        self.kernel_size = None
        self.image_path: str = image_path
        self.image: cv2.typing.MatLike = self.__load_image(load_type)

    def __load_image(self, load_type: ImageLoadType) -> cv2.typing.MatLike:
        loaded_image = load_image(self.image_path, load_type)
        check_loaded_image(loaded_image)
        print(f"type: {type(loaded_image)}")
        return loaded_image

    def __load_image_grayscale(self) -> cv2.typing.MatLike:
        loaded_image = self.__load_image(ImageLoadType.GRAYSCALE)
        return loaded_image

    def export(self, file_name: str, folder_path: str = "../output/") -> None:
        export_image(self.image, file_name, folder_path)

    def reset(self) -> None:
        self.image = self.__load_image_grayscale()

    def show(self, window_name: str = "My Image", size_x: int = 500, size_y: int = 650) -> None:
        show_image(image=self.image, window_name=window_name, size_x=size_x, size_y=size_y)

    def get_contours(self):
        return cv2.findContours(self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

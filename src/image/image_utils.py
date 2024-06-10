import numpy as np
import cv2
import matplotlib.pyplot as plt
from utility.file_utils import *
from enum import Enum


class ImageLoadType(int, Enum):
    UNCHANGED = cv2.IMREAD_UNCHANGED
    GRAYSCALE = cv2.IMREAD_GRAYSCALE
    COLOR = cv2.IMREAD_COLOR


def show_image(
        image: cv2.typing.MatLike,
        window_name: str = "My Image",
        size_x: int = 600,
        size_y: int = 600,
):
    # cv2.resizeWindow(window_name, 400, 400)
    if image is None:
        print(f"image not pass")
        return
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, size_x, size_y)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)


def plot_image(
        image: cv2.typing.MatLike,
        size=None
):
    if size is None:
        size = [600, 600]
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")  # Hide the axes
    plt.show()


def load_image(image_path: str, flag_code: ImageLoadType = ImageLoadType.UNCHANGED):
    return cv2.imread(image_path, flag_code)


def check_loaded_image(image: cv2.typing.MatLike):
    assert image is not None, "MyDebug: imread error"


def export_image(image: cv2.typing.MatLike, file_name: str, folder_path: str = "../output/"):
    if folder_path is None:
        folder_path = "../output"
    else:
        check_path_compatibility(folder_path)
    file_path = join_path(folder_path, file_name)
    write_image(file_path, image)


def write_image(file_path: str, image: cv2.typing.MatLike) -> None:
    cv2.imwrite(file_path, image)

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

path = "../data/Palm/After1/IMG_0016.JPG"
path_best = "../data/mod/palm-tone-edit.png"
dataset_path = "../data/Palm/After1"


def resize_image_show(image, width=600, height=600, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def plot_datas(datas: list[dict]):
    fig, axes = plt.subplots(
        nrows=len(datas), ncols=len(datas[0]), figsize=(12, 4 * len(datas))
    )

    for i, data_row in enumerate(datas):
        for j, (key, img) in enumerate(data_row.items()):
            axes[i, j].imshow(img, cmap="gray")
            axes[i, j].axis("off")  # Turn off axis
            axes[i, j].set_title(key)
    plt.tight_layout()
    plt.show()


def load_image(image_path: str):
    return cv2.imread(image_path, 0)


def export_image(
    image: cv2.typing.MatLike, file_name: str, folder_path: str = "output/"
):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name)
    cv2.imwrite(file_path, image)


def export_masking_dataset(output_path: str = "data/Palm/output/"):
    images = os.listdir(dataset_path)
    print(f"Export image: {output_path}")
    for image_file in images:
        if image_file.endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")
        ):
            image = load_image(os.path.join(dataset_path, image_file))

            export_image(image, file_name=image_file, folder_path=output_path)
            print(f"export image: {image_file}")


def check_path_compatibility(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def show_image(
    window_name: str = "My Image",
    image: np.ndarray = None,
    size_x: int = 600,
    size_y: int = 600,
):
    # cv2.resizeWindow(window_name, 400, 400)
    if image is None:
        print(f"image not pass")
        return
    cv2.imshow(window_name, cv2.resize(image, (size_x, size_y)))


def check_loaded_image(image: np.ndarray):
    assert image is not None, "MyDebug: imread error"


def get_my_image_path():

    os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    image_path = os.path.join(dir_path, path)

    print(f"image_path: {image_path}")

    img = cv2.imread(image_path, 1)

    print(f"is loaded image: ", check_loaded_image(img))


def change_to_project_path():
    script_path = os.path.dirname(os.path.abspath(__file__))
    current_path = os.getcwd()

    if current_path != script_path:
        os.chdir(script_path)
        print("Changed current path to script path:", script_path)
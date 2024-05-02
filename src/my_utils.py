import cv2
from matplotlib import pyplot as plt
import os
import my_masking_2 as mk2

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
    fig, axes = plt.subplots(nrows=len(datas), ncols=len(datas[0]), figsize=(12, 4 * len(datas)))

    for i, data_row in enumerate(datas):
        for j, (key, img) in enumerate(data_row.items()):
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')  # Turn off axis
            axes[i, j].set_title(key)
    plt.tight_layout()
    plt.show()


def load_image(image_path: str):
    return cv2.imread(image_path, 0)


def export_image(image: cv2.typing.MatLike, file_name: str, folder_path: str = "output/"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name)
    cv2.imwrite(file_path, image)


def export_masking_dataset(output_path: str = "data/Palm/output/"):
    images = os.listdir(dataset_path)
    print(f'Export image: {output_path}')
    for image_file in images:
        if image_file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')):
            image = load_image(os.path.join(dataset_path, image_file))

            export_image(image, file_name=image_file, folder_path=output_path)
            print(f"export image: {image_file}")

import os
import logging
import cv2
import matplotlib.pyplot as plt


def change_to_main_root():
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    print(f"Current directory: {current_dir}")

    while current_dir != os.path.dirname(current_dir):
        if "src" in os.listdir(current_dir):
            project_root = current_dir
            os.chdir(project_root)
            print(f"Changed to project root: {project_root}")
            return
        current_dir = os.path.dirname(current_dir)


def check_loaded_image(image: cv2.typing.MatLike):
    assert image is not None, "MyDebug: imread error"


def is_path_exists(file_path: str) -> bool:
    return os.path.exists(file_path)


def check_path_exists(file_path: str) -> None:
    assert is_path_exists(file_path), f"MyDebug: {file_path} - image not found"


def check_path_compatibility(folder_path: str):
    if not is_path_exists(folder_path):
        os.makedirs(folder_path)


def get_filename_without_extension(file_path: str) -> str:
    base_name = os.path.basename(file_path)
    file_name, _ = os.path.splitext(base_name)
    return file_name


def save_comparison(image_path: str, mask_path: str, name: str = "test"):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_image = cv2.imread(mask_path)

    _, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs = axs.ravel()

    axs[0].axis("off")
    axs[0].imshow(image)

    axs[1].axis("off")
    axs[1].imshow(mask_image, cmap="gray")

    plt.savefig(name + ".png")


def save_result_original(
    image_path: str,
    mask_image: cv2.typing.MatLike,
    mask_pred_image: cv2.typing.MatLike,
    name: str = "test",
):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_image = mask_image.astype(int)
    mskp = mask_pred_image
    _, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs = axs.ravel()

    axs[0].axis("off")
    axs[0].imshow(img / 255.0)

    axs[1].axis("off")
    axs[1].imshow(mask_image * 255, cmap="gray")

    axs[2].axis("off")
    axs[2].imshow(mskp * 255, cmap="gray")

    plt.savefig(name + ".png")

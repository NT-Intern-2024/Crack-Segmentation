import os
import logging
import cv2
import matplotlib.pyplot as plt


def set_logger_config():
    logging_format = "%(asctime)s - %(message)s"
    logging.basicConfig(filename="all-process.log", format=logging_format, level=logging.INFO)


def change_to_project_path(main_script_path: str):
    """
    Change the current working directory to the directory where the script is located.
    """
    try:
        script_path = os.path.dirname(os.path.abspath(main_script_path))
        current_path = os.getcwd()

        logging.info(f"Current path: {current_path}")

        if current_path != script_path:
            os.chdir(script_path)
            print(f"current path: {os.getcwd()}")
            logging.info(f"Changed current path to script path: {script_path}")
        else:
            logging.info("Current path is already the script path.")
    except Exception as e:
        logging.error(f"Error changing to script path: {e}")

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


def save_comparison(image_path: str, mask_path: str, name: str ='test'):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_image = cv2.imread(mask_path)

    _, axs = plt.subplots(1, 2, figsize=(10,5))
    axs = axs.ravel()

    axs[0].axis('off')
    axs[0].imshow(image)

    axs[1].axis('off')
    axs[1].imshow(mask_image, cmap= 'gray')

    plt.savefig(name + '.png')

def save_result_original(image_path: str, mask_image: cv2.typing.MatLike, mask_pred_image: cv2.typing.MatLike, name: str ='test'):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_image = mask_image.astype(int)
    mskp = mask_pred_image
    _, axs = plt.subplots(1, 3, figsize=(15,5))
    axs = axs.ravel()

    axs[0].axis('off')
    axs[0].imshow(img/255.)

    axs[1].axis('off')
    axs[1].imshow(mask_image*255, cmap= 'gray')

    axs[2].axis('off')
    axs[2].imshow(mskp*255, cmap= 'gray')

    plt.savefig(name + '.png')


logger = logging.getLogger(__name__)
set_logger_config()

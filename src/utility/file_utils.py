import os

import yaml
import logging


def is_path_exists(folder_path: str) -> bool:
    return os.path.exists(folder_path)


def list_directory(folder_path: str) -> list[str]:
    if not is_path_exists(folder_path):
        raise FileNotFoundError(f"Directory '{folder_path}' not found.")
    return os.listdir(folder_path)


def check_path_exists(folder_path: str, error_text: str = None) -> None:
    logging.info(f"check path: {folder_path}")
    if not is_path_exists(folder_path):
        output_error_text = error_text if error_text is not None else f"'{folder_path}' not found."
        logging.error(f"path not exists")
        raise FileNotFoundError(output_error_text)


def load_file(file_path: str):
    with open(file_path, 'r') as file:
        if file_path.endswith(".yaml"):
            loaded_file = yaml.safe_load(file)
    return loaded_file


def check_path_compatibility(folder_path: str):
    if not is_path_exists(folder_path):
        os.makedirs(folder_path)


def join_path(parent_path: str, child_path: str):
    return os.path.join(parent_path, child_path)


def get_filename_without_extension(file_path: str) -> str:
    base_name = os.path.basename(file_path)
    file_name, _ = os.path.splitext(base_name)
    return file_name
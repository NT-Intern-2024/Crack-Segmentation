import os
import logging


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
            logging.info(f"Changed current path to script path: {script_path}")
        else:
            logging.info("Current path is already the script path.")
    except Exception as e:
        logging.error(f"Error changing to script path: {e}")


def change_to_main_root():
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    while current_dir != os.path.dirname(current_dir):
        if "src" in os.listdir(current_dir):
            project_root = current_dir
            os.chdir(project_root)
            return
        current_dir = os.path.dirname(current_dir)

import os
import logging


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

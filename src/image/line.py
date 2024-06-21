from utility.file_utils import *
from utility.logger import *

import cv2
import numpy as np

def export_image_from_lines(width: int, height: int, lines: list, output_pattern_name: str = "line", output_path: str = ""):
    line_count = 1
    check_path_compatibility(output_path)

    for line in lines:
        image_name = "{}-{}".format(output_pattern_name, line_count)
        export_image_from_line(width, height, line, image_name, output_path)
        line_count += 1

def export_image_from_line(width: int, height: int, line: list, output_name: str = "line", output_path: str = ""):
    check_path_compatibility(output_path)

    binary_image = np.zeros((width, height), dtype=np.uint8)
    for point in line:
        binary_image[point[0], point[1]] = 255
    image_path = "{}/{}.png".format(output_path, output_name)
    cv2.imwrite(image_path, binary_image)

    logger.info(f"Save image: {image_path}")
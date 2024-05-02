import my_utils
import my_masking_2 as mk2
import cv2
import matplotlib.pyplot as plt
import numpy as np

# image_path = my_utils.path
image_path = "../output/IMG_0001.JPG"

image_file = mk2.ImageMaskingBuilder(image_path)

image_file.do_canny()
contours, _ = image_file.get_contours()

# Initialize list to store points
points_list = []

# Iterate over contours
for i, contour in enumerate(contours):
    print(f" ------------- contour {i} ---------------")

    # Iterate over points in contour
    for point in contour:
        x, y = point[0]  # Extract x, y coordinates
        points_list.append((x, y))  # Add point to list

# Plot the points on the image
for point in points_list:
    plt.plot(point[0], point[1], 'ro')  # Plot each point as a red dot

# Show the plot
plt.show()

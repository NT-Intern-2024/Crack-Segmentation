import my_utils
import my_masking_2 as mk2
import cv2
import matplotlib.pyplot as plt
import numpy as np

# image_path = my_utils.path
image_path = "../output/IMG_0001.JPG"

image_file = mk2.ImageMaskingBuilder(image_path)
image_file.do_canny()

# Find contours
contours, _ = cv2.findContours(image_file.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a list to store contours with points
contours_with_points = []

# Generate distinct colors for contours
num_contours = len(contours)
distinct_colors = [np.random.rand(3, ) for _ in range(num_contours)]

# Iterate over contours
for i, contour in enumerate(contours):
    # Initialize a list to store points in the current contour
    contour_points = []

    # Get color for the current contour
    color = distinct_colors[i]
    print(f"contour: {i}")

    # Iterate over points in contour
    for point in contour:
        x, y = point[0]  # Extract x, y coordinates
        contour_points.append((x, y))  # Add point to contour_points list

    # Add the contour with its points and color to the list
    contours_with_points.append((contour_points, color))

print("---------- plotting -------------")

# Plot each contour with points
for contour_points, color in contours_with_points:
    # Extract x and y coordinates from contour points
    x = [point[0] for point in contour_points]
    y = [point[1] for point in contour_points]

    # Plot points with the specified color
    plt.scatter(x, y, color=color)

plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
plt.show()

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

# Initialize list to store points
points_list = []

# Iterate over contours
for i, contour in enumerate(contours):
    # Generate a random color for each contour
    color = np.random.randint(0, 256, 3).tolist()  # Generate random RGB color

    print(f"contour {i}, {color}")
    # Iterate over points in contour
    for point in contour:
        x, y = point[0]  # Extract x, y coordinates
        points_list.append((x, y, color))  # Add point and color to list

        plt.scatter(x, y, color=np.array(color) / 255)
        print(f"contour {i} \t plot: {x},{y}")

# Plot points
# for x, y, color in points_list:
# plt.scatter(x, y, color=np.array(color) / 255)  # Divide by 255 to scale to [0,1] for Matplotlib

plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
plt.show()

# for point in points_list:
#     plt.plot(point[0], point[1], 'ro')  # Plot each point as a red dot

# Display or further process the points_list as needed
# print(points_list)

# plt.imshow(image.image)
# plt.title('Original Image with Edge Points')
# plt.axis('off')


# height, width = image_file.image.shape
# output_image = np.zeros((height, width, 3), np.uint8)
#
# # Draw the points on the blank image
# for point in points_list:
#     x, y = point
#     cv2.circle(output_image, (x, y), 2, (0, 255, 0), -1)  # Draw a green circle at each point
#
# # Save the image with the points
# cv2.imwrite('output.jpg', output_image)
#
# # Optionally, display the image
# cv2.imshow('Output Image with Points', output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Show the plot
# plt.show()

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from utility.project import *

change_to_main_root()
# Load the binary skeletonized image
image = cv2.imread('./output/good_sample_skel/image256-skel.png', cv2.IMREAD_GRAYSCALE)

# Ensure the image is binary
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Extract features from contours
features = []
for contour in contours:
    # Calculate the moments to find the centroid
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])  # Centroid x
        cy = int(M['m01'] / M['m00'])  # Centroid y
    else:
        cx, cy = 0, 0

    # Calculate the area and perimeter of the contour
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Calculate the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Append the features
    features.append([cx, cy, area, perimeter, w, h])

features = np.array(features)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=0).fit(features)
labels = kmeans.labels_

# Create an output image to visualize the clusters
output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

# Define colors for clusters
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]

# Draw contours with cluster colors
for i, contour in enumerate(contours):
    color = colors[labels[i] % len(colors)]
    cv2.drawContours(output_image, [contour], -1, color, 2)

# Display the clustered image
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title('Clustered Palm Lines')
plt.show()

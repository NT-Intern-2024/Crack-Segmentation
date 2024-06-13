import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the binary skeletonized image
image_path = './output/good_sample_skel/image256-skel.png'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

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

    # Calculate the orientation of the contour
    if len(contour) >= 5:  # At least 5 points are needed to fit an ellipse
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
    else:
        angle = 0

    # Append the features
    features.append([cx, cy, area, perimeter, w, h, angle])

features = np.array(features)

# Check if we have enough features
if len(features) < 2:
    print("Not enough features to cluster.")
else:
    # Use PCA to reduce features to 2D for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    # Plot the features
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1])
    plt.title('Feature Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from sklearn.cluster import KMeans
from utility.project import *
from image.transform import *
from image.image_utils import *


def preprocess_image(image_path):
    # Load the skeletonized image
    skeleton_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    check_loaded_image(skeleton_image)
    return skeleton_image

def extract_features(skeleton_image):
    # Apply connected component labeling
    num_labels, labels_im = cv2.connectedComponents(skeleton_image)

    # Extract properties of labeled regions
    regions = measure.regionprops(labels_im)

    # Extract features (centroid, orientation, length)
    features = []
    for region in regions:
        if region.area > 10:  # Filter out small regions
            centroid = region.centroid
            orientation = region.orientation
            length = region.major_axis_length
            features.append((centroid, orientation, length))

    return features, regions

def cluster_features(features, n_clusters=4):
    # Prepare the feature matrix for clustering (using orientation and length)
    X = np.array([[f[1], f[2]] for f in features])  # Example: using orientation and length

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    labels = kmeans.labels_

    return labels

def visualize_clusters(skeleton_image, regions, labels, n_clusters=4):
    # Create a blank image for visualization
    output_image = np.zeros((skeleton_image.shape[0], skeleton_image.shape[1], 3), dtype=np.uint8)

    # Define colors for each cluster
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    # Draw the lines with different colors based on their cluster label
    for i, region in enumerate(regions):
        if region.area > 10:
            coords = region.coords
            for coord in coords:
                output_image[coord[0], coord[1]] = colors[labels[i] % n_clusters]

    # Plot the original and clustered images
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(skeleton_image, cmap='gray'), plt.title('Skeletonized Image')
    plt.subplot(122), plt.imshow(output_image), plt.title('Clustered Lines')
    plt.show()

def main(image_path: str, test_image_path: str):
    skeleton_image = preprocess_image(image_path)
    features, regions = extract_features(skeleton_image)
    labels = cluster_features(features)
    visualize_clusters(skeleton_image, regions, labels)

    image = load_image(test_image_path)
    check_loaded_image(image)
    skeleton_image = skeletonize_image(image)

    features, regions = extract_features(skeleton_image)
    labels = cluster_features(features)
    visualize_clusters(skeleton_image, regions, labels)

if __name__ == "__main__":
    change_to_main_root()
    image_path = './output/good_sample_skel/image104-skel.png'  # Replace with the path to your skeletonized image
    test_image_path  =  "./sample/line-complex-100x100.png"
    main(image_path, test_image_path)



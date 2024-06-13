import cv2
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def extract_features(image_path):
    # Load the binary skeletonized image
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
    
    return features, contours

def dbscan_clustering(features, eps, min_samples):
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    labels = dbscan.labels_
    
    return labels

def kmeans_clustering(features, n_clusters):
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    labels = kmeans.labels_
    
    return labels

def visualize_clusters(image_path, labels, contours):
    # Load the binary skeletonized image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Create an output image to visualize the clusters
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Define colors for clusters
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    
    # Draw contours with cluster colors
    for i, contour in enumerate(contours):
        label = labels[i]
        color = colors[label % len(colors)]
        cv2.drawContours(output_image, [contour], -1, color, 2)
    
    # Display the clustered image
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title('Clustered Palm Lines')
    plt.show()

def main():
    # Example usage
    # image_path = 'skeletonized_palm.png'
    # image_path = './output/good_sample_skel/image396-skel.png'
    image_path = './output/good_sample_skel/image256-skel.png'


    
    # Extract features
    features, contours = extract_features(image_path)
    
    # Perform DBSCAN clustering
    eps = 10
    min_samples = 1
    dbscan_labels = dbscan_clustering(features, eps, min_samples)
    
    # Perform K-Means clustering
    n_clusters = 4
    # kmeans_labels = kmeans_clustering(features, n_clusters)
    
    # Visualize DBSCAN clusters
    visualize_clusters(image_path, dbscan_labels, contours)
    
    # Visualize K-Means clusters
    # visualize_clusters(image_path, kmeans_labels, contours)

if __name__ == "__main__":
    main()

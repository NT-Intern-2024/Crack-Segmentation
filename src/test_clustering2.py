import cv2
import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

def extract_features_2(image_path):
    # Load the binary skeletonized image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure the image is binary
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract features (for example, centroid and area)
    features = []
    for contour in contours:
        # Calculate the moments to find the centroid
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])  # Centroid x
            cy = int(M['m01'] / M['m00'])  # Centroid y
        else:
            cx, cy = 0, 0
    
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
    
        # Append the features
        features.append([cx, cy, area])
    
    features = np.array(features)
    
    return features, contours

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

def graph_based_clustering(features, adjacency_matrix, n_clusters):
    # Perform Spectral Clustering based on the adjacency matrix
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed').fit(adjacency_matrix)
    labels = clustering.labels_
    
    return labels

def construct_adjacency_matrix(contours):
    # Construct an adjacency matrix based on intersections between contours
    n_contours = len(contours)
    adjacency_matrix = np.zeros((n_contours, n_contours))
    
    for i in range(n_contours):
        for j in range(i + 1, n_contours):
            if contours_intersect(contours[i], contours[j]):
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
    
    return adjacency_matrix

def contours_intersect(contour1, contour2):
    # Implement logic to check if contours intersect
    # This can involve checking bounding boxes, intersection points, etc.
    # Example logic:
    rect1 = cv2.boundingRect(contour1)
    rect2 = cv2.boundingRect(contour2)
    
    intersect = not (rect1[0] > rect2[0] + rect2[2] or
                     rect1[0] + rect1[2] < rect2[0] or
                     rect1[1] > rect2[1] + rect2[3] or
                     rect1[1] + rect1[3] < rect2[1])
    
    return intersect

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
    image_path = './output/good_sample_skel/image396-skel.png'
    # image_path = './output/good_sample_skel/image256-skel.png'
    
    # Extract features
    features, contours = extract_features(image_path)
    
    # Construct adjacency matrix based on intersections
    adjacency_matrix = construct_adjacency_matrix(contours)
    
    # Perform graph-based clustering
    n_clusters = 4  # Adjust as needed
    labels = graph_based_clustering(features, adjacency_matrix, n_clusters)
    
    # Visualize clusters
    visualize_clusters(image_path, labels, contours)

if __name__ == "__main__":
    main()

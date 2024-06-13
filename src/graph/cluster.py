import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D

def plot_silhouette_scores(data, max_k: int):
    silhouette_scores = []

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    for k in range(1, max_k + 1):
        _, labels, _ = cv2.kmeans(data.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        sihouette_avg = silhouette_score(data, labels)
        silhouette_scores.append(sihouette_avg)

    plt.plot(range(1, max_k + 1), silhouette_scores, marker="o")
    plt.xlabel("Num of Cluster (k)")
    plt.ylabel("Sihoutte Score")
    plt.title("Diff k")
    plt.show()

def plot_cluster_pca(data, n_clusters):
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data.astype(np.float32), n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    if data.shape[1] > 2:
        print(f"data.shape[1] > 2")
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)
        reduced_center = pca.transform(centers)
    else:
        reduced_data = data
        reduced_center = centers
    
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis")
    plt.scatter(reduced_center[:, 0], reduced_center[:, 1], s=300, c="red", marker="x")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"Cluters Vis with k={n_clusters}")
    plt.show()

    # For simplicity, let's assume we're working with the first two dimensions of the features
    # If you have more than 2 dimensions and want to visualize all, you can use 3D plots or PCA for dimensionality reduction

# def plot_data(data, centers):
#     # Extracting the first two dimensions of the features for plotting
#     data_to_plot = data[:, :2]

#     # Extracting the first two dimensions of the centers
#     centers_to_plot = centers[:, :2]

#     # Plot the data points
#     plt.scatter(data_to_plot[:, 0], data_to_plot[:, 1], c=label.flatten(), s=50, cmap='viridis')
#     # Plot the centers
#     plt.scatter(centers_to_plot[:, 0], centers_to_plot[:, 1], c='red', s=200, alpha=0.75, marker='X')

#     plt.title('K-means Clustering')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.show()

def plot_dbscan(data):
    db = DBSCAN(min_samples=5).fit(data)
    labels = db.labels_

    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis")
    plt.title("DBSCAN")
    plt.xlabel("Feat 1")
    plt.ylabel("Feat 2")
    plt.show()
    

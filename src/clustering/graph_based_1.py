import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

# Example points representing key points along lines
points = np.array([
    [0, 0], [1, 1], [2, 2],  # Line 1
    [10, 0], [11, 1], [12, 2],  # Line 2
    [5, 5], [6, 5], [7, 5],  # Line 3
])

# Create a graph
G = nx.Graph()

# Add nodes
for i, point in enumerate(points):
    G.add_node(i, pos=point)

# Add edges based on Euclidean distance
threshold_distance = 2
for i in range(len(points)):
    for j in range(i + 1, len(points)):
        distance = np.linalg.norm(points[i] - points[j])
        if distance < threshold_distance:
            G.add_edge(i, j, weight=distance)

# Visualize the graph
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True)
plt.show()

# Apply Spectral Clustering
adj_matrix = nx.adjacency_matrix(G)
spectral = SpectralClustering(n_clusters=3, affinity='precomputed')
labels = spectral.fit_predict(adj_matrix.toarray())

# Visualize the clustering result
for label in set(labels):
    cluster_points = points[labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}')
plt.legend()
plt.show()

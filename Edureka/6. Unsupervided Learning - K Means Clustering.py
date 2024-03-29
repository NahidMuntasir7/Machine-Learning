# unsupervided learning: k means clustering

from sklearn.cluster import KMeans
import numpy as np

# Generate some sample data
data = np.random.rand(100, 2)  # 100 samples, 2 features

# Create KMeans instance
kmeans = KMeans(n_clusters=3)  # Number of clusters you want to create

# Fit the model to the data
kmeans.fit(data)

# Get cluster labels
labels = kmeans.labels_

# Get cluster centroids
centroids = kmeans.cluster_centers_

# Print the cluster labels and centroids
print("Cluster Labels:", labels)
print("Cluster Centroids:", centroids)

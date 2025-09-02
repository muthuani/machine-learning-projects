from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate a synthetic dataset with 3 clusters
X, y = make_blobs(n_samples=300, centers=3, random_state=42)

# Apply KMeans clustering (3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get the cluster labels
labels = kmeans.labels_

#Visualize the clustering result
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KMeans Clustering on make_blobs Dataset')
plt.show()
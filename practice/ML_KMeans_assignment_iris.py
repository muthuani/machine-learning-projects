from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (sepal length, sepal width, etc.)

# Apply KMeans clustering (3 clusters since there are 3 species)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get the cluster labels (which cluster each point belongs to)
labels = kmeans.labels_

#Visualize the results (sepal length vs sepal width)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('KMeans Clustering on Iris Dataset')
plt.show()
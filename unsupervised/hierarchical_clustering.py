import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage

print("Running Hierarchical Clustering...")

# Load dataset
data = pd.read_csv("train.csv")

# Remove target column
X = data.drop("price_range", axis=1)

# Take a sample (dendrogram is clearer)
X_sample = X.sample(n=100, random_state=42)

# Perform hierarchical clustering
linked = linkage(X_sample, method='ward')

print("Clustering completed")

# Plot dendrogram
plt.figure(figsize=(12, 6))

dendrogram(linked)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")

plt.show()

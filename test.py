import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Perform PCA with 3 components
pca = PCA(n_components=3, svd_solver='full')
X_r = pca.fit_transform(X)

# Create scatter plots
plt.figure(figsize=(18, 6))

# Plot 1st vs. 2nd principal component
plt.subplot(131)
plt.scatter(X_r[:, 0], X_r[:, 1], c=y)
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')
plt.title('1st vs. 2nd Principal Component')

# Plot 1st vs. 3rd principal component
plt.subplot(132)
plt.scatter(X_r[:, 0], X_r[:, 2], c=y)
plt.xlabel('1st Principal Component')
plt.ylabel('3rd Principal Component')
plt.title('1st vs. 3rd Principal Component')

# Plot 2nd vs. 3rd principal component
plt.subplot(133)
plt.scatter(X_r[:, 1], X_r[:, 2], c=y)
plt.xlabel('2nd Principal Component')
plt.ylabel('3rd Principal Component')
plt.title('2nd vs. 3rd Principal Component')

plt.show()

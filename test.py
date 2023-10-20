import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # The feature matrix
y = iris.target  # The target labels

# Apply PCA for 3D reduction
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Apply t-SNE for 3D reduction
tsne = TSNE(n_components=3)
X_tsne = tsne.fit_transform(X)

# Create subplots for 1st vs. 2nd, 1st vs. 3rd, and 2nd vs. 3rd dimensions
fig = plt.figure(figsize=(15, 5))

# 1st vs. 2nd dimension
ax = fig.add_subplot(131, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
ax.set_xlabel('1st Principal Component')
ax.set_ylabel('2nd Principal Component')
ax.set_title('PCA: 1st vs. 2nd')

# 1st vs. 3rd dimension
ax = fig.add_subplot(132, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 2], c=y, cmap='viridis')
ax.set_xlabel('1st Principal Component')
ax.set_ylabel('3rd Principal Component')
ax.set_title('PCA: 1st vs. 3rd')

# 2nd vs. 3rd dimension
ax = fig.add_subplot(133, projection='3d')
ax.scatter(X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis')
ax.set_xlabel('2nd Principal Component')
ax.set_ylabel('3rd Principal Component')
ax.set_title('PCA: 2nd vs. 3rd')

plt.show()

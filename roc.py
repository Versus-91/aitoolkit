import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import pandas as pd

iris = load_iris()
data = iris.data
le = LabelEncoder()
df = pd.read_csv(os.path.join('datasets', "iris.csv"))
pca = PCA(n_components=3)
X = df.iloc[:, :-1]
y = le.fit_transform(df['Species'])
transformed_data = pca.fit_transform(X=X)

# Create scatter plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

scatter = axes[0].scatter(transformed_data[:, 0],
                          transformed_data[:, 2], c=y, cmap='viridis')
axes[0].set_xlabel(f"PC 1")
axes[0].set_ylabel(f"PC 3")

axes[1].scatter(transformed_data[:, 0], transformed_data[:, 1],
                c=y, cmap='viridis')
axes[1].set_xlabel(f"PC 1")
axes[1].set_ylabel(f"PC 2")

axes[2].scatter(transformed_data[:, 1], transformed_data[:, 2],
                c=y, cmap='viridis')
axes[2].set_xlabel(f"PC 2")
axes[2].set_ylabel(f"PC 3")

# Create a legend with class names
unique_labels = list(set(y))
class_names = le.classes_
legend_patches = [mpatches.Patch(color=scatter.cmap(scatter.norm(
    label)), label=class_names[label]) for label in unique_labels]
plt.legend(handles=legend_patches, title="Classes", loc='upper right')

plt.show()

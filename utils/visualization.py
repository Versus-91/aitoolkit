

import os
from matplotlib import patches, pyplot as plt
from sklearn.calibration import LabelEncoder
import pandas as pd
from sklearn.decomposition import PCA


def draw_pca(data_frame, y, y_type=None):
    le = LabelEncoder()
    pca = PCA(n_components=3)
    X = data_frame.iloc[:, :-1]
    y = le.fit_transform(data_frame[y])
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
    legend_patches = [patches.Patch(color=scatter.cmap(scatter.norm(
        label)), label=class_names[label]) for label in unique_labels]
    plt.legend(handles=legend_patches, title="Classes", loc='upper right')
    return fig

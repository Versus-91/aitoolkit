from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
items = pd.read_csv('iris.csv')

# Assuming that your CSV file has columns for features (e.g., 'sepal_length', 'sepal_width', 'petal_length', 'petal_width')
X = items.iloc[:, :-1]  # Features
y = label_encoder.fit_transform(items.iloc[:, -1])

# # Split the data into training and testing sets
# for i, item in enumerate(data.columns):
#     print(item)

label_encoder = LabelEncoder()
iris = items
X = items.iloc[:, :-1]
y = label_encoder.fit_transform(items.iloc[:, -1])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
# Standardize the feat
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Create a logistic regression model
model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
# Train the model on the training
model.fit(X_train, y_train)
# Make predictions on test data
y_pred = model.predict(X_test)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
# Standardize the feat
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Apply t-SNE for dimensionalieduction and visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
# Apply PCA for dimensionality reion and visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
# Plot the t-SNE and PCA res
a = plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.title("t-SNE Visualization")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title("PCA Visualization")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

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
# Load the Iris dataset
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# Replace 'iris.csv' with the actual path to your CSV file
data = pd.read_csv('iris.csv')

# Assuming that your CSV file has columns for features (e.g., 'sepal_length', 'sepal_width', 'petal_length', 'petal_width')
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target variable in the last column
# Split the data into training and testing sets
for i, item in enumerate(data.columns):
    print(item)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Create a logistic regression model
model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE for dimensionality reduction and visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Apply PCA for dimensionality reduction and visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the t-SNE and PCA results
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
# display(sns.pairplot(items, hue="target"), target="mpl2")
# display(a, target="mpl")
print(f"Accuracy: {accuracy * 100:.2f}%")

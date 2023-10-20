from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import asyncio
import panel as pn
from pyodide.ffi import create_proxy
from panel.io.pyodide import show
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
from js import document, console, Uint8Array, window, File
import io
from sklearn.preprocessing import LabelEncoder
file_input = pn.widgets.FileInput(accept='.csv', width=180)
button_upload = pn.widgets.Button(
    name='Upload', button_type='primary', width=100)
row = pn.Row(file_input, button_upload, height=75)
data_frame = None
table = pn.widgets.Tabulator(pagination='remote', page_size=10)
document.getElementById('table').style.display = 'none'


async def process_file(e):
    global data_frame
    console.log("Attempted file upload: " + e.target.value)
    file_list = e.target.files
    first_item = file_list.item(0)
    if len(file_list) == 1:
        array_buf = Uint8Array.new(await first_item.arrayBuffer())
        bytes_list = bytearray(array_buf)
        data_frame = pd.read_csv(io.BytesIO(bytes_list))
        # table.value = df
        # document.getElementById('table').style.display = 'block'
        # test(df)


def test():
    # Load the Iris dataset
    label_encoder = LabelEncoder()
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Create a logistic regression model
    model = LogisticRegression(
        solver='lbfgs', multi_class='auto', max_iter=1000)
    # Train the model on the training data
    model.fit(X_train, y_train)
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
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

    display(a, target="mpl")


async def init():
    button_upload.on_click(process_file)
    await show(row, 'fileinput')
    await show(table, 'table')
upload_file = create_proxy(process_file)
document.getElementById("parseCVS").addEventListener("change", upload_file)

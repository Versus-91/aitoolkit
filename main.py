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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pyscript import Element
from sklearn.metrics import classification_report

from utils.kde import draw_kde
from utils.data_process import get_feature_types


def plot_confusion_matrix(confusion_matrix, lables):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=lables)
    disp.plot()
    return disp.figure_


async def visualize_output(model, x):
    coefficients = model.coef_
    feature_names = x.columns
    coef_df = pd.DataFrame(
        coefficients, columns=feature_names, index=model.classes_)
    output_table.value = coef_df
    document.getElementById('output_table').style.display = 'block'
    await show(output_table, target="output_table")


output_table = pn.widgets.Tabulator()
document.getElementById('output_table').style.display = 'none'
file_input = pn.widgets.FileInput(accept='.csv', width=180)
button_upload = pn.widgets.Button(
    name='Upload', button_type='primary', width=100)
row = pn.Row(file_input, button_upload, height=75)
data_frame = None


async def process_file(e):
    global data_frame
    console.log("Attempted file upload: " + e.target.value)
    file_list = e.target.files
    first_item = file_list.item(0)
    if len(file_list) == 1:
        array_buf = Uint8Array.new(await first_item.arrayBuffer())
        bytes_list = bytearray(array_buf)
        data_frame = pd.read_csv(io.BytesIO(bytes_list))
        # display(draw_kde(data_frame), target="cm")
        # table.value = df
        # document.getElementById('table').style.display = 'block'
        # test(df)


async def test(event):
    numerical, ordinal, nominal = get_feature_types(data_frame.columns)
    for item in numerical:
        print("numerical", item)
    for item in ordinal:
        print("ordinal", item)
    for item in nominal:
        print("nomial", item)
    target_element = document.getElementById("target")
    chart_element = Element("cm")
    chart_element.element.innerHTML = ""
    label_encoder = LabelEncoder()
    X = data_frame.iloc[:, :-1]
    y = data_frame.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    model = LogisticRegression(random_state=1, max_iter=500)
    model.fit(X=X_train, y=y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_pred=y_pred, y_true=y_test, labels=model.classes_)
    plot_confusion_matrix(cm, model.classes_)
    print(classification_report(y_true=y_test, y_pred=y_pred))
    await visualize_output(model, X)
    display(plot_confusion_matrix(cm, model.classes_), target="cm")


async def init():
    button_upload.on_click(process_file)
    await show(row, 'fileinput')
    await show(table, 'table')


def test_click(event):
    print(event.target.data)


upload_file = create_proxy(process_file)
run_at_click = create_proxy(test)
document.getElementById("parseCVS").addEventListener("change", upload_file)
document.getElementById("apply").addEventListener("click", run_at_click)

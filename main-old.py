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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report


def plot_confusion_matrix(confusion_matrix, lables):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=lables)
    fig = disp.plot()


items = pd.read_csv(os.path.join('datasets', "titanic.csv"))

label_encoder = LabelEncoder()

df = pd.DataFrame(pd.read_csv("Iris.csv"))
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
model = LogisticRegression(random_state=1, max_iter=500)
model.fit(X=X_train, y=y_train)
y_pred = model.predict(X_test)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)
accuracy = accuracy_score(y_test, y_pred)

# print(classification_report(y_true=y_test, y_pred=y_pred))
# cm = confusion_matrix(y_pred=y_pred, y_true=y_test, labels=model.classes_)
# # plot_confusion_matrix(cm, model.classes_)
# for label in np.unique(y_test):
#     plt.scatter(X_pca[y_test == label, 0], X_pca[y_test ==
#                 label, 1], label=f'Class {label}', s=60)

# for i, (true_label, predicted_label) in enumerate(zip(y_test, y_pred)):
#     if true_label != predicted_label:
#         plt.text(X_pca[i, 0], X_pca[i, 1], str(i), color='red', fontsize=12)

# plt.title(f'PCA Visualization of Model Results (Accuracy: {accuracy:.2f})')
# plt.legend()
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')

# plt.show()
kde = sns.kdeplot(data=df.loc[:, df.columns != 'Id'], fill=True, common_norm=False, palette="crest",
                  alpha=.5, linewidth=0)
plt.show()
print("done")

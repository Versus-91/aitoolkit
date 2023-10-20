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

label_encoder = LabelEncoder()
items = pd.read_csv('iris.csv')

# # Split the data into training and testing sets
# for i, item in enumerate(data.columns):
#     print(item)

label_encoder = LabelEncoder()
iris = items
X = items.iloc[:, :-1]
y = items.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
# # Standardize the feat
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# Create a logistic regression model
model = LogisticRegression()
# Train the model on the training
model.fit(X_train, y_train)
# Make predictions on test data
y_pred = model.predict(X_test)
print('The accuracy of the Logistic Regression is',
      accuracy_score(y_pred, y_test))

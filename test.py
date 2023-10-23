import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns
# Load the Iris dataset
df = pd.DataFrame(pd.read_csv("Iris.csv",index_col=False))
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
# Predict probabilities for each class
probs = model.predict_proba(X)

# Create a DataFrame to store the probabilities
prob_df = pd.DataFrame(data=probs, columns=model.classes_)
print(prob_df.head())
# Plot KDE of the predicted probabilities
plt.figure(figsize=(10, 6))
sns.kdeplot(data=prob_df, common_norm=False, fill=True)
plt.xlabel("Probability")
plt.ylabel("Density")
plt.title("KDE of Predicted Probabilities")
plt.show()
# Create a boxplot of the predicted probabilities
plt.figure(figsize=(10, 6))
sns.boxplot(data=prob_df)
plt.xlabel("Class")
plt.ylabel("Probability")
plt.title("Boxplot of Predicted Probabilities")
plt.show()

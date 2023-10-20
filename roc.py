from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Example true labels and predicted probabilities (replace with your actual data)
true_labels = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
predicted_probs = [0.8, 0.3, 0.6, 0.7, 0.2, 0.9, 0.4, 0.1, 0.75, 0.2]

# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)

# Calculate the AUC (Area Under the Curve)
roc_auc = roc_auc_score(true_labels, predicted_probs)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

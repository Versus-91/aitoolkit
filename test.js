// Require the necessary libraries
const { LogisticRegression } = require('ml-regression');
const { Matrix } = require('ml-matrix');

// Example dataset (features and labels)
const features = [
  [1, 2],
  [2, 3],
  [3, 4],
  // Add more feature vectors here
];
const labels = [0, 1, 1]; // Example binary classification labels (0 or 1)

// Create a matrix from the feature data
const X = new Matrix(features);

// Create a Logistic Regression model with L1 and L2 regularization
const model = new LogisticRegression(X, labels, {
  numSteps: 100, // Number of iterations
  learningRate: 0.1,
  lambda: 0.1, // Regularization parameter (adjust for L1 and L2)
  penalty: 'l1', // Type of penalty ('l1' for L1, 'l2' for L2)
});

// Train the model
model.train();

// Predict using the trained model
const newDataPoint = [4, 5]; // New data point for prediction
const prediction = model.predict(new Matrix([newDataPoint]));
console.log('Prediction:', prediction);

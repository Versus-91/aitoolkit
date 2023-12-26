// Import TensorFlow.js
const math = require("mathjs")
const tf = require('@tensorflow/tfjs-node');
const weights = tf.randomUniform([4], -1, 1);

class LinearRegression {
    constructor(iterations = 1000, lr = 0.1) {
        this.iterations = iterations;
        this.lr = lr;
        this.weights = null
    }
    fit(X, y) {
        const limit = 1 / Math.sqrt(X.shape[1]);
        const ones = tf.ones([X.shape[0], 1]);
        X = tf.concat([X, ones], 1);
        this.weights = tf.randomUniform([X.shape[1], 1], -limit, limit); // Adjusted initialization
        const learningRate = 0.01; // Define learning rate
        const iterations = 1000; // Define the number of iterations

        for (let i = 0; i < iterations; i++) {
            const y_pred = X.matMul(this.weights);
            const mse = tf.mean(tf.pow(tf.sub(y, y_pred), 2)).mul(0.5); // Calculate Mean Squared Error (MSE)
            const error = tf.sub(y_pred, y);
            const gradient = X.transpose().matMul(error).div(X.shape[0]); // Compute gradient
            const deltaWeights = gradient.mul(learningRate); // Multiply gradient by learning rate
            this.weights = this.weights.sub(deltaWeights); // Update weights
        }
    }
    predict(X) {
        const ones = tf.ones([X.shape[0], 1])
        X = tf.concat([X, ones], 1)
        return X.matMul(this.weights)
    }

}
const trainingData = {
    x: [[1, 2], [2, 3], [3, 4], [4, 5]], // Multiple features for each x
    y: [3, 5, 7, 9] // Corresponding y values
};

// Convert data to TensorFlow tensors
const X_train = tf.tensor2d(trainingData.x); // Multiple features in each x
const y_train = tf.tensor2d(trainingData.y, [trainingData.y.length, 1]);

var model = new LinearRegression()
model.fit(X_train, y_train)
const preds = model.predict(X_train)
preds.print()


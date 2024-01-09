// Generate some sample data
const tf = require('@tensorflow/tfjs-node');

const data = tf.tensor([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
]);
const labels = tf.tensor2d([
    [0],
    [0],
    [1],
    [1],
]);

// Define the model parameters
const learningRate = 0.1;
const iterations = 100;
const optimizer = tf.train.sgd(learningRate);

let weights = tf.variable(tf.randomNormal([2, 1]));
let bias = tf.variable(tf.scalar(Math.random()));

// Training the model
for (let i = 0; i < iterations; i++) {
    optimizer.minimize(() => {
        const predictions = data.matMul(weights).add(bias).sigmoid();
        const loss = tf.losses.sigmoidCrossEntropy(labels, predictions);
        return loss;
    });
}

// Making predictions
const newInput = tf.tensor([[5, 6]]);
const prediction = newInput.matMul(weights).add(bias).sigmoid();
prediction.print();

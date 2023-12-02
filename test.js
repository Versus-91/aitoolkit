const tf = require('@tensorflow/tfjs');
const {
    getClasses,
    getClassesAsNumber,
    getCrossValidationSets,
    getDataset,
    getDistinctClasses,
    getNumbers,
} = require('ml-dataset-iris')
class LogisticRegression {
    constructor(learningRate, iterations) {
        this.learningRate = learningRate;
        this.iterations = iterations;
        this.weights = null;
        this.classes = null;
    }

    train(features, targets) {
        const uniqueClasses = [...new Set(targets)];
        this.classes = uniqueClasses;

        // Convert targets to one-hot encoding
        const encodedTargets = targets.map(target =>
            uniqueClasses.map(cls => (cls === target ? 1 : 0))
        );

        const numFeatures = features[0].length;
        const numClasses = uniqueClasses.length;

        // Initialize weights with zeros
        this.weights = tf.zeros([numFeatures, numClasses]);

        // Convert features to TensorFlow tensors
        const xs = tf.tensor2d(features);

        // Gradient Descent
        for (let i = 0; i < this.iterations; i++) {
            const predictions = xs.matMul(this.weights).softmax();
            const error = tf.tensor2d(encodedTargets).sub(predictions);

            const gradient = xs.transpose().matMul(error).div(xs.shape[0]);
            const delta = gradient.mul(this.learningRate);

            this.weights = this.weights.add(delta);
        }
    }

    predict(features) {
        const input = tf.tensor2d(features);
        const prediction = input.matMul(this.weights).softmax().arraySync();
        input.dispose();
        return prediction;
    }
}

// Example usage:
const lr = new LogisticRegression(0.01, 1000);

const features = [
    [1.2, 0.7],
    [0.4, 0.5],
    [2.3, 1.9],
    [1.8, 1.0],
    [1.5, 1.5]
];

const targets = [0, 1, 2, 1, 2]; // Classes represented numerically
lr.train(getNumbers(), getClassesAsNumber());

const prediction = lr.predict(getNumbers());



function findMaxValueIndicesForEachArray(nestedArray) {
    return nestedArray.map(innerArray => {
        let maxVal = Number.NEGATIVE_INFINITY;
        let maxIndex = -1;

        innerArray.forEach((value, index) => {
            if (value > maxVal) {
                maxVal = value;
                maxIndex = index;
            }
        });

        return maxIndex;
    });
}

// Example usage:
const nestedArray = [
    [3, 5, 2],
    [7, 1, 9],
    [4, 8, 6]
];

const maxIndicesForEachArray = findMaxValueIndicesForEachArray(prediction);
let y = getClassesAsNumber()
let sum = 0
maxIndicesForEachArray.forEach((element, i) => {
    if (element === y[i]) {
        sum += 1
    } else {
        // console.log(element[i], y[i]);
    }
});
console.log(y);
console.log('True preds:', sum);

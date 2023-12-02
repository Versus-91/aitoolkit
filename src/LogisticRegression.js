export class LogisticRegression {
    constructor({ numFeatures, numClasses, learningRate, l1Regularization = 0, l2Regularization = 0 }) {
        this.numFeatures = numFeatures;
        this.numClasses = numClasses;
        this.learningRate = learningRate;
        this.l1Regularization = l1Regularization;
        this.l2Regularization = l2Regularization;


        this.weights = [];
        this.biases = [];

        for (let i = 0; i < numClasses; i++) {
            this.weights.push(window.tf.variable(window.tf.randomNormal([numFeatures, 1])));
            this.biases.push(window.tf.variable(window.tf.zeros([1])));
        }
    }

    _logisticRegression(inputs, classIndex) {
        const logits = window.tf.add(window.tf.matMul(inputs, this.weights[classIndex]), this.biases[classIndex]);
        return window.tf.sigmoid(logits);
    }

    _loss(predictions, targets) {
        let loss = window.tf.tidy(() => {
            const epsilon = 1e-7;
            const clippedPredictions = window.tf.clipByValue(predictions, epsilon, 1 - epsilon);
            const term1 = window.tf.mul(targets, window.tf.log(clippedPredictions));
            const term2 = window.tf.mul(window.tf.sub(1, targets), window.tf.log(window.tf.sub(1, clippedPredictions)));
            let loss = window.tf.neg(window.tf.mean(window.tf.add(term1, term2)));

            // L1 regularization
            if (this.l1Regularization !== 0) {
                const l1Loss = this.weights.reduce((acc, weight) => window.tf.add(acc, window.tf.sum(window.tf.abs(weight))), window.tf.scalar(0));
                loss = window.tf.add(loss, window.tf.mul(this.l1Regularization, l1Loss));
            }
            // L2 regularization
            if (this.l2Regularization !== 0) {
                const l2Loss = this.weights.reduce((acc, weight) => window.tf.add(acc, window.tf.sum(window.tf.square(weight))), window.tf.scalar(0));
                loss = window.tf.add(loss, window.tf.mul(this.l2Regularization, l2Loss));
            }
            return loss;
        })
        return loss;
    }

    fit(features, labels, epochs = 100) {
        const optimizer = window.tf.train.sgd(this.learningRate);

        for (let epoch = 0; epoch < epochs; epoch++) {
            for (let i = 0; i < this.numClasses; i++) {
                const classLabels = labels.slice([0, i], [labels.shape[0], 1]);

                optimizer.minimize(() => {
                    const predictions = this._logisticRegression(features, i);
                    const currentLoss = this._loss(predictions, classLabels);
                    return currentLoss;
                });
            }
        }
    }

    predict(inputs) {
        let probs = window.tf.tidy(() => {
            const scores = [];
            for (let i = 0; i < this.numClasses; i++) {
                const currentClassPrediction = this._logisticRegression(inputs, i);
                scores.push(currentClassPrediction);
            }
            const classProbabilities = window.tf.stack(scores, 1);
            return classProbabilities.argMax(1);
        })
        return probs;
    }
}

// const numbers = getNumbers();
// const classes = getClassesAsNumber();

// let preds = window.tf.tidy(() => {
//     const oneHotEncodedLabels = window.tf.oneHot(classes, 3);
//     const model = new LogisticRegression(4, 3, 0.1, 0);
//     model.fit(numbers, oneHotEncodedLabels);
//     const predictions = model.predict(numbers);
//     return predictions.arraySync()
// })



// console.log(preds);
// console.log("memory", window.tf.memory().numTensors);

const tf = require('@tensorflow/tfjs-node');
const iris = require('ml-dataset-iris');
class MultiClassLogisticRegression {
    constructor(n_iter = 10000, thres = 1e-3) {
        this.n_iter = n_iter;
        this.thres = thres;
    }

    async fit(X, y, batch_size = 64, lr = 0.001, rand_seed = 4, verbose = false) {
        tf.setBackend('cpu'); // Set the TensorFlow.js backend

        this.classes = Array.from(new Set(y.arraySync())); // Unique classes
        this.class_labels = {};
        this.classes.forEach((c, i) => { this.class_labels[c] = i; });

        X = this.add_bias(X);
        y = this.one_hot(y);
        y.print()
        this.loss = [];
        this.weights = tf.zeros([this.classes.length, X.shape[1]]);

        await this.fit_data(X, y, batch_size, lr, verbose);
        return this;
    }

    async fit_data(X, y, batch_size, lr, rand_seed, l1 = 0, l2 = 0, verbose) {
        let i = 0;
        while (!this.n_iter || i < this.n_iter) {
            const predicted = this.predict_(X);
            const loss = await this.cross_entropy(y, predicted);

            // Compute gradients
            const grads = await tf.tidy(() => {
                const error = y.sub(predicted);
                const gradWeights = error.transpose().matMul(X).div(X.shape[0]);
                const gradL1 = this.weights.sign().mul(l1);
                const gradL2 = this.weights.mul(2 * l2);
                const regularizedGrads = gradWeights.add(gradL1).add(gradL2);
                return gradWeights;
            });

            // Update weights using gradients
            this.weights = this.weights.add(grads.mul(-lr));

            const maxGrad = await tf.max(tf.abs(grads)).data();
            if (maxGrad < this.thres) break;

            if (i % 1000 === 0 && verbose) {
                const accuracy = await this.evaluate_(X, y);
                console.log(`Training Accuracy at ${i} iterations is ${accuracy}`);
            }
            i++;
        }
    }

    predict(X) {
        return this.predict_(this.add_bias(X));
    }

    predict_(X) {
        const pre_vals = X.matMul(this.weights.transpose());
        return this.softmax(pre_vals);
    }

    softmax(z) {
        const expZ = tf.exp(z);
        return expZ.div(expZ.sum(1, true));
    }

    predict_classes(X) {
        this.probs_ = this.predict(X);
        return this.probs_.argMax(1).arraySync().map(c => this.classes[c]);
    }

    add_bias(X) {
        const ones = tf.ones([X.shape[0], 1]);
        return ones.concat(X, 1);
    }

    one_hot(y) {
        return tf.oneHot(y.flatten(), this.classes.length);
    }

    async score(X, y) {
        const accuracy = await this.evaluate_(X, this.one_hot(y));
        return accuracy;
    }

    async evaluate_(X, y) {
        const predictions = this.predict_(X);
        const predictedClasses = predictions.argMax(1);
        const trueClasses = y.argMax(1);
        const accuracy = tf.sum(predictedClasses.equal(trueClasses)).div(X.shape[0]);
        return accuracy.array();
    }

    async cross_entropy(y, probs) {
        const epsilon = 1e-15;
        const clippedProbs = probs.clipByValue(epsilon, 1.0 - epsilon);
        const negLogProb = tf.mul(y, clippedProbs.log());
        const loss = negLogProb.mul(-1).sum().div(y.shape[0]);
        return loss;
    }
}
// Mock data
const X_train = tf.randomNormal([100, 5]); // Training features
const y_train = tf.randomUniform([100], 0, 3, 'int32'); // Training labels

// Create an instance of MultiClassLogisticRegression
const model = new MultiClassLogisticRegression();
const features = tf.tensor2d(iris.getNumbers());
const cls = tf.tensor(iris.getClassesAsNumber(), null, "int32");
y_train.print()
// features.print()
// X_train.print()
// y_train.print()
// // Fit the model with the training data
model.fit(X_train, y_train, 64, 0.001, 4, true).then(() => {
    console.log(model.predict(X_train).dataSync());
})
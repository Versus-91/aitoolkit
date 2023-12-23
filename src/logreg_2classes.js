import { Matrix } from 'ml-matrix';

export default class LogisticRegressionTwoClasses {
    constructor(options = {}) {
        const { numSteps = 50000, learningRate = 5e-4, l1 = 0.1, l2 = null, weights = null } = options;
        this.numSteps = numSteps;
        this.learningRate = learningRate;
        this.l1 = l1;
        this.l2 = l2;
        this.weights = weights ? Matrix.checkMatrix(weights) : null;
    }

    train(features, target) {
        let weights = Matrix.zeros(1, features.columns);

        for (let step = 0; step < this.numSteps; step++) {
            const scores = features.mmul(weights.transpose());
            const predictions = sigmoid(scores);

            // Update weights with gradient
            const error = Matrix.columnVector(predictions)
                .neg()
                .add(target);
            const gradient = features.transpose().mmul(error);
            // Apply L2 regularization
            if (this.l2 !== null && this.l2 !== 0) {
                const l2RegularizationTerm = weights.mul(this.l2);
                gradient.add(l2RegularizationTerm.transpose());
            }

            // Apply L1 regularization
            if (this.l1 !== null && this.l1 !== 0) {
                const l1RegularizationTerm = weights.abs().mul(this.l1);
                gradient.add(l1RegularizationTerm.transpose());
            }
            weights = weights.add(gradient.mul(this.learningRate).transpose());
        }

        this.weights = weights;
    }

    testScores(features) {
        const finalData = features.mmul(this.weights.transpose());
        return sigmoid(finalData);
    }

    predict(features) {
        const finalData = features.mmul(this.weights.transpose());
        return sigmoid(finalData).map(Math.round);
    }

    static load(model) {
        return new LogisticRegressionTwoClasses(model);
    }

    toJSON() {
        return {
            numSteps: this.numSteps,
            learningRate: this.learningRate,
            weights: this.weights,
        };
    }
}

function sigmoid(scores) {
    scores = scores.to1DArray();
    let result = [];
    for (let i = 0; i < scores.length; i++) {
        result.push(1 / (1 + Math.exp(-scores[i])));
    }
    return result;
}
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

    async fit(features, labels, x_test, y_test, epochs = 1000) {
        window.x_train = features
        window.y_train = labels
        window.y_test = y_test
        window.x_test = x_test
        const res = await window.pyodide.runPythonAsync(`
        import js
        from sklearn.metrics import f1_score
        from sklearn.metrics import recall_score,precision_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        def logspace(start, stop, num):
            result = []
            for i in range(num):
                value = start * ((stop / start) ** (i / (num - 1)))
                result.append(value)
            return result
        x_train = js.x_train.to_py()
        y_train = js.y_train.to_py()
        x_test = js.x_test.to_py()
        y_test = js.y_test.to_py()

        model = LogisticRegression(penalty=None,max_iter= 1000)
        model.fit(x_train,y_train)
        preds = model.predict(x_test)
        probs = model.predict_proba(x_test)

        accuracy = accuracy_score(y_test, preds)
        # Create different values for alpha
        alphas = logspace(1e-4, 1e2, 100)

        # Fit logistic regression models for different alpha values
        coefs = []
        for a in alphas:
            clf = LogisticRegression(penalty='l1', C=1/a, solver='liblinear', random_state=42,max_iter= 1000)
            clf.fit(x_train, y_train)
            coefs.append(clf.coef_[0])
        print(accuracy)
        preds,probs,coefs,alphas
      `);
        const result = res.toJs()
        const final_result = {
            preds: Array.from(result[0]),
            probs: result[1].map(m => Array.from(m)),
            coefs: result[2].map(m => Array.from(m)),
            alphas: result[3],
        }
        return final_result




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

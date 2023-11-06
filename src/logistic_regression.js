import * as tfvis from '@tensorflow/tfjs-vis';
import { binarize } from './utils'

export default class LogisticRegression {
    constructor(chart_controller) {
        this.model = null
        this.chart_controller = chart_controller
    }
    async train(x_train, y_train, featureCount, num_classes, epochs = 200, batch_size = 32) {
        const loss = num_classes == 2 ? 'binaryCrossentropy' : 'categoricalCrossentropy';
        const activation = num_classes == 2 ? 'sigmoid' : 'softmax';
        const model = tf.sequential();
        model.add(tf.layers.dense({
            units: num_classes == 2 ? 1 : num_classes,
            inputShape: [featureCount],
            activation: activation
        }));
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: loss,
            metrics: ['accuracy']
        });

        // Train the model
        await model.fit(x_train, y_train, {
            epochs: epochs,
            batchSize: batch_size,
            callbacks: tf.callbacks.earlyStopping({ monitor: 'loss', patience: 40 }),
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log('Accuracy', logs.acc);
                }
            }
        });
        this.model = model
    };
    async evaluate(x_test, y_test, lables = [], is_binary = false) {
        const resultsDiv = document.getElementById('results');
        // const evaluation = await model.evaluate(X, y);
        const predictions = this.model.predict(x_test);
        const predictedLabels = window.tf.tidy(() => {
            return is_binary ? binarize(predictions).as1D() : predictions.argMax(1);
        })

        // const trueLabels = y_test.arraySync();
        // const pre = tf.metrics.precision(trueLabels, predictedLabels)
        // const re = tf.metrics.recall(trueLabels, predictedLabels)
        // const confusionMatrix = tf.math.confusionMatrix(
        //     trueLabels,
        //     predictedLabels,
        //     3
        // );
        //roc for 1st class        
        let y = window.tf.tidy(() => {
            const modifiedTensor = is_binary ? [] : tf.where(tf.equal(y_test.argMax(1), 0), 1, 0);
            let [area, fprs, tprs] = is_binary ? this.chart_controller.drawROC(y_test, predictions)
                : this.chart_controller.drawROC(modifiedTensor, predictions.slice([0, 0], [-1, 1]))
            this.chart_controller.roc_chart("roc", tprs, fprs)
            return is_binary ? y_test : y_test.argMax(1)

        })
        const confusionMatrix = await tfvis.metrics.confusionMatrix(y, predictedLabels);
        const container = document.getElementById("confusion-matrix");
        tfvis.render.confusionMatrix(container, {
            values: confusionMatrix,
            // tickLabels: lables
        });
        let result = {
            predictions: predictions.arraySync()
        }
        window.tf.dispose(predictions)
        window.tf.dispose(y)
        window.tf.dispose(this.model)
        return result
    }
}

import * as tfvis from '@tensorflow/tfjs-vis';

export default class Trainer {
    constructor() {

    }

    async trainModel(xs, ys, numClasses) {
        const model = tf.sequential();

        model.add(tf.layers.dense({
            units: numClasses,
            activation: 'softmax',
            inputShape: [4],
        }));

        model.compile({
            optimizer: tf.train.adam(),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy'],
        });

        const trainingConfig = {
            epochs: 100,
            shuffle: true,
            validationSplit: 0.1,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    console.log(`Epoch ${epoch}, loss: ${logs.loss}, accuracy: ${logs.acc}`);
                },
            },
        };

        const history = await model.fit(xs, ys, trainingConfig);
        return model;
    }
    async train() {
        let model = tf.sequential();

    }
    async trainLogisticRegression(x_train, y_train, featureCount, num_classes) {
        const model = tf.sequential();
        model.add(tf.layers.dense({
            units: num_classes,
            inputShape: [featureCount],
            activation: 'softmax'
        }));
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy', // Use sparse categorical cross-entropy
            metrics: ['accuracy']
        });

        // Train the model
        await model.fit(X, y, {
            epochs: 300,
            batchSize: 32,
            callbacks: tf.callbacks.earlyStopping({ monitor: 'loss', patience: 40 }),
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    // console.log('Accuracy', logs.acc);
                }
            }
        });
        return model;
    };
    async evaluate_classification() {
        const resultsDiv = document.getElementById('results');
        // const evaluation = await model.evaluate(X, y);
        const predictions = model.predict(x_test);
        const predictedLabels = predictions.argMax(1);

        // const trueLabels = y_test.arraySync();
        // const pre = tf.metrics.precision(trueLabels, predictedLabels)
        // const re = tf.metrics.recall(trueLabels, predictedLabels)
        // const confusionMatrix = tf.math.confusionMatrix(
        //     trueLabels,
        //     predictedLabels,
        //     3
        // );
        console.log(y_test.arraySync());
        const modifiedTensor = tf.where(tf.equal(y_test, 0), 1, 0);
        let [area, fprs, tprs] = chart.drawROC(modifiedTensor, predictions.slice([0, 0], [-1, 1]))

        chart.roc_chart("roc", tprs, fprs)
        const confusionMatrix = await tfvis.metrics.confusionMatrix(y_test, predictedLabels);
        const container = document.getElementById("confusion-matrix");

        tfvis.render.confusionMatrix(container, {
            values: confusionMatrix,
            tickLabels: encoder.inverseTransform([0, 1, 2])
        });
    }
    async evaluate_regression() {
        const resultsDiv = document.getElementById('results');
        // const evaluation = await model.evaluate(X, y);
        const predictions = model.predict(x_test);
        const predictedLabels = predictions.argMax(1);

        // const trueLabels = y_test.arraySync();
        // const pre = tf.metrics.precision(trueLabels, predictedLabels)
        // const re = tf.metrics.recall(trueLabels, predictedLabels)
        // const confusionMatrix = tf.math.confusionMatrix(
        //     trueLabels,
        //     predictedLabels,
        //     3
        // );
        console.log(y_test.arraySync());
        const modifiedTensor = tf.where(tf.equal(y_test, 0), 1, 0);
        let [area, fprs, tprs] = chart.drawROC(modifiedTensor, predictions.slice([0, 0], [-1, 1]))

        chart.roc_chart("roc", tprs, fprs)
        const confusionMatrix = await tfvis.metrics.confusionMatrix(y_test, predictedLabels);
        const container = document.getElementById("confusion-matrix");

        tfvis.render.confusionMatrix(container, {
            values: confusionMatrix,
            tickLabels: encoder.inverseTransform([0, 1, 2])
        });
    }
    async train_linear_regression(featureCount, x_train, y_train) {
        const model = tf.sequential();

        model.add(
            tf.layers.dense({
                inputShape: [featureCount],
                units: 1,
            })
        );
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: "meanSquaredError",
            metrics: [tf.metrics.meanAbsoluteError]
        });
        const trainLogs = [];
        const lossContainer = document.getElementById("loss-cont");
        const accContainer = document.getElementById("acc-cont");
        console.log("Training...");
        await model.fit(x_train, y_train, {
            batchSize: 64,
            epochs: 3000,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    console.log(logs);
                    // trainLogs.push({
                    //     rmse: Math.sqrt(logs.loss),
                    //     val_rmse: Math.sqrt(logs.val_loss),
                    //     mae: logs.meanAbsoluteError,
                    //     val_mae: logs.val_meanAbsoluteError,
                    // })
                    // tfvis.show.history(lossContainer, trainLogs, ["rmse", "val_rmse"])
                    // tfvis.show.history(accContainer, trainLogs, ["mae", "val_mae"])
                }
            }
        });
        return model;
    };
}
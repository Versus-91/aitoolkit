
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
    async trainLogisticRegression(featureCount, trainDs, validDs) {
        const model = tf.sequential();
        function logit(x) {
            return tf.div(tf.scalar(1), tf.add(tf.scalar(1), tf.exp(tf.neg(x))));
        }
        model.add(
            tf.layers.dense({
                units: 2,
                activation: "softmax",
                inputShape: [featureCount]
            })
        );
        const optimizer = tf.train.adam(0.001);
        model.compile({
            optimizer: "adam",
            loss: "binaryCrossentropy",
            metrics: ["accuracy"]
        });
        const trainLogs = [];
        const lossContainer = document.getElementById("loss-cont");
        const accContainer = document.getElementById("acc-cont");
        console.log("Training...");
        await model.fitDataset(trainDs, {
            epochs: 50,
            validationData: validDs,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    trainLogs.push(logs);
                    tfvis.show.history(lossContainer, trainLogs, ["loss", "val_loss"]);
                    tfvis.show.history(accContainer, trainLogs, ["acc", "val_acc"]);
                }
            }
        });
        return model;
    };
    async train_linear_regression(featureCount, x_train, y_train) {
        const model = tf.sequential();

        model.add(
            tf.layers.dense({
                inputShape: [featureCount],
                units: 1,
            })
        );
        model.compile({
            optimizer: tf.train.sgd(0.01),
            loss: "meanAbsoluteError",
            metrics: [tf.metrics.meanAbsoluteError]
        });
        const trainLogs = [];
        const lossContainer = document.getElementById("loss-cont");
        const accContainer = document.getElementById("acc-cont");
        console.log("Training...");
        await model.fit(x_train, y_train, {
            batchSize: 64,
            epochs: 200,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    console.log(logs.loss);
                    trainLogs.push({
                        rmse: Math.sqrt(logs.loss),
                        val_rmse: Math.sqrt(logs.val_loss),
                        mae: logs.meanAbsoluteError,
                        val_mae: logs.val_meanAbsoluteError,
                    })
                    tfvis.show.history(lossContainer, trainLogs, ["rmse", "val_rmse"])
                    tfvis.show.history(accContainer, trainLogs, ["mae", "val_mae"])
                }
            }
        });
        return model;
    };
}
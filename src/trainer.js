
import * as tfvis from '@tensorflow/tfjs-vis';
import KNN from 'ml-knn';
import {
    getNumbers, getDataset, getClasses
} from 'ml-dataset-iris';
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
    knn_test() {
        var train_dataset = getNumbers();
        var train_labels = getClasses();
        var knn = new KNN(train_dataset, train_labels, { k: 2 });
        console.log(knn);
        var test_dataset = train_dataset;

        var ans = knn.predict(test_dataset);

        console.log(ans);

    }

}
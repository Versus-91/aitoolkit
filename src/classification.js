import * as tfvis from '@tensorflow/tfjs-vis';

export default class Classification {
    constructor(chart_controller) {
        this.chart_controller = chart_controller
    }
    async trainLogisticRegression(x_train, y_train, featureCount, num_classes, epochs = 500, batch_size = 32) {
        const model = tf.sequential();
        model.add(tf.layers.dense({
            units: num_classes,
            inputShape: [featureCount],
            activation: 'softmax'
        }));
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
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
        return model;
    };
    async evaluate(x_test, y_test, model, lables = []) {
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
        //roc for 1st class        
        const modifiedTensor = tf.where(tf.equal(y_test.argMax(1), 0), 1, 0);
        let [area, fprs, tprs] = this.chart_controller.drawROC(modifiedTensor, predictions.slice([0, 0], [-1, 1]))
        this.chart_controller.roc_chart("roc", tprs, fprs)
        const confusionMatrix = await tfvis.metrics.confusionMatrix(y_test.argMax(1), predictedLabels);
        const container = document.getElementById("confusion-matrix");
        tfvis.render.confusionMatrix(container, {
            values: confusionMatrix,
            // tickLabels: lables
        });
    }
}
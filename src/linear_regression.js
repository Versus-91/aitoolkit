export default class LinearRegression {

    async train(featureCount, x_train, y_train) {
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
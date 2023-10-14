export default class Trainer {
    constructor() {

    }
    async train() {
        let model = tf.sequential();

    }
    async trainLogisticRegression(featureCount, trainDs, validDs) {
        const model = tf.sequential();
        model.add(
            tf.layers.dense({
                units: 2,
                activation: "softmax",
                inputShape: [featureCount]
            })
        );
        const optimizer = tf.train.adam(0.001);
        model.compile({
            optimizer: optimizer,
            loss: "binaryCrossentropy",
            metrics: ["accuracy"]
        });
        const trainLogs = [];
        const lossContainer = document.getElementById("loss-cont");
        const accContainer = document.getElementById("acc-cont");
        console.log("Training...");
        await model.fitDataset(trainDs, {
            epochs: 100,
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
}
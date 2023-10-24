"use strict";
import * as tf from '@tensorflow/tfjs';
import { DataFrame, LabelEncoder,Series } from 'danfojs/dist/danfojs-base';
import $ from 'jquery';
import Papa from 'papaparse';
import ChartController from "./src/charts.js";
import DataLoader from "./src/data.js";
import Trainter from "./src/trainer.js";
import UI from "./src/ui.js";
window.jQuery = window.$ = $
let data_parser = new DataLoader();

let ui = new UI(data_parser);
let trainer = new Trainter();
let chart = new ChartController(data_parser)

function handleFileSelect(evt) {
    var target = evt.target || evt.srcElement;
    if (target.value.length == 0) {
        return;
    }
    var file = evt.target.files[0];
    Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        dynamicTyping: true,
        complete: async function (results) {
            ui.createDatasetPropsDropdown(results.data);
            ui.renderDatasetStats(results.data);
            const features = ["Glucose"];
            const [trainDs, validDs, xTest, yTest] = data_parser.createDataSets(
                results.data,
                features,
                0.1,
                16
            );
            await test(results.data);
            console.log(data_parser.findMissinValues(results.data));
            const portions = data_parser.findTargetPercents(results.data, "Species");
            ui.drawTargetPieChart(portions, Object.keys(portions).filter(m => m !== "count"), "y_pie_chart");
        }
    });
}

async function test(data) {
    const df = new DataFrame(data)
    let encoder = new LabelEncoder()
    let X = df.loc({ columns: df.columns.slice(1, -1) }).tensor
    encoder.fit(df["Species"].values);
    let lables = new Series(encoder.transform(df["Species"].values))
    const y = tf.oneHot(lables.tensor, 3);

    const resultsDiv = document.getElementById('results');
    // Create a simple logistic regression model
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 3, activation: 'softmax', inputShape: [4] }));

    // Compile the model
    model.compile({
        optimizer: 'sgd',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    // Train the model
    await model.fit(X, y, {
        epochs: 200,
        callbacks: tf.callbacks.earlyStopping({ monitor: 'loss', patience: 40 }),
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log('Accuracy', logs.acc);
            }
        }
    });

    const evaluation = await model.evaluate(X, y);
    const predictions = model.predict(X);
    predictions.print();
    const predictedLabels = predictions.argMax(1);
    predictedLabels.print();
    const trueLabels = y.argMax(1);
    trueLabels.print()

    // Compute the confusion matrix
    // const confusionMatrix = tf.math.confusionMatrix(
    //     trueLabels,
    //     predictedLabels,
    //     3
    // );
    // confusionMatrix.print()
    // Display the confusion matrix
    // const confusionMatrixContainer = document.getElementById('confusion-matrix');
    // tfvis.render.confusionMatrix(
    //     confusionMatrixContainer,
    //     { values: confusionMatrix }
    // );

    resultsDiv.innerHTML = `
                <h2>Logistic Regression Results:</h2>
                <p>Accuracy: ${await evaluation[1].dataSync()}</p>`
        ;
}

// evaluateModel();
document.getElementById("parseCVS").addEventListener("change", handleFileSelect)

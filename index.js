"use strict";
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { DataFrame, LabelEncoder, Series } from 'danfojs/dist/danfojs-base';
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
let cvs_data = ""
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
            cvs_data = data_parser.splitData(results.data)
            ui.createDatasetPropsDropdown(results.data);
            document.getElementById("train-button").onclick = async () => {
                await train(cvs_data)
            }
            ui.renderDatasetStats(results.data);
            // console.log(data_parser.findMissinValues(results.data));
            const portions = data_parser.findTargetPercents(results.data, "Species");
            ui.drawTargetPieChart(portions, Object.keys(portions).filter(m => m !== "count"), "y_pie_chart");
        }
    });
}

async function train(data) {
    const df = new DataFrame(data.training_data)
    let encoder = new LabelEncoder()
    encoder.fit(df['Species'])
    let sf_enc = encoder.transform(df['Species'].values)
    df.drop({ columns: ["Species"], inplace: true });
    df.addColumn("y", sf_enc, { inplace: true });

    const df_test = new DataFrame(data.test_data)
    encoder.fit(df_test['Species'])
    let sf_enc_test = encoder.transform(df_test['Species'].values)
    df_test.drop({ columns: ["Species"], inplace: true });
    df_test.addColumn("y", sf_enc_test, { inplace: true });
    // let encoded_lables = encoder.transform(lables.values)
    // console.log(encoded_lables.tensor);
    let X = df.loc({ columns: df.columns.slice(1, -1) }).tensor
    let y = df.column("y").tensor;
    y = y.toFloat()
    let x_test = df_test.loc({ columns: df.columns.slice(1, -1) }).tensor
    let y_test = df_test.column("y").tensor;
    y_test = y_test.toFloat()

    const resultsDiv = document.getElementById('results');
    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 3, // Three output units for the three classes (0, 1, 2)
        inputShape: [4], // Input shape for your four features
        activation: 'softmax' // Softmax activation for multiclass classification
    }));
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'sparseCategoricalCrossentropy', // Use sparse categorical cross-entropy
        metrics: ['accuracy']
    });

    // Train the model
    await model.fit(X, y, {
        batchSize: 3,
        epochs: 30,
        callbacks: tf.callbacks.earlyStopping({ monitor: 'loss', patience: 40 }),
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log('Accuracy', logs.acc);
            }
        }
    });

    // const evaluation = await model.evaluate(X, y);
    const predictions = model.predict(x_test);
    const predictedLabels = predictions.argMax(1);
    const trueLabels = y_test.arraySync();
    const pre = tf.metrics.precision(trueLabels, predictedLabels)
    const re = tf.metrics.recall(trueLabels, predictedLabels)
    pre.print()
    re.print();
    // const confusionMatrix = tf.math.confusionMatrix(
    //     trueLabels,
    //     predictedLabels,
    //     3
    // );
    const confusionMatrix = await tfvis.metrics.confusionMatrix(y_test, predictedLabels);
    const container = document.getElementById("confusion-matrix");

    tfvis.render.confusionMatrix(container, {
        values: confusionMatrix,
    });
    // resultsDiv.innerHTML = `
    //             <h2>Logistic Regression Results:</h2>
    //             <p>Accuracy: ${await evaluation[1].dataSync()}</p>`
    //     ;
}

// evaluateModel();
document.getElementById("parseCVS").addEventListener("change", handleFileSelect)
document.getElementById("pca-button").addEventListener("click", chart.draw_pca)



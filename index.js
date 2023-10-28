"use strict";
import * as ss from 'simple-statistics'
import { DataFrame, LabelEncoder, Series, tensorflow } from 'danfojs/dist/danfojs-base';
import * as tfvis from '@tensorflow/tfjs-vis';
import $ from 'jquery';
import Papa from 'papaparse';
import ChartController from "./src/charts.js";
import DataLoader from "./src/data.js";
import Trainter from "./src/trainer.js";
import UI from "./src/ui.js";
window.tf = tensorflow
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
            const df1 = new DataFrame(results.data);
            let new_col = ["count", "mean", "std", "min", "median", "max", "variance"]
            let description = df1.describe()
            description.addColumn("D", new_col, { inplace: true })
            description.plot("desc").table()
            console.log(df1.shape)
            const na = df1.isNa().sum({ axis: 0 }).div(df1.isNa().count({ axis: 0 })).round(4)
            na.plot("plot_div").table()
            // let data = [[1, 2, 3], [NaN, 5, 6], [NaN, 30, 40], [39, undefined, 78]]
            // let cols = ["A", "B", "C"]
            // let df = new DataFrame(data, { columns: cols })
            // Calculate KDE values at specific points
            // df.isNa().sum().print()
            // let missing_values = df1.isNa().sum().div(df1.isNa().count()).round(4)
            // missing_values.print()
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
    y = tf.oneHot(tf.tensor1d(sf_enc).toInt(), 3);
    console.log(y);
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
                console.log('Accuracy', logs.acc);
            }
        }
    });

    // const evaluation = await model.evaluate(X, y);
    const predictions = model.predict(x_test);
    const predictedLabels = predictions.argMax(1);

    const trueLabels = y_test.arraySync();
    // const pre = tf.metrics.precision(trueLabels, predictedLabels)
    // const re = tf.metrics.recall(trueLabels, predictedLabels)
    // const confusionMatrix = tf.math.confusionMatrix(
    //     trueLabels,
    //     predictedLabels,
    //     3
    // );
    console.log(predictions.slice([0, 0], [-1, 1]).arraySync())

    let [area, fprs, tprs] = chart.drawROC(y_test, predictions.slice([0, 0], [-1, 1]))
    const newSeries = [];
    for (let i = 0; i < fprs.length; i++) {
        newSeries.push({
            x: fprs[i],
            y: tprs[i],
        });
    }
    const rocValues = [];
    const rocSeries = [];
    rocSeries.push("seriesName");
    rocValues.push(newSeries);
    tfvis.render.linechart(
        document.getElementById('roc'),
        { values: rocValues, series: rocSeries },
        {
            width: 450,
            height: 320,
        },
    );
    console.log("fprs", fprs);
    console.log("tprs", tprs);
    chart.roc_chart("roc2", tprs, fprs)
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




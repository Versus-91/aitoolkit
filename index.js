"use strict";
import { DataFrame, tensorflow, OneHotEncoder, LabelEncoder } from 'danfojs/dist/danfojs-base';
import $ from 'jquery';
import Papa from 'papaparse';
import ChartController from "./src/charts.js";
import DataLoader from "./src/data.js";
import Trainter from "./src/trainer.js";
import UI from "./src/ui.js";
import { FeatureCategories, Settings } from './feature_types.js';
import { ModelFactory } from './src/model_factory.js';
import * as tfvis from '@tensorflow/tfjs-vis';
import * as d3 from "d3";
import DataTable from 'datatables.net-dt';
import * as sk from 'scikitjs'

sk.setBackend(tensorflow)

window.tf = tensorflow
window.jQuery = window.$ = $
let data_parser = new DataLoader();
let ui = new UI(data_parser);
let trainer = new Trainter();
let chart_controller = new ChartController(data_parser);


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
        complete: function (result) {
            let dataset = new DataFrame(result.data)
            ui.createDatasetPropsDropdown(dataset);
            document.getElementById("train-button").onclick = async () => {
                document.getElementById("train-button").classList.add("is-loading")
                await visualize(dataset)
                document.getElementById("train-button").classList.remove("is-loading")
            }

        }
    });
}
async function visualize(dataset) {
    ui.renderDatasetStats(dataset);

    let numericColumns = []
    dataset.columns.forEach(column => {
        if (dataset.column(column).dtype !== 'string' && column !== "Id") {
            numericColumns.push(column)
        }
    });
    const target = document.getElementById("target").value;
    let is_classification = document.getElementById(target).value !== FeatureCategories.Numerical;
    if (numericColumns.length > 0) {
        chart_controller.plot_tsne(dataset.loc({ columns: numericColumns }).values, is_classification ? dataset.loc({ columns: [target] }).values : []);
        chart_controller.draw_pca(dataset.loc({ columns: numericColumns }).values, is_classification ? dataset.loc({ columns: [target] }).values : []);

        chart_controller.draw_kde(dataset, numericColumns)
    }
    if (is_classification) {
        let counts = dataset.column(target).valueCounts()
        const df = new DataFrame({
            values: counts.values,
            labels: counts.$index,
        });

        df.plot("y_pie_chart").pie({ config: { values: "values", labels: "labels" } });
    }
    let table_columns = []
    dataset.columns.forEach(element => {
        table_columns.push({ title: element })
    });

    await train(dataset)

}
async function train(data) {
    let dataset = data.copy()

    const target = document.getElementById("target").value;
    dataset = data_parser.handle_missing_values(dataset)
    let selected_columns = ui.find_selected_columns(dataset.columns)
    let model_factory = new ModelFactory()
    selected_columns = selected_columns.filter(m => m !== target)
    const x_train = dataset.loc({ columns: selected_columns })
    const y_train = dataset.column(target)
    const x_test = x_train
    const y_test = y_train
    if (document.getElementById(target).value !== FeatureCategories.Numerical) {
        let model_name = document.getElementById('model_name').value
        switch (model_name) {
            case Settings.classification.k_nearest_neighbour.lable:
                let knn_classifier = model_factory.createModel(Settings.classification.k_nearest_neighbour)
                knn_classifier.train(x_train.values, dataset.column(target).values, 5)
                let y_preds = knn_classifier.evaluate(x_train.values)
                let evaluation_result = evaluate_classification(y_preds, y_train.values)
                let encoder = new LabelEncoder()
                encoder.fit(y_train)
                let encoded_ys = encoder.transform(y_train.values)
                let encoded_yhats = encoder.transform(y_preds)

                let numericColumns = []
                x_train.columns.forEach(column => {
                    if (x_train.column(column).dtype !== 'string' && column !== "Id") {
                        numericColumns.push(column)
                    }
                });
                let table_columns = []
                x_train.addColumn("y", dataset.column(target), { inplace: true })
                x_train.addColumn("predictions", y_preds, { inplace: true })

                x_train.columns.forEach(element => {
                    table_columns.push({ title: element })
                });

                new DataTable('#predictions_table', {
                    responsive: true,
                    columns: table_columns,
                    data: x_train.values
                });

                chart_controller.draw_classification_pca(x_train.loc({ columns: numericColumns }).values, y_train.values, evaluation_result.indexes)
                plot_confusion_matrix(window.tf.tensor(encoded_yhats), window.tf.tensor(encoded_ys))
                break;
            case Settings.classification.logistic_regression.lable:
                const unique_classes = [...new Set(dataset.column(target).values)]
                const is_binary_classification = unique_classes.length === 2 ? 1 : 0;
                if (is_binary_classification) {
                    let binary_logistic_regression = model_factory.createModel(Settings.classification.logistic_regression)
                    model = await binary_logistic_regression.train(x_train.tensor, y_train.tensor, selected_columns.length, 2)
                    await binary_logistic_regression.evaluate(x_train.tensor, y_train.tensor, model, [], true)
                } else {
                    let logistic_regression = model_factory.createModel(Settings.classification.logistic_regression, chart_controller)
                    let encoder = new OneHotEncoder()
                    encoder.fit(y_train.values)
                    encoder.transform(y_train.values)
                    let lr = new sk.LogisticRegression({ fitIntercept: false })
                    await lr.fit(x_train.values, encoder.transform(y_train.values))
                    console.log(lr.coef)


                    await logistic_regression.train(x_train_tensor, y_train_tensor, selected_columns.length, unique_classes.length)
                    let result = await logistic_regression.evaluate(x_train_tensor, y_train_tensor)

                    let table_columns = []
                    x_train.addColumn("y", dataset.column(target), { inplace: true })
                    x_train.addColumn("predictions", result.predictions, { inplace: true })


                    x_train.columns.forEach(element => {
                        table_columns.push({ title: element })
                    });
                    new DataTable('#predictions_table', {
                        responsive: true,
                        columns: table_columns,
                        data: x_train.values
                    });

                    x_train_tensor.dispose()
                    x_train_tensor.dispose()
                    console.log(window.tf.memory().numTensors);
                }
            case Settings.classification.random_forest.lable:
                const model = model_factory.createModel(Settings.classification.random_forest, null, {
                    seed: 3,
                    maxFeatures: 0.8,
                    replacement: true,
                    nEstimators: 25
                });
                let encoder_rf = new LabelEncoder()
                encoder_rf.fit(y_train.values)
                encoder_rf.transform(y_train.values)
                let encoded_y = encoder_rf.transform(y_train.values)

                model.train(x_train.values, encoded_y)
                let preds = model.predict(x_train.values)

                console.log(encoder_rf.classes);

                plot_confusion_matrix(window.tf.tensor(preds), window.tf.tensor(encoded_y), encoder_rf.classes)
                console.log(preds);
                break;
            default:
                break;
        }
    } else {
        model = await trainer.train_linear_regression(selected_columns.length, dataset.loc({ columns: selected_columns }).tensor, dataset.column(target).tensor)
    }
    // else {
    //     model = await trainer.train_linear_regression(selected_columns.length, dataset.loc({ columns: selected_columns }).tensor, dataset.column(target).tensor)
    // }
    // let y = df.loc({ columns: [target] }).tensor
    // let preds = model.predict(y)
    // const squaredErrors = tf.square(preds.sub(y))
    // const mse = squaredErrors.mean().dataSync()[0]
    // console.log("mse ", mse);
    // const yMean = y.mean().dataSync()[0];
    // const totalVariation = tf.sum(tf.square(y.sub(yMean)));
    // console.log("variance", totalVariation.dataSync()[0]);
    // const unexplainedVariation = tf.sum(squaredErrors);
    // const r2 = 1 - unexplainedVariation.dataSync()[0] / totalVariation.dataSync()[0];
    // console.log("r2", r2);
    // let encoder = new LabelEncoder()
    // encoder.fit(df['Species'])
    // let sf_enc = encoder.transform(df['Species'].values)
    // df.drop({ columns: ["Species"], inplace: true });
    // df.addColumn("y", sf_enc, { inplace: true });
    // const df_test = new DataFrame(data.test_data)
    // encoder.fit(df_test['Species'])
    // let sf_enc_test = encoder.transform(df_test['Species'].values)
    // df_test.drop({ columns: ["Species"], inplace: true });
    // df_test.addColumn("y", sf_enc_test, { inplace: true });
    // // let encoded_lables = encoder.transform(lables.values)
    // // console.log(encoded_lables.tensor);
    // let X = df.loc({ columns: df.columns.slice(1, -1) }).tensor
    // let y = df.column("y").tensor;
    // y = tf.oneHot(tf.tensor1d(sf_enc).toInt(), 3);
    // console.log(y);
    // let x_test = df_test.loc({ columns: df.columns.slice(1, -1) }).tensor
    // let y_test = df_test.column("y").tensor;
    // y_test = y_test.toFloat()
}
function evaluate_classification(y_preds, y_test) {
    console.assert(y_preds.length === y_test.length, "preds and test should have the same length.")
    let missclassification_indexes = []
    let currect_classifications_sum = 0
    y_test.forEach((element, i) => {
        if (element === y_preds[i]) {
            currect_classifications_sum++
        } else {
            missclassification_indexes.push(i)

        }
    });
    return {
        accuracy: (currect_classifications_sum / y_preds.length) * 100,
        indexes: missclassification_indexes
    }
}
async function plot_confusion_matrix(y, predictedLabels, lables) {
    const confusionMatrix = await tfvis.metrics.confusionMatrix(y, predictedLabels);
    const container = document.getElementById("confusion-matrix");
    tfvis.render.confusionMatrix(container, {
        values: confusionMatrix,
        tickLabels: lables ?? null
    });
    window.tf.dispose(y)
    window.tf.dispose(predictedLabels)
    window.tf.dispose(confusionMatrix)
}
function test() {
    chart_controller.draw_kde(null, null)
}
document.getElementById("parseCVS").addEventListener("change", handleFileSelect)
document.getElementById("test_rf").addEventListener("click", test)









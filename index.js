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
import { LogisticRegression } from './src/LogisticRegression.js';
import { Matrix } from 'ml-matrix';
import Bulma from '@vizuaalog/bulmajs';
import { calculateMetrics } from './src/utils.js';
import SVM from "libsvm-js/asm";

sk.setBackend(tensorflow)

window.tf = tensorflow
window.jQuery = window.$ = $
let data_parser = new DataLoader();
let ui = new UI(data_parser);
let trainer = new Trainter();
let chart_controller = new ChartController(data_parser);
let X
let y

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
                await train(dataset, result.data.length)
                document.getElementById("train-button").classList.remove("is-loading")
            }

            document.getElementById("visualize").onclick = async () => {
                console.log("sss");
                document.getElementById("visualize").classList.add("is-loading")
                await visualize(dataset, result.data.length)
                document.getElementById("visualize").classList.remove("is-loading")
            }

        }
    });
}
async function visualize(dataset, len) {
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
        document.getElementById("container").innerHTML = "";
        numericColumns.forEach(col => {
            console.log(col);
            chart_controller.draw_kde(dataset, col)
        });

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

    // await train(dataset, len)

}

async function train(data, len) {
    const limit = Math.ceil(len * 70 / 100)
    let dataset = data.copy()
    const target = document.getElementById("target").value;
    dataset = data_parser.handle_missing_values(dataset)
    let selected_columns = ui.find_selected_columns(dataset.columns)
    let model_factory = new ModelFactory()
    selected_columns = selected_columns.filter(m => m !== target)

    const features = dataset.loc({ columns: selected_columns })
    const targets = dataset.column(target)

    const train_bound = `0:${limit}`
    const test_bound = `${limit}:${len}`

    const x_train = features.iloc({ rows: [`0: ${limit}`] })
    const y_train = targets.iloc([train_bound])
    const x_test = features.iloc({ rows: [`${limit}: ${len}`] });
    const y_test = targets.iloc([test_bound]);

    let numericColumns = []
    features.columns.forEach(column => {
        if (x_train.column(column).dtype !== 'string' && column !== "Id") {
            numericColumns.push(column)
        }
    });
    if (document.getElementById(target).value !== FeatureCategories.Numerical) {
        let model_name = document.getElementById('model_name').value
        switch (model_name) {
            case Settings.classification.k_nearest_neighbour.lable: {
                let knn_classifier = model_factory.createModel(Settings.classification.k_nearest_neighbour)
                let results = []
                let encoder = new LabelEncoder()
                encoder.fit(targets)
                let encoded_y_train = encoder.transform(y_train.values)
                let encoded_y_test = encoder.transform(y_test.values)
                console.log(window.tf.memory().numTensors);
                for (let k = 3; k < 9; k++) {
                    await knn_classifier.train(x_train.values, encoded_y_train, k)
                    let y_preds = knn_classifier.predict(x_test.values)
                    let evaluation_result = evaluate_classification(y_preds, encoded_y_test)
                    results.push({ k: k, predictions: y_preds, evaluation: evaluation_result })
                }
                console.log(window.tf.memory().numTensors);

                let best_result = results[0];
                results.forEach(element => {
                    if (element.evaluation.accuracy > best_result.accuracy) {
                        best_result = element
                    }
                });

                let encoded_yhats = best_result.predictions
                //results table
                let table_columns = []
                let tbl = x_test.copy()

                tbl.addColumn("y", y_test, { inplace: true })
                tbl.addColumn("predictions", encoder.inverseTransform(best_result.predictions), { inplace: true })

                tbl.columns.forEach(element => {
                    table_columns.push({ title: element })
                });


                new DataTable('#predictions_table', {
                    responsive: true,
                    columns: table_columns,
                    data: tbl.values,
                    rowCallback: function (row, data, index) {
                        var column1Value = data[table_columns.length - 1];
                        var column2Value = data[table_columns.length - 2];
                        if (column1Value !== column2Value) {
                            $(row).css('background-color', '#97233F');
                            $(row).css('color', 'white');
                        }
                    }
                });

                let knn_table_column_names = []
                knn_table_column_names.push({ title: "k" })
                knn_table_column_names.push({ title: "accuracy" })
                let knn_accuracies = results.map(m => [m.k, m.evaluation.accuracy.toFixed(2)])
                console.log(knn_accuracies, knn_table_column_names);
                new DataTable('#knn_table', {
                    responsive: true,
                    columns: knn_table_column_names,
                    data: knn_accuracies,
                });
                chart_controller.draw_classification_pca(x_test.values, y_test.values, best_result.evaluation.indexes)
                console.log(encoder.$labels);
                plot_confusion_matrix(window.tf.tensor(encoded_yhats), window.tf.tensor(encoded_y_test), encoder.$labels)
                break;
            }
            case Settings.classification.support_vectore_machine.lable: {
                let model = model_factory.createModel(Settings.classification.support_vectore_machine, {
                    kernel: SVM.KERNEL_TYPES.RBF,
                    type: SVM.SVM_TYPES.C_SVC,
                    gamma: 0.25,
                    cost: 1,
                    quiet: true
                })
                let results = []
                let encoder = new LabelEncoder()
                encoder.fit(targets)
                let encoded_y_train = encoder.transform(y_train.values)
                let encoded_y_test = encoder.transform(y_test.values)
                model.train(x_train.values, encoded_y_train)
                let y_preds = model.predict(x_test.values)
                let evaluation_result = evaluate_classification(y_preds, encoded_y_test)
                //results table
                let table_columns = []
                let tbl = x_test.copy()

                tbl.addColumn("y", y_test, { inplace: true })
                tbl.addColumn("predictions", encoder.inverseTransform(y_preds), { inplace: true })

                tbl.columns.forEach(element => {
                    table_columns.push({ title: element })
                });

                new DataTable('#predictions_table', {
                    responsive: true,
                    columns: table_columns,
                    data: tbl.values,
                    rowCallback: function (row, data, index) {
                        var column1Value = data[table_columns.length - 1];
                        var column2Value = data[table_columns.length - 2];
                        if (column1Value !== column2Value) {
                            $(row).css('background-color', '#97233F');
                            $(row).css('color', 'white');
                        }
                    }
                });
                chart_controller.draw_classification_pca(x_test.values, y_test.values, evaluation_result.indexes)
                plot_confusion_matrix(window.tf.tensor(y_preds), window.tf.tensor(encoded_y_test), encoder.inverseTransform(Object.values(encoder.$labels)))
                break;
            }
            case Settings.classification.boosting.lable: {
                let model = model_factory.createModel(Settings.classification.boosting,null, {
                    booster: 'gbtree',
                    objective: 'multi:softmax',
                    max_depth: 5,
                    eta: 0.1,
                    min_child_weight: 1,
                    subsample: 0.5,
                    colsample_bytree: 1,
                    silent: 1,
                    iterations: 200
                })
                let results = []
                let encoder = new LabelEncoder()
                encoder.fit(targets)
                let encoded_y_train = encoder.transform(y_train.values)
                let encoded_y_test = encoder.transform(y_test.values)
                await model.train(x_train.values, encoded_y_train)
                let y_preds = await model.predict(x_test.values)
                let evaluation_result = evaluate_classification(y_preds, encoded_y_test)
                //results table
                let table_columns = []
                let tbl = x_test.copy()

                tbl.addColumn("y", y_test, { inplace: true })
                tbl.addColumn("predictions", encoder.inverseTransform(y_preds), { inplace: true })

                tbl.columns.forEach(element => {
                    table_columns.push({ title: element })
                });

                new DataTable('#predictions_table', {
                    responsive: true,
                    columns: table_columns,
                    data: tbl.values,
                    rowCallback: function (row, data, index) {
                        var column1Value = data[table_columns.length - 1];
                        var column2Value = data[table_columns.length - 2];
                        if (column1Value !== column2Value) {
                            $(row).css('background-color', '#97233F');
                            $(row).css('color', 'white');
                        }
                    }
                });
                chart_controller.draw_classification_pca(x_test.values, y_test.values, evaluation_result.indexes)
                plot_confusion_matrix(window.tf.tensor(y_preds), window.tf.tensor(encoded_y_test), encoder.inverseTransform(Object.values(encoder.$labels)))
                break;
            }
            case Settings.classification.logistic_regression.lable: {
                const unique_classes = [...new Set(dataset.column(target).values)]
                const is_binary_classification = unique_classes.length === 2 ? 1 : 0;
                if (is_binary_classification) {
                    let binary_logistic_regression = model_factory.createModel(Settings.classification.logistic_regression)
                    model = await binary_logistic_regression.train(x_train.tensor, y_train.tensor, selected_columns.length, 2)
                    await binary_logistic_regression.evaluate(x_train.tensor, y_train.tensor, model, [], true)
                } else {
                    let logistic_regression = model_factory.createModel(Settings.classification.logistic_regression, chart_controller)
                    let model = new LogisticRegression()
                    let encoder = new OneHotEncoder()
                    encoder.fit(y_train.values)
                    let y_train_t = encoder.transform(y_train.values)
                    let y_train_tensor = tf.tensor(y_train_t)
                    let x_train_tensor = x_train.tensor

                    X = x_train.values;
                    y = y_train_t;
                    let X = new Matrix(x_train.values)
                    let xx = Matrix.columnVector(y_train_t)
                    const logreg = new LogisticRegression();
                    logreg.train(X, xx);
                    const finalResults = logreg.predict(X);
                    console.log(finalResults);
                    await logistic_regression.train(x_train_tensor, y_train_tensor, selected_columns.length, unique_classes.length)
                    let result = await logistic_regression.evaluate(x_train_tensor, y_train_tensor, encoder.$labels)
                    let table_columns = []
                    x_train.addColumn("y", dataset.column(target), { inplace: true })
                    x_train.addColumn("predictions: " + encoder.$labels, result.predictions, { inplace: true })
                    x_train.columns.forEach(element => {
                        table_columns.push({ title: element })
                    });

                    const lastColumnIndex = table_columns.length - 1;

                    table_columns[lastColumnIndex].render = function (data, type, row) {
                        if (type === 'display') {
                            const maxNumber = Math.max(...data);
                            data = data.map(num => num === maxNumber ? `<b>${num.toFixed(2)}</b>` : num.toFixed(2));
                            return data.join(' ')
                        }
                        return data;
                    };
                    new DataTable('#predictions_table', {
                        responsive: true,
                        columns: table_columns,
                        data: x_train.values,
                        rowCallback: function (row, data, index) {
                            var column1Value = data[table_columns.length - 1];
                            var column2Value = data[table_columns.length - 2];
                            if (column1Value !== column2Value) {
                                $(row).css('background-color', '#97233F');
                                $(row).css('color', 'white');
                            }
                        }
                    });

                    x_train_tensor.dispose()
                    y_train_tensor.dispose()
                    console.log(window.tf.memory().numTensors);
                    break
                }
                break
            }
            case Settings.classification.random_forest.lable: {

                const model = model_factory.createModel(Settings.classification.random_forest, null, {
                    seed: 3,
                    maxFeatures: 2,
                    replacement: true,
                    nEstimators: 50,
                    treeOptions: {
                        maxDepth: 5
                    }
                });
                let encoder_rf = new LabelEncoder()
                encoder_rf.fit(y_train.values)
                encoder_rf.transform(y_train.values)

                let encoded_y = encoder_rf.transform(y_train.values)
                let encoded_y_test = encoder_rf.transform(y_test.values)

                model.train(x_train.values, encoded_y)
                let preds = model.predict(x_test.values)

                const evaluation_result = evaluate_classification(preds, encoded_y_test)
                plot_confusion_matrix(window.tf.tensor(preds), window.tf.tensor(encoded_y_test), encoder_rf.inverseTransform(Object.values(encoder_rf.$labels)))
                chart_controller.draw_classification_pca(x_test.values, encoded_y_test, evaluation_result.indexes)

                let table_columns = []
                x_test.addColumn("y", y_test, { inplace: true })
                x_test.addColumn("predictions : " + encoder_rf.inverseTransform([0, 1, 2]), encoder_rf.inverseTransform(preds), { inplace: true })
                let metrics = await calculateMetrics(encoded_y_test, preds)
                console.log(metrics);
                x_test.columns.forEach(element => {
                    table_columns.push({ title: element })
                });

                new DataTable('#predictions_table', {
                    responsive: true,
                    columns: table_columns,
                    data: x_test.values
                });

                break
            }
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
function roundInnerArrayNumbers(arr) {
    for (let i = 0; i < arr.length; i++) {
        for (let j = 0; j < arr[i].length; j++) {
            arr[i][j] = (arr[i][j]).toFixed(2); // Round each number in the inner arrays
        }
    }
    return arr;
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
document.getElementById("parseCVS").addEventListener("change", handleFileSelect)
window.pyodide = await loadPyodide();
await pyodide.loadPackage("scikit-learn");













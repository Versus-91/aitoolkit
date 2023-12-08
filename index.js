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
import { Matrix } from 'ml-matrix';
import { ConfusionMatrix } from 'ml-confusion-matrix';

import Bulma from '@vizuaalog/bulmajs';
import { calculateRecall, calculateF1Score, calculatePrecision } from './src/utils.js';
import SVM from "libsvm-js/asm";
import util from 'libsvm-js/src/util.js';
document.addEventListener("DOMContentLoaded", async function (event) {
    // your code here
    sk.setBackend(tensorflow)

    window.tf = tensorflow
    window.jQuery = window.$ = $
    let data_parser = new DataLoader();
    let ui = new UI(data_parser);
    let trainer = new Trainter();
    let chart_controller = new ChartController(data_parser);
    let X
    let y
    const divs = ["lasso_plot"]
    const tbls = ["lasso_plot", "predictions_table", "results", "knn_table", "metrics_table"]

    function handleFileSelect(evt) {
        var target = evt.target || evt.srcElement;
        if (target.value.length == 0) {
            return;
        }
        var file = evt.target.files[0];
        ui.reset(divs, tbls)
        Papa.parse(file, {
            header: true,
            transform: (val) => {
                if (val === "?") {
                    return NaN
                }
                return val
            },
            transformHeader: (val) => {

                return val.replace(/[^a-zA-Z ]/g, "").trim()
            },
            skipEmptyLines: true,
            dynamicTyping: true,
            complete: function (result) {
                console.log(result);
                let dataset = new DataFrame(result.data)
                ui.createDatasetPropsDropdown(dataset);
                document.getElementById("train-button").onclick = async () => {
                    ui.reset(divs, tbls)
                    document.getElementById("train-button").classList.add("is-loading")
                    await train(dataset, result.data.length)
                    document.getElementById("train-button").classList.remove("is-loading")
                }

                document.getElementById("visualize").onclick = async () => {
                    document.getElementById("visualize").classList.add("is-loading")
                    await visualize(dataset, result.data.length, file.name)
                    document.getElementById("visualize").classList.remove("is-loading")
                }

            }
        });
    }
    async function visualize(dataset, len, file_name) {
        ui.renderDatasetStats(dataset);
        let selected_columns = ui.find_selected_columns(dataset.columns)
        let numericColumns = []
        dataset.columns.forEach(column => {
            if (dataset.column(column).dtype !== 'string' && column !== "Id" && selected_columns.includes(column)) {
                numericColumns.push(column)
            }
        });
        const target = document.getElementById("target").value;
        let is_classification = document.getElementById(target).value !== FeatureCategories.Numerical;
        if (numericColumns.length > 0) {
            chart_controller.plot_tsne(dataset.loc({ columns: numericColumns.filter(m => m) }).values, is_classification ? dataset.loc({ columns: [target] }).values : []);
            chart_controller.draw_pca(dataset.loc({ columns: numericColumns }).values, is_classification ? dataset.loc({ columns: [target] }).values : []);
            document.getElementById("container").innerHTML = "";
            numericColumns.forEach(col => {
                chart_controller.draw_kde(dataset, col)
            });

        }
        if (is_classification) {
            let counts = dataset.column(target).valueCounts()
            const df = new DataFrame({
                values: counts.values,
                labels: counts.$index,
            });

            chart_controller.classification_target_chart(counts.values, counts.$index, file_name, "y_pie_chart")
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
                    for (let k = 2; k < 12; k++) {
                        await knn_classifier.train(x_train.values, encoded_y_train, k)
                        let y_preds = knn_classifier.predict(x_test.values)
                        let evaluation_result = evaluate_classification(y_preds, encoded_y_test)
                        results.push({ k: k, predictions: y_preds, evaluation: evaluation_result })
                    }

                    let best_result = results[0];
                    results.forEach(element => {
                        if (element.evaluation.accuracy > best_result.accuracy) {
                            best_result = element
                        }
                    });

                    let predictions = best_result.predictions
                    let knn_table_column_names = []
                    knn_table_column_names.push({ title: "k" })
                    knn_table_column_names.push({ title: "accuracy" })
                    let knn_accuracies = results.map(m => [m.k, m.evaluation.accuracy.toFixed(2)])
                    new DataTable('#knn_table', {
                        responsive: true,
                        columns: knn_table_column_names,
                        data: knn_accuracies,
                        bDestroy: true,
                    });
                    chart_controller.draw_classification_pca(x_test.values, y_test.values, best_result.evaluation.indexes)
                    const matrix = await plot_confusion_matrix(window.tf.tensor(predictions), window.tf.tensor(encoded_y_test), encoder.inverseTransform(Object.values(encoder.$labels)))
                    metrics_table(encoder.inverseTransform(Object.values(encoder.$labels)), matrix)
                    predictions_table(x_test, y_test, encoder, best_result.predictions)
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
                    chart_controller.draw_classification_pca(x_test.values, y_test.values, evaluation_result.indexes)
                    predictions_table(x_test, y_test, encoder, y_preds)
                    const matrix = plot_confusion_matrix(window.tf.tensor(y_preds), window.tf.tensor(encoded_y_test), encoder.inverseTransform(Object.values(encoder.$labels)))
                    metrics_table(encoder.inverseTransform(Object.values(encoder.$labels)), matrix)
                    break;
                }
                case Settings.classification.naive_bayes.lable: {
                    let model = model_factory.createModel(Settings.classification.naive_bayes)
                    let results = []
                    let encoder = new LabelEncoder()
                    encoder.fit(targets)
                    let encoded_y_train = encoder.transform(y_train.values)
                    let encoded_y_test = encoder.transform(y_test.values)
                    model.train(x_train.values, encoded_y_train)
                    let y_preds = model.predict(x_test.values)
                    let evaluation_result = evaluate_classification(y_preds, encoded_y_test)
                    chart_controller.draw_classification_pca(x_test.values, y_test.values, evaluation_result.indexes)
                    const matrix = await plot_confusion_matrix(window.tf.tensor(y_preds), window.tf.tensor(encoded_y_test), encoder.inverseTransform(Object.values(encoder.$labels)))
                    metrics_table(encoder.inverseTransform(Object.values(encoder.$labels)), matrix)
                    predictions_table(x_test, y_test, encoder, y_preds)
                    break;
                }
                case Settings.classification.boosting.lable: {
                    let model = model_factory.createModel(Settings.classification.boosting, null, {
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
                    chart_controller.draw_classification_pca(x_test.values, y_test.values, evaluation_result.indexes)
                    const matrix = await plot_confusion_matrix(window.tf.tensor(y_preds), window.tf.tensor(encoded_y_test), encoder.inverseTransform(Object.values(encoder.$labels)))
                    predictions_table(x_test, y_test, encoder, y_preds)
                    metrics_table(encoder.inverseTransform(Object.values(encoder.$labels)), matrix)
                    break;
                }
                case Settings.classification.logistic_regression.lable: {
                    let logistic_regression = model_factory.createModel(Settings.classification.logistic_regression, chart_controller, {
                        numFeatures: 4, numClasses: 3, learningRate: 0.01, l1Regularization: 0, l2Regularization: 0
                    })
                    let encoder = new LabelEncoder()
                    encoder.fit(y_train.values)
                    let y = encoder.transform(y_train.values)
                    let y_t = encoder.transform(y_test.values)
                    let { preds, probs, coefs, alphas } = await logistic_regression.fit(x_train.values, y, x_test.values, y_t)

                    let evaluation_result = evaluate_classification(preds, y_t)
                    chart_controller.draw_classification_pca(x_test.values, y_t, evaluation_result.indexes)
                    const classes = encoder.inverseTransform(Object.values(encoder.$labels))
                    const matrix = await plot_confusion_matrix(window.tf.tensor(preds), window.tf.tensor(y_t), classes)
                    metrics_table(classes, matrix)

                    chart_controller.regularization_plot(alphas, coefs, x_test.columns)
                    predictions_table(x_test, y_test, encoder, preds, probs);
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
                    const classes = encoder_rf.inverseTransform(Object.values(encoder_rf.$labels))
                    chart_controller.draw_classification_pca(x_test.values, encoded_y_test, evaluation_result.indexes)
                    const matrix = await plot_confusion_matrix(window.tf.tensor(preds), window.tf.tensor(encoded_y_test), classes)
                    metrics_table(classes, matrix)
                    chart_controller.draw_classification_pca(x_test.values, encoded_y_test, evaluation_result.indexes)

                    predictions_table(x_test, y_test, encoder_rf, preds);

                    break
                }
                default:
                    break;
            }
        } else {
            model = await trainer.train_linear_regression(selected_columns.length, dataset.loc({ columns: selected_columns }).tensor, dataset.column(target).tensor)
        }

        function predictions_table(x, y, encoder, preds, probs = null) {

            let table_columns = [];
            if (probs !== null) {
                x.addColumn("probs", probs, { inplace: true });
            }
            x.addColumn("y", y, { inplace: true });
            x.addColumn("predictions: ", encoder.inverseTransform(preds), { inplace: true });
            x.columns.forEach(element => {
                table_columns.push({ title: element });
            });
            new DataTable('#predictions_table', {
                responsive: true,
                columns: table_columns,
                data: x.values,
                bDestroy: true,
                rowCallback: function (row, data, index) {
                    var column1Value = data[table_columns.length - 1];
                    var column2Value = data[table_columns.length - 2];
                    if (column1Value !== column2Value) {
                        $(row).css('background-color', '#97233F');
                        $(row).css('color', 'white');
                    }
                }
            });
        }
        function metrics_table(labels, matrix) {

            let metrics = []
            for (let i = 0; i < matrix.length; i++) {
                metrics.push([
                    labels[i],
                    calculateRecall(i, matrix).toFixed(4),
                    calculatePrecision(i, matrix).toFixed(4),
                    calculateF1Score(i, matrix).toFixed(4),
                ]
                )
            }
            new DataTable('#metrics_table', {
                responsive: true,
                columns: [{ title: "Class" }, { title: "Recall" }, { title: "Precision" }, { title: "f1 score" }],
                data: metrics,
                info: false,
                search: false,
                ordering: false,
                searching: false,
                paging: false,
                bDestroy: true,
            });
        }
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
        console.log(confusionMatrix);
        const container = document.getElementById("confusion-matrix");
        tfvis.render.confusionMatrix(container, {
            values: confusionMatrix,
            tickLabels: lables ?? null
        });
        window.tf.dispose(y)
        window.tf.dispose(predictedLabels)
        return confusionMatrix

    }
    document.getElementById("parseCVS").addEventListener("change", handleFileSelect)

    window.pyodide = await loadPyodide();
    await pyodide.loadPackage("scikit-learn");

});






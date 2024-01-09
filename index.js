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
import Plotly from 'plotly.js-dist';

import Bulma from '@vizuaalog/bulmajs';
import { calculateRecall, calculateF1Score, calculatePrecision } from './src/utils.js';
import SVM from "libsvm-js/asm";
import util from 'libsvm-js/src/util.js';
import Table from '@editorjs/table';
import EditorJS from '@editorjs/editorjs';
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
        try {
            Papa.parse(file, {
                header: true,
                transform: (val) => {
                    if (val === "?" || val === "NA") {
                        return NaN
                    }
                    return val
                },
                transformHeader: (val) => {

                    return val.replace(/[^a-zA-Z ]/g, "").trim()
                },
                skipEmptyLines: true,
                dynamicTyping: true,
                complete: async function (result) {
                    let dataset = new DataFrame(result.data)
                    ui.createDatasetPropsDropdown(dataset);
                    await visualize(dataset, result.data.length, file.name)
                    document.getElementById("train-button").onclick = async () => {
                        ui.reset(divs, tbls)
                        ui.start_loading()
                        await train(dataset, result.data.length)
                        ui.stop_loading()
                    }

                    document.getElementById("visualize").onclick = async () => {
                        document.getElementById("visualize").classList.add("is-loading")
                        await visualize(dataset, result.data.length, file.name)
                        document.getElementById("visualize").classList.remove("is-loading")
                    }

                }
            });
        } catch (error) {
            ui.stop_loading()
            ui.show_error_message(error.message, "#7E191B")
        }

    }

    function get_numeric_columns(dataset, filter) {
        let selected_columns = ui.find_selected_columns(dataset.columns, !filter)
        let numericColumns = []
        dataset.columns.forEach(column => {
            if (dataset.column(column).dtype !== 'string' && column !== "Id" && selected_columns.includes(column)) {
                numericColumns.push(column)
            }
        });
        return numericColumns
    }
    async function visualize(dataset, len, file_name) {
        try {

            ui.renderDatasetStats(dataset);
            let numericColumns = get_numeric_columns(dataset, false)
            const target = document.getElementById("target").value;
            const index = numericColumns.findIndex(m => m === target)
            if (index === -1) {
                numericColumns.push(target)
            }
            const filterd_dataset = dataset.loc({ columns: numericColumns })
            filterd_dataset.dropNa({ axis: 1, inplace: true })
            numericColumns = numericColumns.filter(m => m !== target)
            let is_classification = document.getElementById(target).value !== FeatureCategories.Numerical;
            if (numericColumns.length > 0) {
                document.getElementById("container").innerHTML = "";
                numericColumns.forEach(col => {
                    chart_controller.draw_kde(filterd_dataset, col)
                });
                // chart_controller.plot_tsne(filterd_dataset.loc({ columns: numericColumns }).values, is_classification ? filterd_dataset.loc({ columns: [target] }).values : []);
                // if (numericColumns.length > 2) {
                //     chart_controller.draw_pca(filterd_dataset.loc({ columns: numericColumns }).values, is_classification ? filterd_dataset.loc({ columns: [target] }).values : []);
                // }
            }
            if (is_classification) {
                let counts = filterd_dataset.column(target).valueCounts()
                chart_controller.classification_target_chart(counts.values, counts.$index, file_name, "y_pie_chart")
            }
        } catch (error) {
            ui.stop_loading()
            ui.show_error_message(error.message, "#7E191B")
        }
    }

    async function train(data, len) {
        try {
            let dataset = data.copy()
            let model_name = document.getElementById('model_name').value
            const target = document.getElementById("target").value;
            dataset = data_parser.handle_missing_values(dataset)
            let selected_columns = ui.find_selected_columns(dataset.columns)
            const index = selected_columns.findIndex(m => m === target)
            if (index === -1) {
                selected_columns.push(target)
            }
            if (selected_columns.length < 2) {
                throw new Error("most select at least 2 features")
            }
            let filterd_dataset = dataset.loc({ columns: selected_columns })
            filterd_dataset.dropNa({ axis: 1, inplace: true })

            const targets = filterd_dataset.column(target)
            filterd_dataset.drop({ columns: target, inplace: true })
            const cross_validation_setting = +document.getElementById("cross_validation").value
            filterd_dataset = data_parser.encode_dataset(filterd_dataset, ui.find_selected_columns_types(filterd_dataset.columns), model_name)
            let x_train, y_train, x_test, y_test;
            if (cross_validation_setting === 1) {
                const limit = Math.ceil(len * 70 / 100)
                const train_bound = `0:${limit}`
                const test_bound = `${limit}:${len}`
                x_train = filterd_dataset.iloc({ rows: [`0: ${limit}`] })
                y_train = targets.iloc([train_bound])
                x_test = filterd_dataset.iloc({ rows: [`${limit}: ${len}`] });
                y_test = targets.iloc([test_bound]);
            } else if (cross_validation_setting === 2) {
                x_train = filterd_dataset
                y_train = targets
                x_test = filterd_dataset
                y_test = targets
            }

            let model_factory = new ModelFactory();
            if (document.getElementById(target).value !== FeatureCategories.Numerical) {
                switch (model_name) {
                    case Settings.classification.k_nearest_neighbour.label: {
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
                    case Settings.classification.support_vector_machine.label: {
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
                        const matrix = await plot_confusion_matrix(window.tf.tensor(y_preds), window.tf.tensor(encoded_y_test), encoder.inverseTransform(Object.values(encoder.$labels)))
                        metrics_table(encoder.inverseTransform(Object.values(encoder.$labels)), matrix)
                        break;
                    }
                    case Settings.classification.naive_bayes.label: {
                        let model = model_factory.createModel(Settings.classification.naive_bayes)
                        let results = []
                        let encoder = new LabelEncoder()
                        encoder.fit(targets)
                        let encoded_y_train = encoder.transform(y_train.values)
                        let encoded_y_test = encoder.transform(y_test.values)
                        model.train(x_train.values, encoded_y_train)
                        let y_preds = Array.from(model.predict(x_test.values))
                        let evaluation_result = evaluate_classification(y_preds, encoded_y_test)
                        chart_controller.draw_classification_pca(x_test.values, y_test.values, evaluation_result.indexes)
                        const matrix = await plot_confusion_matrix(window.tf.tensor(y_preds), window.tf.tensor(encoded_y_test), encoder.inverseTransform(Object.values(encoder.$labels)))
                        metrics_table(encoder.inverseTransform(Object.values(encoder.$labels)), matrix)
                        predictions_table(x_test, y_test, encoder, y_preds)
                        break;
                    }
                    case Settings.classification.boosting.label: {
                        let model = model_factory.createModel(Settings.classification.boosting, {
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
                    case Settings.classification.logistic_regression.label: {
                        let logistic_regression = model_factory.createModel(Settings.classification.logistic_regression, chart_controller, {
                            numFeatures: 4, numClasses: 3, learningRate: 0.01, l1Regularization: 0, l2Regularization: 0
                        })

                        let encoder = new LabelEncoder()
                        encoder.fit(y_train.values)
                        let y = encoder.transform(y_train.values)
                        let y_t = encoder.transform(y_test.values)
                        logistic_regression.train(x_train.values, y)
                        const preds = logistic_regression.predict(x_test.values)
                        let evaluation_result = evaluate_classification(preds, y_t)
                        chart_controller.draw_classification_pca(x_test.values, y_t, evaluation_result.indexes)
                        const classes = encoder.inverseTransform(Object.values(encoder.$labels))
                        const matrix = await plot_confusion_matrix(window.tf.tensor(preds), window.tf.tensor(y_t), classes)
                        metrics_table(classes, matrix)
                        let probs = logistic_regression.predict_probas(x_test.values)
                        // chart_controller.regularization_plot(alphas, coefs, x_test.columns)
                        predictions_table(x_test, y_test, encoder, preds, probs);
                        // chart_controller.probablities_boxplot(probs, classes)
                        // chart_controller.probablities_violin_plot(probs, classes)

                        break
                    }
                    case Settings.classification.random_forest.label: {

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
                        ui.show_error_message("to be implemented", "#7E191B")
                        break;
                }
            } else {
                switch (model_name) {
                    case Settings.regression.linear_regression.label:
                        let model = model_factory.createModel(Settings.regression.linear_regression, null, {});
                        let y_t = y_train.values.map((item) => [item])
                        let y_test_t = y_test.values.map((item) => [item])
                        model.train([x_train.values].flat(), [y_t].flat())
                        let preds = model.predict([x_test.values].flat(), [y_test_t].flat())
                        const xs = Array.from(Array(x_test.$data.length).keys())
                        var trace1 = {
                            x: xs,
                            y: y_test.values,
                            type: 'scatter',
                            name: "y"
                        };
                        var trace2 = {
                            x: xs,
                            y: preds.flat(),
                            type: 'scatter',
                            name: "pred"
                        };

                        var data = [trace1, trace2];
                        console.log(model.stats());
                        Plotly.newPlot('regression_y_yhat', data, { title: "y vs y hat", plot_bgcolor: "#E5ECF6" });

                }
            }

        } catch (error) {
            ui.stop_loading()
            ui.show_error_message(error.message, "#7E191B")
            throw error
        }

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

    async function plot_confusion_matrix(y, predictedLabels, labels = null) {
        let tabs = Bulma('.tabs-wrapper').data('tabs');
        tabs.setActive(2)
        const confusionMatrix = await tfvis.metrics.confusionMatrix(y, predictedLabels);
        const container = document.getElementById("confusion-matrix");
        await tfvis.render.confusionMatrix(container, {
            values: confusionMatrix,
            tickLabels: labels
        });
        window.tf.dispose(y)
        window.tf.dispose(predictedLabels)
        return confusionMatrix

    }

    ui.init_upload_button(handleFileSelect)
    // const webR = new WebR();
    // await webR.init();
    // let result = await webR.evalR(`
    // fit <- lm(mpg ~ am, data=mtcars)
    // summary_data<-summary(fit)
    // result <- list(
    //     coefficients = as.data.frame(summary_data$coefficients)
    //   )

    // `)
    // console.log((await result.toJs()).values);
    const editorjs = new EditorJS({
        holder: 'editorjs',
        tools: {
            table: {
                class: Table,
                inlineToolbar: true,
                config: {
                    rows: 2,
                    cols: 3,
                    withHeadings: true,
                },
            },
        },
        //data: editorjsdata,
    });
    editorjs.isReady.then(() => {
        editorjs.blocks.insert("table")

    })
});






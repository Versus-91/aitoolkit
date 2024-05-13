"use strict";
import { DataFrame, tensorflow, LabelEncoder } from 'danfojs/dist/danfojs-base';
import Papa from 'papaparse';
import ChartController from "./src/charts.js";
import DataLoader from "./src/data.js";
import UI from "./src/ui.js";
import { FeatureCategories, Settings } from './feature_types.js';
import { ModelFactory } from './src/model_factory.js';
import * as tfvis from '@tensorflow/tfjs-vis';
import * as sk from 'scikitjs'
import Plotly from 'plotly.js-dist';
import Bulma from '@vizuaalog/bulmajs';
import { calculateRecall, calculateF1Score, calculatePrecision, metrics } from './src/utils.js';
import SVM from "libsvm-js/asm";
import tippy from 'tippy.js';
import 'tippy.js/dist/tippy.css'; // optional for styling
document.addEventListener("DOMContentLoaded", async function (event) {
    // your code here
    sk.setBackend(tensorflow)
    let data_frame;
    var mltool = {
        model_number: 0
    };
    window.tf = tensorflow
    let data_parser = new DataLoader();
    let chart_controller = new ChartController(data_parser);
    let ui = new UI(data_parser, chart_controller);
    const html_content_ids = ["lasso_plot", "formulas", "regression_y_yhat", "probs_violin_plot", "probs_box_plot"]
    const table_ids = ["lasso_plot", "predictions_table", "results", "knn_table", "metrics_table", "stats_table", "sample_data_table"]
    const plots = ["tsne", "pca-1", "pca-2", "pca-3"]

    async function handleFileSelect(evt, url) {
        var target = evt?.target || evt?.srcElement;
        let file;
        if (target?.value.length == 0) {
            return;
        }
        if (!url) {
            file = evt.target.files[0];
            await process_file(file)
        } else {
            fetch(url)
                .then(response => response.blob())
                .then(async blob => {
                    file = new File([blob], "url");
                    await process_file(file)
                })
                .catch(error => {
                    console.error('Error fetching the file:', error);
                });
        }

    }
    async function process_file(file) {
        try {
            ui.reset(html_content_ids, table_ids, plots);
            ui.toggle_loading_progress();
            Papa.parse(file, {
                worker: false,
                header: true,
                transform: (val) => {
                    if (val === "?" || val === "NA") {
                        return NaN
                    }
                    return val
                },
                transformHeader: (val) => {
                    return val.replace(/[^a-zA-Z0-9 ]/g, "").trim()
                },
                skipEmptyLines: true,
                dynamicTyping: true,
                complete: async function (result) {
                    ui.toggle_loading_progress(true);
                    if (result.data.length > 10000) {
                        result.data = result.data.slice(0, 10000)
                    }
                    let dataset = new DataFrame(result.data);
                    data_frame = new DataFrame(result.data);
                    ui.createDatasetPropsDropdown(dataset);
                    ui.createSampleDataTable(dataset);
                    await ui.visualize(dataset, result.data.length, file.name);
                    tippy('#kde_help', {
                        interactive: true,
                        popperOptions: {
                            positionFixed: true,
                        },
                        content: 'Default bandwidth method :Silverman’s rule of thumb',
                    });
                    tippy('#normalization_help', {
                        interactive: true,
                        popperOptions: {
                            positionFixed: true,
                        },
                        content: '<p>not functional yet</p><p>standard scaler uses z = (x - u) / s</p><p>Transform features by scaling each feature to a given range</p>',
                        allowHTML: true,
                    });
                    tippy('#imputation_help', {
                        interactive: true,
                        popperOptions: {
                            positionFixed: true,
                        },
                        content: 'currently we are just deleting rows with missing values',
                        allowHTML: true,
                    });
                    tippy('#cv_help', {
                        interactive: true,
                        popperOptions: {
                            positionFixed: true,
                        },
                        content: 'option 1 and 2 are working',
                        allowHTML: true,
                    });
                    document.querySelector('#feature_selection_modal').addEventListener('update_graphs', async function (e) {
                        await ui.visualize(data_frame);
                    });
                    document.getElementById("train-button").onclick = async () => {
                        ui.reset(html_content_ids, table_ids.filter(m => m !== "sample_data_table"));
                        ui.start_loading();
                        await train(dataset, result.data.length);
                        ui.stop_loading();
                    }

                }
            });
        } catch (error) {
            ui.toggle_loading_progress(true);
            ui.stop_loading();
            ui.show_error_message(error.message, "#7E191B");
        }
    }

    async function dimension_reduction(is_pca = true) {
        try {
            let dataset = data_frame;
            ui.renderDatasetStats(dataset);
            let numericColumns = ui.get_numeric_columns(dataset, true)
            const target = document.getElementById("target").value;
            const index = numericColumns.findIndex(m => m === target)
            if (index === -1) {
                numericColumns.push(target)
            }
            const filterd_dataset = dataset.loc({ columns: numericColumns })
            filterd_dataset.dropNa({ axis: 1, inplace: true })
            numericColumns = numericColumns.filter(m => m !== target)
            let is_classification = document.getElementById(target).value !== FeatureCategories.Numerical;
            if (numericColumns.length > 2) {
                if (is_pca) {
                    document.getElementById("dim_red_button_pca").classList.add("is-loading")
                    await chart_controller.draw_pca(filterd_dataset.loc({ columns: numericColumns }).values, is_classification ? filterd_dataset.loc({ columns: [target] }).values : []);
                    document.getElementById("dim_red_button_pca").classList.remove("is-loading")
                } else {
                    document.getElementById("dim_red_button_tsne").classList.add("is-loading")
                    await chart_controller.plot_tsne(filterd_dataset.loc({ columns: numericColumns }).values, is_classification ? filterd_dataset.loc({ columns: [target] }).values : []);
                    document.getElementById("dim_red_button_tsne").classList.remove("is-loading")
                }
            } else {
                throw new Error("Most select at least 2 features.")
            }


        } catch (error) {
            ui.stop_loading()
            ui.show_error_message(error.message, "#7E191B")
            document.getElementById("dim_red_button_pca").classList.remove("is-loading")
            document.getElementById("dim_red_button_tsne").classList.remove("is-loading")


        }
    }
    async function train(data, len) {
        try {
            let dataset = data.copy()
            let model_name = document.getElementById('model_name').value
            model_name = parseInt(model_name)
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
            let model_settings = ui.get_model_settings();
            mltool.model_number++
            ui.create_model_result_tab(mltool.model_number)
            ui.show_settings(model_settings, mltool.model_number);
            let tabs = Bulma('.tabs-wrapper').data('tabs');
            tabs.setActive(2)
            // $(this).toggleClass("is-active ");
            if (document.getElementById(target).value !== FeatureCategories.Numerical) {
                let uniqueLabels = [...new Set(y_train.values)];
                switch (model_name) {
                    case Settings.classification.k_nearest_neighbour.value: {
                        let knn_classifier = model_factory.createModel(Settings.classification.k_nearest_neighbour, model_settings)
                        let results = []
                        let encoder = new LabelEncoder()
                        encoder.fit(targets)
                        let encoded_y_train = encoder.transform(y_train.values)
                        let encoded_y_test = encoder.transform(y_test.values)
                        for (let k = model_settings.min; k <= model_settings.max; k++) {
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
                            paging: false,
                            searching: false,
                        });
                        await chart_controller.draw_classification_pca(x_test.values, y_test.values, best_result.evaluation.indexes, uniqueLabels, mltool.model_number)
                        const classes = encoder.inverseTransform(Object.values(encoder.$labels))
                        const matrix = await plot_confusion_matrix(window.tf.tensor(predictions), window.tf.tensor(encoded_y_test), encoder.inverseTransform(Object.values(encoder.$labels)), encoder.transform(classes))
                        //metrics_table(encoder.inverseTransform(Object.values(encoder.$labels)), matrix)
                        predictions_table(x_test, y_test, encoder, best_result.predictions)
                        break;
                    }
                    case Settings.classification.support_vector_machine.value: {

                        let model = model_factory.createModel(Settings.classification.support_vector_machine, {
                            kernel: SVM.KERNEL_TYPES[model_settings.kernel.toUpperCase()],
                            type: SVM.SVM_TYPES.C_SVC,
                            coef0: model_settings.bias,
                            gamma: model_settings.gamma,
                            degree: model_settings.degree,
                            quiet: true
                        })
                        let results = []
                        let encoder = new LabelEncoder()
                        encoder.fit(targets)
                        let encoded_y_train = encoder.transform(y_train.values)
                        let encoded_y_test = encoder.transform(y_test.values)
                        await model.train(x_train.values, encoded_y_train)
                        let y_preds = model.predict(x_test.values)
                        const classes = encoder.inverseTransform(Object.values(encoder.$labels))
                        await plot_confusion_matrix(window.tf.tensor(y_preds), window.tf.tensor(encoded_y_test), encoder.inverseTransform(Object.values(encoder.$labels)), encoder.transform(classes))
                        let evaluation_result = evaluate_classification(y_preds, encoded_y_test)
                        await chart_controller.draw_classification_pca(x_test.values, y_test.values, evaluation_result.indexes, uniqueLabels, mltool.model_number)
                        predictions_table(x_test, y_test, encoder, y_preds)
                        //metrics_table(encoder.inverseTransform(Object.values(encoder.$labels)), matrix)
                        break;
                    }
                    case Settings.classification.naive_bayes.value: {
                        let model = model_factory.createModel(Settings.classification.naive_bayes, model_settings)
                        let results = []
                        let encoder = new LabelEncoder()
                        encoder.fit(targets)
                        let encoded_y_train = encoder.transform(y_train.values)
                        let encoded_y_test = encoder.transform(y_test.values)
                        let y_preds
                        if (model_settings.type === 'Gaussian') {
                            await model.train(x_train.values, encoded_y_train)
                            y_preds = Array.from(model.predict(x_test.values))
                        } else {
                            y_preds = Array.from(await model.train(x_train.values, encoded_y_train, x_test.values))
                        }
                        let evaluation_result = evaluate_classification(y_preds, encoded_y_test)
                        await chart_controller.draw_classification_pca(x_test.values, y_test.values, evaluation_result.indexes, uniqueLabels, mltool.model_number)
                        const classes = encoder.inverseTransform(Object.values(encoder.$labels))
                        const matrix = await plot_confusion_matrix(window.tf.tensor(y_preds), window.tf.tensor(encoded_y_test), encoder.inverseTransform(Object.values(encoder.$labels)), encoder.transform(classes))
                        //metrics_table(encoder.inverseTransform(Object.values(encoder.$labels)), matrix)
                        predictions_table(x_test, y_test, encoder, y_preds)
                        break;
                    }
                    case Settings.classification.boosting.value: {
                        let model = model_factory.createModel(Settings.classification.boosting, {
                            booster: model_settings.booster ?? "gbtree",
                            objective: uniqueLabels.length > 1 ? "multi:softmax" : "binary:logistic",
                            max_depth: model_settings.depth,
                            eta: model_settings.eta,
                            min_child_weight: 1,
                            subsample: 0.5,
                            colsample_bytree: 1,
                            silent: 1,
                            iterations: model_settings.iterations ?? 200
                        })
                        let results = []
                        let encoder = new LabelEncoder()
                        encoder.fit(targets)
                        let encoded_y_train = encoder.transform(y_train.values)
                        let encoded_y_test = encoder.transform(y_test.values)
                        await model.train(x_train.values, encoded_y_train)
                        let y_preds = await model.predict(x_test.values)
                        const classes = encoder.inverseTransform(Object.values(encoder.$labels))
                        let evaluation_result = evaluate_classification(y_preds, encoded_y_test)
                        const matrix = await plot_confusion_matrix(window.tf.tensor(y_preds), window.tf.tensor(encoded_y_test), encoder.inverseTransform(Object.values(encoder.$labels)), encoder.transform(classes))
                        await chart_controller.draw_classification_pca(x_test.values, y_test.values, evaluation_result.indexes, uniqueLabels, mltool.model_number)
                        predictions_table(x_test, y_test, encoder, y_preds)
                        //metrics_table(encoder.inverseTransform(Object.values(encoder.$labels)), matrix)
                        break;
                    }
                    case Settings.classification.logistic_regression.value: {
                        let logistic_regression = model_factory.createModel(Settings.classification.logistic_regression, chart_controller, {
                            numFeatures: 4, numClasses: 3, learningRate: 0.01, l1Regularization: 0, l2Regularization: 0
                        })

                        let encoder = new LabelEncoder()
                        encoder.fit(y_train.values)
                        let y = encoder.transform(y_train.values)
                        let y_t = encoder.transform(y_test.values)
                        let probs, stats, test, coefs, alphas;
                        [stats, probs, test, coefs, alphas] = await logistic_regression.train(x_train.values, y_train.values, x_test.values, x_train.columns)
                        let preds = []
                        probs.forEach((item => {
                            let key = Object.keys(encoder.$labels)[item.indexOf(Math.max(...item))]
                            preds.push(parseInt(encoder.$labels[key]))
                        }));
                        // preds = encoder.transform(preds);
                        let evaluation_result = evaluate_classification(preds, y_t)
                        await chart_controller.draw_classification_pca(x_test.values, encoder.inverseTransform(y_t), evaluation_result.indexes, uniqueLabels, mltool.model_number)
                        const classes = encoder.inverseTransform(Object.values(encoder.$labels))
                        const matrix = await plot_confusion_matrix(window.tf.tensor(preds), window.tf.tensor(y_t), classes, encoder.transform(classes))
                        metrics_table(classes, matrix)
                        let metrics = []
                        for (let i = 0; i < stats.length; i++) {
                            metrics.push([
                                ...stats[i]])
                        }
                        new DataTable('#stats_table', {
                            responsive: true,
                            columns: [{ title: "variable" }, { title: "coefficient" }, { title: "std error" }, { title: "z" }, { title: "p value" }, { title: "[0.025" }, { title: "0.975]" }],
                            data: metrics,
                            info: false,
                            search: false,
                            ordering: false,
                            searching: false,
                            paging: false,
                            bDestroy: true,
                        });
                        // chart_controller.regularization_plot(alphas, coefs, x_test.columns)
                        predictions_table(x_test, y_test, encoder, preds, probs);
                        chart_controller.probabilities_boxplot(probs, uniqueLabels, y_t)
                        chart_controller.probablities_violin_plot(probs, classes, uniqueLabels)
                        break
                    }
                    case Settings.classification.discriminant_analysis.value: {
                        let model = model_factory.createModel(Settings.classification.discriminant_analysis, { type: model_settings.type === "linear" ? 0 : 1, priors: model_settings.priors })
                        let encoder = new LabelEncoder()
                        encoder.fit(y_train.values)
                        let y = encoder.transform(y_train.values)
                        let y_t = encoder.transform(y_test.values)
                        let preds = await model.train(x_train.values, y, x_test.values)
                        preds = Array.from(preds)
                        let evaluation_result = evaluate_classification(preds, y_t)
                        await chart_controller.draw_classification_pca(x_test.values, encoder.inverseTransform(y_t), evaluation_result.indexes, uniqueLabels, mltool.model_number)
                        const classes = encoder.inverseTransform(Object.values(encoder.$labels))
                        const matrix = await plot_confusion_matrix(window.tf.tensor(preds), window.tf.tensor(y_t), classes, encoder.transform(classes))
                        metrics_table(classes, matrix)
                        predictions_table(x_test, y_test, encoder, preds);
                        break
                    }
                    case Settings.classification.random_forest.value: {
                        let num_features = typeof model_settings.features === "number" ? model_settings.features : parseInt(Math.sqrt(x_train.columns.length).toFixed(0))
                        const model = model_factory.createModel(Settings.classification.random_forest, {
                            seed: 3,
                            maxFeatures: num_features,
                            replacement: true,
                            nEstimators: model_settings.estimators,
                            treeOptions: {
                                maxDepth: model_settings.depth
                            },
                            criteria: model_settings.criteria
                        });
                        let encoder_rf = new LabelEncoder()
                        encoder_rf.fit(y_train.values)
                        encoder_rf.transform(y_train.values)

                        let encoded_y = encoder_rf.transform(y_train.values)
                        let encoded_y_test = encoder_rf.transform(y_test.values)

                        let preds = await model.train_test(x_train.values, encoded_y, x_test.values)
                        const evaluation_result = evaluate_classification(preds, encoded_y_test)
                        const classes = encoder_rf.inverseTransform(Object.values(encoder_rf.$labels))
                        const matrix = await plot_confusion_matrix(window.tf.tensor(preds), window.tf.tensor(encoded_y_test), classes, encoder_rf.transform(classes))
                        //metrics_table(classes, matrix)
                        await chart_controller.draw_classification_pca(x_test.values, y_test.values, evaluation_result.indexes, uniqueLabels, mltool.model_number)

                        predictions_table(x_test, y_test, encoder_rf, preds);

                        break
                    }
                    default:
                        ui.show_error_message("to be implemented", "#7E191B")
                        break;
                }
            } else {
                let content = `
                <div class="column is-12">
                    <div id="regression_y_yhat_${mltool.model_number}" width="100%">
                   </div>
                </div>`
                $("#tabs_info li[data-index='" + mltool.model_number + "'] #results_" + mltool.model_number + "").append(content);
                switch (model_name) {
                    case Settings.regression.linear_regression.value: {
                        let model = model_factory.createModel(Settings.regression.linear_regression, null, {});
                        let y_t = y_train.values.map((item) => [item])
                        let y_test_t = y_test.values.map((item) => [item])
                        model.train([x_train.values].flat(), [y_t].flat())
                        let preds = model.predict([x_test.values].flat(), [y_test_t].flat())
                        var trace1 = {
                            x: y_test.values,
                            y: preds.flat(),
                            type: 'scatter',
                            name: "y",
                            mode: 'markers',
                        };

                        var data = [trace1];
                        let model_stats = model.stats().summary.variables;
                        let model_stats_matrix = [];
                        let columns = x_train.columns
                        columns.push("intercept")
                        for (let i = 0; i < model_stats.length; i++) {
                            let row = [];
                            row.push(columns[i])
                            row.push(model_stats[i].coefficients[0])
                            row.push(model_stats[i].standardError)
                            row.push(model_stats[i].tStat)
                            model_stats_matrix.push(row)
                        }

                        new DataTable('#metrics_table', {
                            responsive: true,
                            columns: [{ title: "variable" }, { title: "weight" }, { title: "std error" }, { title: "t stat" }],
                            data: model_stats_matrix,
                            info: false,
                            search: false,
                            ordering: false,
                            searching: false,
                            paging: false,
                            bDestroy: true,
                        });
                        $("#formulas").html = "";
                        $("#formulas").append(`<span>$$y = {x1 + x2 + x3 + ... + x_n + intercept}.$$</span>`)
                        MathJax.typeset(["formulas"]);
                        Plotly.newPlot('regression_y_yhat_' + mltool.model_number, data, { title: "y vs y hat", plot_bgcolor: "#E5ECF6" }, { responsive: true });
                        break;
                    }
                    case Settings.regression.k_nearest_neighbour.value: {
                        model_settings = ui.get_model_settings();
                        let model = model_factory.createModel(Settings.regression.k_nearest_neighbour, model_settings)
                        // for (let k = model_settings.min; k <= model_settings.max; k++) {
                        //     await model.train(x_train.values, encoded_y_train, k)
                        //     let y_preds = model.predict(x_test.values)
                        //     let evaluation_result = evaluate_classification(y_preds, encoded_y_test)
                        //     results.push({ k: k, predictions: y_preds, evaluation: evaluation_result })
                        // }
                        await model.train(x_train.values, y_train.values, 5)
                        let y_preds = model.predict(x_test.values);
                        var trace = {
                            x: y_test.values,
                            y: y_preds,
                            type: 'scatter',
                            name: "y",
                            mode: 'markers',
                        };
                        Plotly.newPlot('regression_y_yhat_' + mltool.model_number, [trace], { title: "y vs y hat", plot_bgcolor: "#E5ECF6" }, { responsive: true });
                        break;
                    }
                    case Settings.regression.boosting.value: {
                        let model = model_factory.createModel(Settings.regression.boosting, {
                            objective: "reg:linear",
                            iterations: model_settings.iterations ?? 200
                        })
                        await model.train(x_train.values, y_train.values)
                        let y_preds = await model.predict(x_test.values)
                        var trace = {
                            x: y_test.values,
                            y: y_preds,
                            type: 'scatter',
                            name: "y",
                            mode: 'markers',
                        };
                        Plotly.newPlot('regression_y_yhat_' + mltool.model_number, [trace], { title: "y vs y hat", plot_bgcolor: "#E5ECF6" }, { responsive: true });
                        predictions_table_regression(x_test, y_test, y_preds)
                        break;
                    }
                    case Settings.regression.support_vector_machine.value: {
                        let model = model_factory.createModel(Settings.regression.support_vector_machine, {
                            kernel: SVM.KERNEL_TYPES[model_settings.kernel.toUpperCase()],
                            type: SVM.SVM_TYPES.EPSILON_SVR,
                            coef0: model_settings.bias,
                            gamma: model_settings.gamma,
                            degree: model_settings.degree,
                            quiet: true
                        })
                        await model.train(x_train.values, y_train.values)
                        let y_preds = await model.predict(x_test.values)
                        var trace = {
                            x: y_test.values,
                            y: y_preds,
                            type: 'scatter',
                            name: "y",
                            mode: 'markers',
                        };
                        Plotly.newPlot('regression_y_yhat_' + mltool.model_number, [trace], { title: "y vs y hat", plot_bgcolor: "#E5ECF6" }, { responsive: true });
                        predictions_table_regression(x_test, y_test, y_preds)
                        break;
                    }
                    case Settings.regression.random_forest.value: {
                        let num_features = typeof model_settings.features === "number" ? model_settings.features : parseInt(Math.sqrt(x_train.columns.length).toFixed(0))
                        const model = model_factory.createModel(Settings.regression.random_forest, {
                            seed: 3,
                            maxFeatures: num_features,
                            replacement: true,
                            nEstimators: model_settings.estimators,
                            treeOptions: {
                                maxDepth: model_settings.depth
                            },
                            criteria: model_settings.criteria
                        });

                        let preds = await model.train_test(x_train.values, y_train.values, x_test.values)
                        var trace = {
                            x: y_test.values,
                            y: preds,
                            type: 'scatter',
                            name: "y",
                            mode: 'markers',
                        };
                        Plotly.newPlot('regression_y_yhat_' + mltool.model_number, [trace], { title: "y vs y hat", plot_bgcolor: "#E5ECF6" }, { responsive: true });
                        predictions_table_regression(x_test, y_test, preds)

                        break
                    }
                }
            }

        } catch (error) {
            ui.stop_loading()
            ui.show_error_message(error.message, "#7E191B")
            throw error
        }

    }

    function predictions_table(x, y, encoder, preds, probs = null) {
        let content = `
        <div class="column is-12">
            <table id="predictions_table_${mltool.model_number}" class="table is-bordered is-hoverable is-narrow display" width="100%">
           </table>
        </div>`
        $("#tabs_info li[data-index='" + mltool.model_number + "'] #results_" + mltool.model_number + "").append(content);
        let table_columns = [];
        if (probs !== null) {
            x.addColumn("probs", probs, { inplace: true });
        }
        x.addColumn("y", y, { inplace: true });
        x.addColumn("predictions: ", encoder.inverseTransform(preds), { inplace: true });
        x.columns.forEach(element => {
            table_columns.push({ title: element });
        });
        new DataTable('#predictions_table_' + mltool.model_number, {
            pageLength: 25,
            responsive: true,
            paging: true,
            "bPaginate": true,
            columns: table_columns,
            data: x.values,
            bDestroy: true,
            columnDefs: [
                {
                    render: function (data, type, row) {
                        for (let i = 0; i < data.length; i++) {
                            data[i] = data[i].toFixed(2);
                        }
                        return data
                    },
                    targets: [-3]
                }
            ],
            rowCallback: function (row, data, index) {
                var prediction = data[table_columns.length - 1];
                var y = data[table_columns.length - 2];
                if (prediction !== y) {
                    $(row).addClass('is-danger');
                }
            }
        });
    }
    function predictions_table_regression(x, y, preds) {
        let content = `
        <div class="column is-12">
            <table id="predictions_table_${mltool.model_number}" class="table is-bordered is-hoverable is-narrow display" width="100%">
           </table>
        </div>`
        $("#tabs_info li[data-index='" + mltool.model_number + "'] #results_" + mltool.model_number + "").append(content);
        let table_columns = [];
        x.addColumn("y", y, { inplace: true });
        x.addColumn("predictions: ", preds, { inplace: true });
        x.columns.forEach(element => {
            table_columns.push({ title: element });
        });
        new DataTable('#predictions_table_' + mltool.model_number, {
            pageLength: 25,
            responsive: true,
            paging: true,
            "bPaginate": true,
            columns: table_columns,
            data: x.values,
            bDestroy: true,
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
            columns: [{ title: "Class" }, { title: "Recall" }, { title: "Precision" }, { title: "f1" }],
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

    async function plot_confusion_matrix(y, predictedLabels, labels, uniqueClasses) {


        const confusionMatrix = await tfvis.metrics.confusionMatrix(y, predictedLabels);
        let div = document.createElement('div');
        div.classList.add('column');
        div.classList.add('is-6');
        div.setAttribute("id", "result_number_" + mltool.model_number);
        let metric = await metrics(y.arraySync(), predictedLabels.arraySync(), uniqueClasses)
        let info = metric[0]
        var header = "";
        header += "<th></th>";
        var tbody = "";
        for (let i = 0; i < labels.length; i++) {
            header += "<th>" + labels[i] + "</th>";
        }

        let len = confusionMatrix[0].length
        let row = "";
        row += "<td>Preceission</td>";
        for (let j = 0; j < len; j++) {
            row += `<td>${info[0][j].toFixed(2)} </td>`;
        }
        tbody += "<tr>" + row + "</tr>";

        row = "";
        row += "<td>Recall</td>";
        for (let j = 0; j < len; j++) {
            row += `<td>${info[1][j].toFixed(2)} </td>`;
        }
        tbody += "<tr>" + row + "</tr>";

        row = ""
        row += "<td>F1 score</td>";
        for (let j = 0; j < len; j++) {
            row += `<td>${info[2][j].toFixed(2)} </td>`;
        }
        tbody += "<tr>" + row + "</tr>";

        row = ""
        row += "<td>Support</td>";
        for (let j = 0; j < len; j++) {
            row += `<td>${info[3][j].toFixed(2)} </td>`;
        }
        tbody += "<tr>" + row + "</tr>";
        div.innerHTML =
            '<div class="table-container"><table class="table is-fullwidth is-bordered is-striped is-narrow is-hoverable is-size-7"><thead>' +
            header +
            "</thead><tbody>" +
            tbody +
            "</tbody></table></div>"
            ;
        $("#tabs_info li[data-index='" + mltool.model_number + "'] #results_" + mltool.model_number + "").append(div);
        $("#tabs_info li[data-index='" + mltool.model_number + "'] #results_" + mltool.model_number + "").append(`
        <div class="column is-6" id="confusion_matrix_${mltool.model_number}">
        </div>
        `);
        window.tf.dispose(y)
        window.tf.dispose(predictedLabels)

        const container = document.getElementById("confusion_matrix_" + mltool.model_number);
        await tfvis.render.confusionMatrix(container, {
            values: confusionMatrix,
            tickLabels: labels
        });
        return confusionMatrix

    }
    ui.init_upload_button(handleFileSelect)
    document.querySelector('#dim_red_button_pca').addEventListener('click', async function (e) {
        await dimension_reduction();
    });
    document.querySelector('#dim_red_button_tsne').addEventListener('click', async function (e) {
        await dimension_reduction(false);
    });
    document.getElementById("sample_data_select").addEventListener('change', async function (e) {
        handleFileSelect(null, e.target.value.toLowerCase() + '.csv')

    })
    $("#myList li").click(function () {
        $(this).toggleClass("active");
    });
    $("#tabs_content").on("click", "li", function () {
        var index = $(this).data("index");
        $("#tabs_content li").not(this).removeClass("is-active");
        $("#tabs_info li").removeClass("is-active");
        $("#tabs_info li[data-index='" + index + "']").addClass("is-active");
        $(this).toggleClass("is-active ");
    });
});






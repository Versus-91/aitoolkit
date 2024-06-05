import { DataFrame, tensorflow, LabelEncoder } from 'danfojs/dist/danfojs-base';
import Papa from 'papaparse';
import ChartController from "./src/charts.js";
import DataLoader from "./src/data.js";
import UI from "./src/ui.js";
import { FeatureCategories, Settings } from './feature_types.js';
import { ModelFactory } from './src/model_factory.js';
import * as sk from 'scikitjs'
import Plotly from 'plotly.js-dist';
import Bulma from '@vizuaalog/bulmajs';
import { evaluate_classification, apply_data_transformation } from './src/utils.js';
import SVM from "libsvm-js/asm";
import { ParserFactory } from "./src/parsers/parser_factory.js";
import tippy from 'tippy.js';
import 'tippy.js/dist/tippy.css'; // optional for styling
document.addEventListener("DOMContentLoaded", async function (event) {
    // your code here
    sk.setBackend(tensorflow)
    let data_frame;
    var mltool = {
        model_number: 0
    };
    window.tensorflow = tensorflow
    let data_parser = new DataLoader();
    let chart_controller = new ChartController(data_parser);
    let ui = new UI(data_parser, chart_controller);
    const html_content_ids = []
    const table_ids = ["sample_data_table"]
    const plots = ["tsne", "pca-1", "pca-2", "pca-3"]

    async function handleFileSelect(evt, url) {
        var target = evt?.target || evt?.srcElement;
        let file;
        if (target?.value.length == 0) {
            return;
        }
        if (!url) {
            file = evt.target.files[0];
            await process_file(file, file.name.split('.')[1])
        } else {
            fetch(url)
                .then(response => response.blob())
                .then(async blob => {
                    file = new File([blob], "url");
                    await process_file(file, 'csv')
                })
                .catch(error => {
                    console.error('Error fetching the file:', error);
                });
        }

    }
    function init_canvas() {
        $('#canvas-container').empty()
        $('#canvas-container').append(`<canvas id="canvasId" responsive='true'></canvas>`)
        let canvas = document.getElementById('canvasId')
        let container = document.getElementById('canvas-container')
        canvas.width = container.offsetWidth;
        canvas.height = 600;
    }
    async function process_file(file, type) {
        try {
            ui.reset(html_content_ids, table_ids, plots);
            ui.toggle_loading_progress(true);
            let options = {
                separator: $('#items_separator').find(":selected").val(),
                delimiter: $('#decimal_separator').find(":selected").val(),
                header: $('#header_checkbox').is(":checked")
            }
            let result = await ParserFactory.createParser(type, options).parse(file)
            if (result.length > 10000) {
                result = result.slice(0, 10000)
            }
            let dataset = new DataFrame(result);
            data_frame = new DataFrame(result);
            ui.createDatasetPropsDropdown(dataset);
            ui.createSampleDataTable(dataset);
            CanvasXpress.destroy('canvasId');
            init_canvas()
            await ui.visualize(dataset, result.length, file.name);
            ui.init_tooltips(tippy)
            document.querySelector('#feature_selection_modal').addEventListener('update_graphs', async function (e) {
                CanvasXpress.destroy('canvasId');
                init_canvas()
                await ui.visualize(data_frame);
            });
            document.getElementById("train-button").onclick = async () => {
                ui.reset(html_content_ids, table_ids.filter(m => m !== "sample_data_table"));
                ui.start_loading();
                await train(dataset, result.length);
                ui.stop_loading();
            }
        } catch (error) {
            ui.toggle_loading_progress(true);
            ui.stop_loading();
            ui.show_error_message(error.message, "#7E191B");
            console.log(error.message);
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
                let transformed_data = apply_data_transformation(filterd_dataset.loc({ columns: numericColumns }), numericColumns);
                if (is_pca) {
                    document.getElementById("dim_red_button_pca").classList.add("is-loading")
                    await chart_controller.draw_pca(transformed_data.values, is_classification ? filterd_dataset.loc({ columns: [target] }).values : [], filterd_dataset.loc({ columns: [target] }).values);
                    document.getElementById("dim_red_button_pca").classList.remove("is-loading")
                } else {
                    document.getElementById("dim_red_button_tsne").classList.add("is-loading")
                    await chart_controller.plot_tsne(transformed_data.values, is_classification ? filterd_dataset.loc({ columns: [target] }).values : [], filterd_dataset.loc({ columns: [target] }).values);
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
            // let dataset = data.copy()
            let dataset = await data.sample(data.$data.length);
            let numericColumns = ui.get_numeric_columns(dataset, true)
            let model_name = document.getElementById('model_name').value
            model_name = parseInt(model_name)
            const target = document.getElementById("target").value;
            dataset = data_parser.handle_missing_values(dataset)
            dataset = apply_data_transformation(dataset, numericColumns);
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
                            let predictions = knn_classifier.predict(x_test.values)
                            let evaluation_result = evaluate_classification(predictions, encoded_y_test)
                            results.push({ k: k, predictions: predictions, evaluation: evaluation_result })
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
                        const classes = encoder.inverseTransform(Object.values(encoder.$labels))
                        await chart_controller.plot_confusion_matrix(window.tensorflow.tensor(predictions), window.tensorflow.tensor(encoded_y_test), encoder.inverseTransform(Object.values(encoder.$labels)), encoder.transform(classes), mltool.model_number)
                        await chart_controller.draw_classification_pca(x_test.values, y_test.values, best_result.evaluation.indexes, uniqueLabels, mltool.model_number)
                        $("#tabs_info li[data-index='" + mltool.model_number + "'] #results_" + mltool.model_number + "").append(`
                                <div class="column is-6">
                                    <table id="knn_table_${mltool.model_number}" class="table is-bordered is-hoverable is-narrow is-size-7 display"
                                        width="100%">
                                    </table>
                                </div>
                            `);
                        new DataTable('#knn_table_' + mltool.model_number, {
                            responsive: true,
                            columns: knn_table_column_names,
                            data: knn_accuracies,
                            bDestroy: true,
                            paging: false,
                            searching: false,
                        });
                        ui.predictions_table(x_test, y_test, encoder, best_result.predictions, null, mltool.model_number)
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
                        let encoder = new LabelEncoder()
                        encoder.fit(targets)
                        let encoded_y_train = encoder.transform(y_train.values)
                        let encoded_y_test = encoder.transform(y_test.values)
                        await model.train(x_train.values, encoded_y_train)
                        let predictions = model.predict(x_test.values)
                        const classes = encoder.inverseTransform(Object.values(encoder.$labels))
                        await chart_controller.plot_confusion_matrix(window.tensorflow.tensor(predictions), window.tensorflow.tensor(encoded_y_test), encoder.inverseTransform(Object.values(encoder.$labels)), encoder.transform(classes), mltool.model_number)
                        let evaluation_result = evaluate_classification(predictions, encoded_y_test)
                        await chart_controller.draw_classification_pca(x_test.values, y_test.values, evaluation_result.indexes, uniqueLabels, mltool.model_number)
                        ui.predictions_table(x_test, y_test, encoder, predictions.null, mltool.model_number)
                        break;
                    }
                    case Settings.classification.naive_bayes.value: {
                        let model = model_factory.createModel(Settings.classification.naive_bayes, model_settings)
                        let encoder = new LabelEncoder()
                        encoder.fit(targets)
                        let encoded_y_train = encoder.transform(y_train.values)
                        let encoded_y_test = encoder.transform(y_test.values)
                        let predictions
                        if (model_settings.type === 'Gaussian') {
                            await model.train(x_train.values, encoded_y_train)
                            predictions = Array.from(model.predict(x_test.values))
                        } else {
                            predictions = Array.from(await model.train(x_train.values, encoded_y_train, x_test.values))
                        }
                        let evaluation_result = evaluate_classification(predictions, encoded_y_test)
                        const classes = encoder.inverseTransform(Object.values(encoder.$labels))
                        await chart_controller.plot_confusion_matrix(window.tensorflow.tensor(predictions), window.tensorflow.tensor(encoded_y_test), encoder.inverseTransform(Object.values(encoder.$labels)), encoder.transform(classes), mltool.model_number)
                        await chart_controller.draw_classification_pca(x_test.values, y_test.values, evaluation_result.indexes, uniqueLabels, mltool.model_number)
                        ui.predictions_table(x_test, y_test, encoder, predictions.null, mltool.model_number)
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
                        let encoder = new LabelEncoder()
                        encoder.fit(targets)
                        let encoded_y_train = encoder.transform(y_train.values)
                        let encoded_y_test = encoder.transform(y_test.values)
                        await model.train(x_train.values, encoded_y_train)
                        let predictions = await model.predict(x_test.values)
                        let evaluation_result = evaluate_classification(predictions, encoded_y_test)
                        const classes = encoder.inverseTransform(Object.values(encoder.$labels))
                        await chart_controller.plot_confusion_matrix(window.tensorflow.tensor(predictions), window.tensorflow.tensor(encoded_y_test), encoder.inverseTransform(Object.values(encoder.$labels)), encoder.transform(classes), mltool.model_number)
                        await chart_controller.draw_classification_pca(x_test.values, y_test.values, evaluation_result.indexes, uniqueLabels, mltool.model_number)
                        ui.predictions_table(x_test, y_test, encoder, predictions.null, mltool.model_number)
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
                        let predictions = []
                        probs.forEach((item => {
                            let key = Object.keys(encoder.$labels)[item.indexOf(Math.max(...item))]
                            predictions.push(parseInt(encoder.$labels[key]))
                        }));
                        let evaluation_result = evaluate_classification(predictions, y_t)
                        const classes = encoder.inverseTransform(Object.values(encoder.$labels))
                        await chart_controller.plot_confusion_matrix(window.tensorflow.tensor(predictions), window.tensorflow.tensor(y_t), classes, encoder.transform(classes), mltool.model_number)
                        await chart_controller.draw_classification_pca(x_test.values, encoder.inverseTransform(y_t), evaluation_result.indexes, uniqueLabels, mltool.model_number)
                        let metrics = []
                        for (let i = 0; i < stats.length; i++) {
                            metrics.push([
                                ...stats[i]])
                        }
                        chart_controller.probabilities_boxplot(probs, uniqueLabels, y_test.values, mltool.model_number)
                        let content = `
                        <div class="column is-6">
                            <table id="stats_table_${mltool.model_number}" class="table is-bordered is-hoverable is-narrow display is-size-7"
                                width="100%">
                            </table>
                        </div>
                        `
                        $("#tabs_info li[data-index='" + mltool.model_number + "'] #results_" + mltool.model_number + "").append(content);
                        new DataTable('#stats_table_' + mltool.model_number, {
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
                        ui.predictions_table(x_test, y_test, encoder, predictions, probs, mltool.model_number);
                        chart_controller.probablities_violin_plot(probs, classes, uniqueLabels)
                        break
                    }
                    case Settings.classification.discriminant_analysis.value: {
                        let model = model_factory.createModel(Settings.classification.discriminant_analysis, { type: model_settings.type === "linear" ? 0 : 1, priors: model_settings.priors })
                        let encoder = new LabelEncoder()
                        encoder.fit(y_train.values)
                        let y = encoder.transform(y_train.values)
                        let y_t = encoder.transform(y_test.values)
                        let predictions = await model.train(x_train.values, y, x_test.values)
                        predictions = Array.from(predictions)
                        let evaluation_result = evaluate_classification(predictions, y_t)
                        const classes = encoder.inverseTransform(Object.values(encoder.$labels))
                        await chart_controller.plot_confusion_matrix(window.tensorflow.tensor(predictions), window.tensorflow.tensor(y_t), classes, encoder.transform(classes), mltool.model_number)
                        await chart_controller.draw_classification_pca(x_test.values, encoder.inverseTransform(y_t), evaluation_result.indexes, uniqueLabels, mltool.model_number)
                        ui.predictions_table(x_test, y_test, encoder, predictions);
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

                        let predictions = await model.train_test(x_train.values, encoded_y, x_test.values)
                        const evaluation_result = evaluate_classification(predictions, encoded_y_test)
                        const classes = encoder_rf.inverseTransform(Object.values(encoder_rf.$labels))
                        await chart_controller.plot_confusion_matrix(window.tensorflow.tensor(predictions), window.tensorflow.tensor(encoded_y_test), classes, encoder_rf.transform(classes), mltool.model_number)
                        await chart_controller.draw_classification_pca(x_test.values, y_test.values, evaluation_result.indexes, uniqueLabels, mltool.model_number)

                        ui.predictions_table(x_test, y_test, encoder_rf, predictions, null, mltool.model_number);

                        break
                    }
                    default:
                        ui.show_error_message("to be implemented", "#7E191B")
                        break;
                }
            } else {
                ui.init_regression_results_tab(mltool.model_number)
                switch (model_name) {
                    case Settings.regression.linear_regression.value: {
                        let model = model_factory.createModel(Settings.regression.linear_regression, model_settings, {});
                        let summary = await model.train_test(x_train.values, y_train.values, x_test.values, y_test.values, x_train.columns)

                        let model_stats_matrix = [];
                        let cols = [...x_train.columns]
                        cols.push("intercept")
                        for (let i = 0; i < cols.length; i++) {
                            let row = [];
                            row.push(cols[i])
                            row.push(summary.get('params')[i].toFixed(2))
                            row.push(summary.get('bse')[i].toFixed(2))
                            row.push(summary.get('pvalues')[i].toFixed(2))
                            model_stats_matrix.push(row)
                        }
                        chart_controller.plot_regularization(summary.get('coefs'), summary.get('alphas'), x_train.columns, mltool.model_number)
                        new DataTable('#metrics_table_' + mltool.model_number, {
                            dom: '<"' + mltool.model_number + '">',
                            initComplete: function () {
                                $('.' + mltool.model_number).html(`R squared:${summary.get('rsquared').toFixed(2)} AIC: ${summary.get('aic').toFixed(2)} BIC: ${summary.get('bic').toFixed(2)} `);
                            },
                            responsive: true,
                            columns: [{ title: "variable" }, { title: "weight" }, { title: "std error" }, { title: "p value" }],
                            data: model_stats_matrix,
                            info: false,
                            search: false,
                            ordering: false,
                            searching: false,
                            paging: false,
                            bDestroy: true,
                        });
                        chart_controller.yhat_plot(y_test.values, summary.get('preds'), mltool.model_number)
                        ui.predictions_table_regression(x_test, y_test, summary.get('preds'), mltool.model_number)

                        break;
                    }
                    case Settings.regression.polynomial_regression.value: {
                        let model = model_factory.createModel(Settings.regression.polynomial_regression, model_settings, {});
                        let summary = await model.train_test(x_train.values, y_train.values, x_test.values, x_train.columns)


                        let model_stats_matrix = [];
                        let cols = [...x_train.columns]
                        cols.push("intercept")
                        for (let i = 0; i < cols.length; i++) {
                            let row = [];
                            row.push(cols[i])
                            row.push(summary.get('params')[i].toFixed(2))
                            row.push(summary.get('bse')[i].toFixed(2))
                            row.push(summary.get('pvalues')[i].toFixed(2))
                            model_stats_matrix.push(row)
                        }

                        new DataTable('#metrics_table_' + mltool.model_number, {
                            dom: '<"' + mltool.model_number + '">',
                            initComplete: function () {
                                $('.' + mltool.model_number).html(`R squared:${summary.get('rsquared').toFixed(2)} AIC: ${summary.get('aic').toFixed(2)} BIC: ${summary.get('bic').toFixed(2)} `);
                            },
                            responsive: true,
                            columns: [{ title: "variable" }, { title: "weight" }, { title: "std error" }, { title: "p value" }],
                            data: model_stats_matrix,
                            info: false,
                            search: false,
                            ordering: false,
                            searching: false,
                            paging: false,
                            bDestroy: true,
                        });
                        chart_controller.yhat_plot(y_test.values, summary.get('preds'), mltool.model_number)
                        ui.predictions_table_regression(x_test, y_test, summary.get('preds'), mltool.model_number)

                        break;
                    }
                    case Settings.regression.k_nearest_neighbour.value: {
                        model_settings = ui.get_model_settings();
                        let model = model_factory.createModel(Settings.regression.k_nearest_neighbour, model_settings)
                        await model.train(x_train.values, y_train.values, 5)
                        let predictions = model.predict(x_test.values);
                        ui.regression_metrics_display(y_test, predictions, mltool.model_number);
                        chart_controller.yhat_plot(y_test.values, predictions, mltool.model_number)
                        ui.predictions_table_regression(x_test, y_test, predictions, mltool.model_number)


                        break;
                    }
                    case Settings.regression.boosting.value: {
                        let model = model_factory.createModel(Settings.regression.boosting, {
                            objective: "reg:linear",
                            iterations: model_settings.iterations ?? 200
                        })
                        await model.train(x_train.values, y_train.values)
                        let predictions = await model.predict(x_test.values)
                        ui.regression_metrics_display(y_test, predictions, mltool.model_number);
                        chart_controller.yhat_plot(y_test.values, predictions, mltool.model_number)
                        ui.predictions_table_regression(x_test, y_test, predictions, mltool.model_number)

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
                        let predictions = await model.predict(x_test.values)
                        ui.regression_metrics_display(y_test, predictions, mltool.model_number);
                        chart_controller.yhat_plot(y_test.values, predictions, mltool.model_number)
                        ui.predictions_table_regression(x_test, y_test, predictions, mltool.model_number)
                        break;
                    }
                    case Settings.regression.kernel_regression.value: {
                        model_settings.types = ''
                        for (let i = 0; i < x_train.columns.length; i++) {
                            model_settings.types += 'c';
                        }
                        let model = model_factory.createModel(Settings.regression.kernel_regression, model_settings, {});
                        let predictions = await model.train_test(x_train.values, y_train.values, x_test.values, x_train.columns)
                        ui.regression_metrics_display(y_test, predictions, mltool.model_number);
                        chart_controller.yhat_plot(y_test.values, predictions, mltool.model_number)
                        ui.predictions_table_regression(x_test, y_test, predictions, mltool.model_number)

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

                        let predictions = await model.train_test(x_train.values, y_train.values, x_test.values)
                        ui.regression_metrics_display(y_test, predictions, mltool.model_number);
                        chart_controller.yhat_plot(y_test.values, predictions, mltool.model_number)
                        ui.predictions_table_regression(x_test, y_test, predictions, mltool.model_number)
                        break
                    }
                    case Settings.regression.bspline_regression.value: {
                        const model = model_factory.createModel(Settings.regression.bspline_regression, {
                            knots: model_settings.knots,
                            degree: model_settings.degree
                        });

                        let predictions = await model.train_test(x_train.values, y_train.values, x_test.values, y_test.values, x_train.columns)
                        ui.regression_metrics_display(y_test, predictions, mltool.model_number);
                        chart_controller.yhat_plot(y_test.values, predictions, mltool.model_number)
                        ui.predictions_table_regression(x_test, y_test, predictions, mltool.model_number)
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
    ui.init_tabs_events();
    Plotly.setPlotConfig({
        displaylogo: false,
        modeBarButtonsToRemove: ['resetScale2d', 'zoom2d', 'pan', 'select2d', 'resetViews', 'sendDataToCloud', 'hoverCompareCartesian', 'lasso2d', 'drawopenpath '], // Remove certain buttons from the mode bar
    });
});






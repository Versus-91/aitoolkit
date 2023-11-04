"use strict";
import { DataFrame, tensorflow, OneHotEncoder, plovt } from 'danfojs/dist/danfojs-base';
import $ from 'jquery';
import Papa from 'papaparse';
import ChartController from "./src/charts.js";
import DataLoader from "./src/data.js";
import Trainter from "./src/trainer.js";
import UI from "./src/ui.js";
import { FeatureCategories, Settings } from './feature_types.js';
import { ModelFactory } from './src/model_factory.js';

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
        chart_controller.plot_tsne(dataset.loc({ columns: numericColumns }).values, is_classification ? dataset.loc({ columns: [target] }).values : null);
    }
    if (is_classification) {

        let counts = dataset.column(target).valueCounts()
        const df = new DataFrame({
            values: counts.values,
            labels: counts.$index,
        });

        df.plot("y_pie_chart").pie({ config: { values: "values", labels: "labels" } });
    }
    await train(dataset)

}
async function train(data) {
    let dataset = data.copy()
    const target = document.getElementById("target").value;
    dataset = data_parser.perprocess_data(dataset)
    let selected_columns = ui.find_selected_columns(dataset.columns)
    let model_factory = new ModelFactory()
    selected_columns = selected_columns.filter(m => m !== target)
    const x_train = dataset.loc({ columns: selected_columns })
    const y_train = dataset.column(target)
    const x_test = x_train
    const y_test = y_train
    var modewl = null
    if (document.getElementById(target).value !== FeatureCategories.Numerical) {
        //knn
        let knn_classifier = model_factory.createModel(Settings.classification.k_nearest_neighbour)
        knn_classifier.train(x_train.values, data.column(target).values, 5)
        let y_preds = knn_classifier.evaluate(x_train.values)
        let evaluation_result = evaluate_classification(y_preds, data.column(target).values)
        return
        // is classification
        const unique_classes = [...new Set(dataset.column(target).values)]
        const is_binary_classification = unique_classes.length === 2 ? 1 : 0;
        if (is_binary_classification) {
            let binary_logistic_regression = model_factory.createModel(Settings.classification.logistic_regression)
            model = await binary_logistic_regression.train(x_train.tensor, y_train.tensor, selected_columns.length, 2)
            await binary_logistic_regression.evaluate(x_train.tensor, y_train.tensor, model, [], true)
        } else {
            let logistic_regression = model_factory.createModel(Settings.classification.logistic_regression, chart_controller)
            let encode = new OneHotEncoder()
            encode.fit(dataset[target])
            let sf_enc = encode.transform(dataset[target].values)
            let model = await logistic_regression.train(x_train.tensor, tf.tensor(sf_enc), selected_columns.length, unique_classes.length)
            await logistic_regression.evaluate(x_train.tensor, tf.tensor(sf_enc), model)
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
    console.log(y_preds);
    console.log(y_test);

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
document.getElementById("parseCVS").addEventListener("change", handleFileSelect)
document.getElementById("knn").addEventListener("click", trainer.knn_test)








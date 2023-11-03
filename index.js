"use strict";
import * as ss from 'simple-statistics'
import { DataFrame, LabelEncoder, Series, tensorflow, concat, OneHotEncoder, getDummies } from 'danfojs/dist/danfojs-base';
import $ from 'jquery';
import Papa from 'papaparse';
import ChartController from "./src/charts.js";
import DataLoader from "./src/data.js";
import Trainter from "./src/trainer.js";
import UI from "./src/ui.js";
import Classification from "./src/classification.js";
import { encode_name } from "./src/utils.js";
import { readCSV } from 'danfojs/dist/danfojs-browser/src/index.js';
import { FeatureCategories } from './feature_types.js';

window.tf = tensorflow
window.jQuery = window.$ = $
let data_parser = new DataLoader();
let ui = new UI(data_parser);
let trainer = new Trainter();
let chart_controller = new ChartController(data_parser);
const classifier = new Classification(chart_controller);


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
            // let new_col = new Series({ stats: ["count", "mean", "std", "min", "median", "max", "variance"] })
            // const headerStyle = {
            //     align: ["left", "center"],
            //     line: { width: 1, color: '#506784' },
            //     fill: { color: '#119DFF' },
            //     font: { family: "Arial", size: 14, color: "white" }
            // };
            // const cellStyle = {
            //     align: ["left", "center"],
            //     line: { color: "#506784", width: 1 },
            //     fill: { color: ['#25FEFD', 'white'] },
            //     font: { family: "Arial", size: 13, color: ["#506784"] }

            // };
            // let description = df1.describe().round(2)
            // description.print()
            // let z = concat({ dfList: [new_col, description], axis: 1 })
            // z.plot("desc").table({
            //     config: {
            //         tableHeaderStyle: headerStyle,
            //         tableCellStyle: cellStyle,
            //     }
            // })
            // console.log(df1.shape)
            // const na = df1.isNa().sum({ axis: 0 }).div(df1.isNa().count({ axis: 0 })).round(2)
            // na.plot("plot_div").table()
            // let data = [[1, 2, 3], [NaN, 5, 6], [NaN, 30, 40], [39, undefined, 78]]
            // let cols = ["A", "B", "C"]
            // let df = new DataFrame(data, { columns: cols })
            // Calculate KDE values at specific points
            // df.isNa().sum().print()
            // let missing_values = df1.isNa().sum().div(df1.isNa().count()).round(4)
            // missing_values.print()
            // sk.setBackend(tf)
            let dataset = new DataFrame(result.data)
            ui.createDatasetPropsDropdown(dataset);
            document.getElementById("train-button").onclick = async () => {
                document.getElementById("train-button").classList.add("is-loading")
                await visualize(dataset)
                document.getElementById("train-button").classList.remove("is-loading")
            }
            // const portions = data_parser.findTargetPercents(results.data, "Species");
            // ui.drawTargetPieChart(portions, Object.keys(portions).filter(m => m !== "count"), "y_pie_chart");
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
    if (numericColumns.length > 0) {
        chart_controller.plot_tsne(dataset.loc({ columns: numericColumns }).values, null);
    }

}
async function train(data) {
    let dataset = new DataFrame(data)
    const target = document.getElementById("target").value;
    dataset = data_parser.perprocess_data(dataset)
    let selected_columns = ui.find_selected_columns(dataset.columns)
    let model = null
    selected_columns = selected_columns.filter(m => m !== target)
    const x_train = dataset.loc({ columns: selected_columns })
    const y_train = dataset.column(target)
    const x_test = x_train
    const y_test = y_train
    if (document.getElementById(target).value !== FeatureCategories.Numerical) {
        // is classification
        const unique_classes = [...new Set(dataset.column(target).values)]
        const is_binary_classification = unique_classes.length === 2 ? 1 : 0;
        if (is_binary_classification) {
            let model = await classifier.trainLogisticRegression(x_train.tensor, y_train.tensor, selected_columns.length, 2)
            await classifier.evaluate(x_train.tensor, y_train.tensor, model, [], true)
        } else {
            let encode = new OneHotEncoder()
            encode.fit(dataset[target])
            let sf_enc = encode.transform(dataset[target].values)
            let model = await classifier.trainLogisticRegression(x_train.tensor, tf.tensor(sf_enc), selected_columns.length, unique_classes.length)
            await classifier.evaluate(x_train.tensor, tf.tensor(sf_enc), model)
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

document.getElementById("parseCVS").addEventListener("change", handleFileSelect)
document.getElementById("knn").addEventListener("click", trainer.knn_test)
document.getElementById("tsne-draw").addEventListener("click", function (params) {
    console.log("clicked");
    let txt = document.getElementById("tsne").value
    var d = ",";
    var lines = txt.split("\n");
    var raw_data = [];
    var dlen = -1;
    let dataok = true;
    
    for (var i = 0; i < lines.length; i++) {
        var row = lines[i];
        if (! /\S/.test(row)) {
            // row is empty and only has whitespace
            continue;
        }
        var cells = row.split(d);
        var data_point = [];
        for (var j = 0; j < cells.length; j++) {
            if (cells[j].length !== 0) {
                data_point.push(parseFloat(cells[j]));
            }
        }
        var dl = data_point.length;
        if (i === 0) { dlen = dl; }
        if (dlen !== dl) {
            // TROUBLE. Not all same length.
            console.log('TROUBLE: row ' + i + ' has bad length ' + dlen);
            dlen = dl; // hmmm... 
            dataok = false;
        }
        raw_data.push(data_point);
    }
    chart_controller.plot_tsne(raw_data, null)
})








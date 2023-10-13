"use strict";
import DataLoader from "./data.js";
import {FeatureCategories} from "./feature_types.js";
let data_parser = new DataLoader();
function createDatasetPropsDropdown(items) {
    let rowMetadata = data_parser.findDataTypes(items);
    let header = "";
    for (const key in rowMetadata) {
        let options = ""
        const lastProperty = Object.keys(items[0])[Object.keys(items[0]).length - 1];
        $('#props').append(`
            <h4>${key.replace(/([A-Z])/g, ' $1').trim()} - ${key === lastProperty ? "Output" : "Input"}</h4>
            <div class="select mb-1">
                <select id="${key}">
                    <option value="1">Numerical</option>
                    <option value="2">Categorical</option>
                    <option value="3">Ordinal</option>
                </select>
            </div>
        `);
        if (rowMetadata[key] === FeatureCategories.Numerical) {
            $('#' + key).val(1)
        } else if (rowMetadata[key] === FeatureCategories.Categorical) {
            $('#' + key).val(2)
        }
    }
}

function handleFileSelect(evt) {
    var target = evt.target || evt.srcElement;
    if (target.value.length == 0) {
        return;
    }
    var file = evt.target.files[0];
    Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        complete: function (results) {
            createDatasetPropsDropdown(results.data);
            renderDatasetStats(results.data);
            renderChart("chart", results.data, "PetalLengthCm", {
                title: "",
                xLabel: "Species"
            });
        }
    });
}

async function trainNewModel() {
    this.linearmodel = tf.sequential();
    this.linearmodel.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    this.linearmodel.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

    const xs = tf.tensor1d([1, 2]);
    const ys = tf.tensor1d([2, 4]);
    await this.linearmodel.fit(xs, ys);
    console.log('training is complete');
}
function renderChart(container, data, column, config) {
    const columnData = data.map(r => r[column]);

    const columnTrace = {
        name: column,
        x: columnData,
        type: "histogram",
        opacity: 0.7,
        marker: {
            color: "dodgerblue"
        }
    };
    Plotly.newPlot(container, [columnTrace], {
        xaxis: {
            title: config.xLabel,
            range: config.range
        },
        yaxis: { title: "Count" },
        title: config.title
    });
};

function renderDatasetStats(data) {
    var header = "";
    var tbody = "";
    const fileds = ["Metric", "Min", "Max", "Median", "Mean", "Standard deviation", "p-value"]
    for (var p in fileds) {
        header += "<th>" + fileds[p] + "</th>";
    }
    const invalidColumns = ["Id"];
    let columnDataTypes = data_parser.findDataTypes(data)
    console.log(columnDataTypes);
    for (const key in columnDataTypes) {
        if (columnDataTypes[key] === FeatureCategories.Categorical) {
            invalidColumns.push(key)
        }
    }
    console.log(invalidColumns);
    for (const key in data[0]) {
        if (!invalidColumns.includes(key)) {
            let row = "";
            const formattedData = data.map(row => {
                return row[key]
            }).filter(function (item) {
                return typeof item === "number"
            });

            const min = Math.min(...formattedData)
            const max = Math.max(...formattedData)
            row += "<td>" + key + "</td>";
            row += "<td>" + min + "</td>";
            row += "<td>" + max + "</td>";
            row += "<td>" + ss.median(formattedData) + "</td>";
            row += "<td>" + ss.mean(formattedData) + "</td>";
            row += "<td>" + ss.standardDeviation(formattedData) + "</td>";
            row += "<td>" + "NA" + "</td>";
            tbody += "<tr>" + row + "</tr>";
        }
    }

    //build a table
    document.querySelector("output").innerHTML =
        '<table class="table is-bordered"><thead>' +
        header +
        "</thead><tbody>" +
        tbody +
        "</tbody></table>"
        ;
}
function linear_test(data) {
    // Inputs
    const xs = tf.tensor([-1, 0, 1, 2, 3, 4]);
    // Answers we want from inputs
    const ys = tf.tensor([-4, -2, 0, 2, 4, 6]);
    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: 1,
        units: 1
    }));
    model.compile({
        optimizer: "sgd",
        loss: "meanSquaredError"
    });

    let log = model.summary();
    // Train
    model.fit(xs, ys, { epochs: 300 }).then(history => {
        const inputs = [10, 10, 15]
        const inputTensor = tf.tensor(inputs);
        const answer = model.predict(inputTensor);
        const answers = answer.dataSync();
        answers.forEach((element, i) => {
            console.log(inputs[i] + ` results in ${Math.round(answers[i])}`);
        });
        const labels = tf.tensor1d(inputs);
        const predictions = tf.tensor1d(answers);
        tfvis.metrics.confusionMatrix(labels, predictions).then((res) => {
            const container = document.getElementById("confusion-matrix")
            tfvis.render.confusionMatrix(container, {
                values: res,
                tickLabels: ["Healthy", "Diabetic"],
            })
            console.log(JSON.stringify(res, null, 2))
        });
        tf.dispose([xs, ys, model, answer, inputTensor]);
    });
}


document.getElementById("parseCVS").addEventListener("change", handleFileSelect)
document.getElementById("test").addEventListener("click", linear_test)


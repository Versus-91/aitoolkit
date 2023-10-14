"use strict";
import DataLoader from "./data.js";
import { FeatureCategories } from "./feature_types.js";
import  Trainter  from "./trainer.js";
let data_parser = new DataLoader();
let trainer = new Trainter();
function createDatasetPropsDropdown(items) {
    let rowMetadata = data_parser.findDataTypes(items);
    let header = "";
    for (const key in rowMetadata) {
        let options = ""
        const lastProperty = Object.keys(items[0])[Object.keys(items[0]).length - 1];
        $('#props').append(`
        <div class="column is-4">
            <h4>${key.replace(/([A-Z])/g, ' $1').trim()} - ${key === lastProperty ? "Output" : "Input"}</h4>
            <div class="select mb-1">
                <select id="${key}">
                    <option value="1">Numerical</option>
                    <option value="2">Nominal</option>
                    <option value="3">Ordinal</option>
                </select>
            </div>
            <label class="checkbox my-2">
                <input type="checkbox">
                Ignore
            </label>
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
        complete: async function (results) {
            createDatasetPropsDropdown(results.data);
            renderDatasetStats(results.data);
            renderChart("chart", results.data, "PetalLengthCm", {
                title: "",
                xLabel: "Species"
            });
            const features = ["Glucose"];

            const [trainDs, validDs, xTest, yTest] = data_parser.createDataSets(
                results.data,
                features,
                0.1,
                16
            );
            const model = await trainer.trainLogisticRegression(
                features.length,
                trainDs,
                validDs
            );
        }
    });
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

document.getElementById("parseCVS").addEventListener("change", handleFileSelect)
document.getElementById("test").addEventListener("click", linear_test)


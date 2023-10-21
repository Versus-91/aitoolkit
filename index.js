"use strict";
import DataLoader from "./data.js";
import Trainter from "./trainer.js";
import UI from "./ui.js";
let data_parser = new DataLoader();
let ui = new UI(data_parser);
let trainer = new Trainter();

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
            ui.createDatasetPropsDropdown(results.data);
            ui.renderDatasetStats(results.data);
            data_parser.findMissinValues(results.data)
            const portions = data_parser.findTargetPercents(results.data, "Species")
            ui.drawTargetPieChart(portions, Object.keys(portions).filter(m => m !== "count"), "y_pie_chart")
            // renderChart("chart", results.data, "PetalLengthCm", {
            //     title: "",
            //     xLabel: "Species"
            // });
            // const features = ["Glucose"];

            // const [trainDs, validDs, xTest, yTest] = data_parser.createDataSets(
            //     results.data,
            //     features,
            //     0.1,
            //     16
            // );
            // const model = await trainer.trainLogisticRegression(
            //     features.length,
            //     trainDs,
            //     validDs
            // );
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


document.getElementById("parseCVS").addEventListener("change", handleFileSelect)


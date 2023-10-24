"use strict";
import Papa from 'papaparse';
import ChartController from "./src/charts.js";
import DataLoader from "./src/data.js";
import Trainter from "./src/trainer.js";
import UI from "./src/ui.js";
var jquery = require("jquery");
window.$ = window.jQuery = jquery;
let data_parser = new DataLoader();
let ui = new UI(data_parser);
let trainer = new Trainter();
let chart = new ChartController(data_parser)

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
            const features = ["Glucose"];
            const [trainDs, validDs, xTest, yTest] = data_parser.createDataSets(
                results.data,
                features,
                0.1,
                16
            );

            data_parser.findMissinValues(results.data)
            const portions = data_parser.findTargetPercents(results.data, "Species")
            ui.drawTargetPieChart(portions, Object.keys(portions).filter(m => m !== "count"), "y_pie_chart")
        }
    });
}
document.getElementById("parseCVS").addEventListener("change", handleFileSelect)

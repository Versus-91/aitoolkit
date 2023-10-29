import Plotly from 'plotly.js-dist';
import * as ss from 'simple-statistics';

import { FeatureCategories, Settings } from "../feature_types.js";
export default class UI {
    constructor(parser) {
        this.data_parser = parser
    }
    drawTargetPieChart(values, lables, containerId) {
        var data = [{
            values: values,
            labels: lables,
            type: 'pie'
        }];
        var layout = {
            height: 400,
            width: 500
        };

        Plotly.newPlot(containerId, data, layout);
    }
    renderChart(container, data, column, config) {
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
    createDatasetPropsDropdown(items) {
        try {
            let rowMetadata = this.data_parser.findDataTypes(items);
            let header = "";
            const lastProperty = Object.keys(items[0])[Object.keys(items[0]).length - 1];
            for (let key in rowMetadata) {
                let options = ""
                key = key.replace(/\s/g, '').replace(/[^\w-]/g, '_');
                $('#props').append(`
                <div class="column is-4">
                    <h4>${this.insertSpaces(key)} - ${key === lastProperty ? "Output" : "Input"}</h4>
                    <div class="select mb-1">
                        <select id="${key}">
                            <option value="1">Numerical</option>
                            <option value="2">Nominal</option>
                            <option value="3">Ordinal</option>
                        </select>
                    </div>
                    <label class="checkbox my-2">
                        <input id="${key + "-checkbox"}" type="checkbox">
                        Ignore
                    </label>
                </div>
                `);
                const id = key
                if (rowMetadata[key] === FeatureCategories.Numerical) {
                    $('#' + id).val(1)
                } else if (rowMetadata[key] === FeatureCategories.Categorical) {
                    $('#' + id).val(2)
                }
            }

            if (rowMetadata[lastProperty] === FeatureCategories.Numerical) {
                $('#props').append(this.createAlgorithmsSelect(1));
            } else if (rowMetadata[lastProperty] === FeatureCategories.Categorical) {
                $('#props').append(this.createAlgorithmsSelect(2));
            }
            $(document).on('change', '#' + lastProperty + '-y', function (e) {
                $("#algorithm").remove();
                $("#props").append(this.createAlgorithmsSelect(e.target.value == 1 ? 1 : 2))
            });
            $('#props').append(`
            <div class="column is-4">
                <h4>Imputation</h4>
                <div class="select mb-1">
                    <select id="imputation">
                        <option value="1">Default</option>
                        <option value="2">Linear regression</option>
                        <option value="3">random forest</option>
                    </select>
                </div>
            </div>
            `)
            $('#props').append(`
            <div class="column is-4">
                <h4>standardize</h4>
                <div class="select mb-1">
                    <select id="normalization">
                        <option value="1">No</option>
                        <option value="2">Scale</option>
                        <option value="3">Normal</option>
                    </select>
                </div>
            </div>
            `)
            $('#props').append(`
            <div class="column is-4">
                <h4>Cross Validation</h4>
                <div class="select mb-1">
                    <select id="cross_validation">
                        <option value="1">70 % training - 30 % test</option>
                        <option value="2">No</option>
                        <option value="3">K-fold</option>
                    </select>
                </div>
            </div>
            `)
            $('#props').append(this.createTargetDropdown(rowMetadata))
            $('#target').val(Object.keys(rowMetadata)[Object.keys(rowMetadata).length - 1])
            $('#props').append(`<div class="column is-4"><button class="button" id="train-button">train</button></div>`);
        } catch (error) {
            console.log(error);
        }

        // $('#kde_select').append(this.createFeaturesDropdown(rowMetadata))
    }
    createAlgorithmsSelect(category) {
        let result = '<div id="algorithm" class="column is-4"><h4>Algorithm</h4><div class="select mb-1"> <select class="select">'
        const lable = category == 1 ? "regression" : "classification"
        for (const key in Settings[lable]) {
            if (Settings.hasOwnProperty.call(Settings[lable], key)) {
                const item = Settings[lable][key];
                result += `<option value="${item.value}">${item.lable}</option>`
            }
        }
        result += '</select></div></div>'
        return result
    }
    createTargetDropdown(items) {
        let result = '<div  class="column is-4"><h4>Target</h4><div class="select mb-1"> <select class="select" id="target">'
        for (const key in items) {
            result += `<option value="${key}">${key}</option>`
        }
        result += '</select></div></div>'
        return result
    }
    createFeaturesDropdown(items) {
        let result = '<div  class="column is-4"><h4>Target</h4><div class="select mb-1"> <select class="select" id="kde_feature">'
        for (const key in items) {
            result += `<option value="${key}">${key}</option>`
        }
        result += '</select></div></div>'
        return result
    }
    insertSpaces(string) {
        string = string.replace(/([a-z])([A-Z])/g, '$1 $2');
        string = string.replace(/([A-Z])([A-Z][a-z])/g, '$1 $2')
        return string;
    }
    renderDatasetStats(data) {
        var header = "";
        var tbody = "";
        const fileds = ["Metric", "Min", "Max", "Median", "Mean", "Standard deviation", "p-value"]
        for (var p in fileds) {
            header += "<th>" + fileds[p] + "</th>";
        }
        const invalidColumns = ["Id"];
        let columnDataTypes = this.data_parser.findDataTypes(data)
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
                row += "<td>" + ss.median(formattedData).toFixed(2) + "</td>";
                row += "<td>" + ss.mean(formattedData).toFixed(2) + "</td>";
                row += "<td>" + ss.standardDeviation(formattedData).toFixed(2) + "</td>";
                row += "<td>" + "NA" + "</td>";
                tbody += "<tr>" + row + "</tr>";
            }
        }

        //build a table
        document.getElementById("output").innerHTML =
            '<table class="table is-bordered is-striped is-narrow is-hoverable"><thead>' +
            header +
            "</thead><tbody>" +
            tbody +
            "</tbody></table>"
            ;
    }
}
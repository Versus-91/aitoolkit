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
            $('#props').empty()
            const default_target = items.columns[items.columns.length - 1]
            items.columns.forEach(column => {
                let key = column.replace(/\s/g, '').replace(/[^\w-]/g, '_');
                $('#props').append(`
                <div class="column is-4">
                    <h4>${this.insertSpaces(key)} - ${key === default_target ? "Output" : "Input"}</h4>
                    <div class="select mb-1">
                        <select id="${column}">
                            <option value="${FeatureCategories.Numerical}">Numerical</option>
                            <option value="${FeatureCategories.Nominal}">Nominal</option>
                            <option value="${FeatureCategories.Ordinal}">Ordinal</option>
                        </select>
                    </div>
                    <label class="checkbox my-2">
                        <input id="${column + "-checkbox"}" type="checkbox" checked>
                        Ignore
                    </label>
                </div>
                `);
                const id = column
                if (items.column(column).dtype !== 'string') {
                    $('#' + id).val(FeatureCategories.Numerical)
                } else {
                    $('#' + id).val(FeatureCategories.Nominal)
                }
            });


            if (items.column(default_target).dtype !== 'string') {
                $('#props').append(this.createAlgorithmsSelect(1));
            } else {
                $('#props').append(this.createAlgorithmsSelect(2));
            }
            $(document).on('change', '#' + default_target + '-y', function (e) {
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
            $('#props').append(this.createTargetDropdown(items))
            $('#target').val(default_target)
            $('#props').append(`<div class="column is-4"><button class="button is-primary mt-5" id="train-button">train</button></div>`);
        } catch (error) {
            console.log(error);
        }

        // $('#kde_select').append(this.createFeaturesDropdown(rowMetadata))
    }
    createAlgorithmsSelect(category) {
        let result = '<div id="algorithm" class="column is-4"><h4>Algorithm</h4><div class="select mb-1"> <select id="model_name" class="select">'
        const lable = category == 1 ? "regression" : "classification"
        for (const key in Settings[lable]) {
            if (Settings.hasOwnProperty.call(Settings[lable], key)) {
                const item = Settings[lable][key];
                result += `<option value="${item.lable}">${item.lable}</option>`
            }
        }
        result += '</select></div></div>'
        return result
    }
    find_selected_columns(columns) {
        const selected_columns = []
        columns.forEach(column => {
            if (!document.getElementById(column + '-checkbox').checked)
                selected_columns.push(column)
        });
        return selected_columns
    }
    createTargetDropdown(items) {
        let result = '<div  class="column is-4"><h4>Target</h4><div class="select mb-1"> <select class="select" id="target">'
        items.columns.forEach(column => {
            let key = column.replace(/\s/g, '').replace(/[^\w-]/g, '_');
            result += `<option value="${key}">${key}</option>`

        });
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
        //build numerical feature table table
        document.getElementById("stats").style.display = "block"
        var header = "";
        var tbody = "";
        const fileds = ["#", "Min", "Max", "Median", "Mean", "Standard deviation", "p-value"]
        for (var p in fileds) {
            header += "<th>" + fileds[p] + "</th>";
        }

        data.columns.forEach(column => {
            const key = column.replace(/\s/g, '').replace(/[^\w-]/g, '_');
            const type = document.getElementById(key).value
            if (type === FeatureCategories.Numerical) {
                let row = "";
                row += "<td>" + column + "</td>";
                row += "<td>" + data.column(column).min() + "</td>";
                row += "<td>" + data.column(column).max() + "</td>";
                row += "<td>" + data.column(column).median().toFixed(2) + "</td>";
                row += "<td>" + data.column(column).mean().toFixed(2) + "</td>";
                row += "<td>" + data.column(column).std().toFixed(2) + "</td>";
                row += "<td>" + "NA" + "</td>";
                tbody += "<tr>" + row + "</tr>";
            }
        });
        document.getElementById("output").innerHTML =
            '<table class="table is-bordered is-striped is-narrow is-hoverable"><thead>' +
            header +
            "</thead><tbody>" +
            tbody +
            "</tbody></table>"
            ;
        //build categorical feature table table
        var header_categorical = "";
        var tbody_categorical = "";
        const fileds_categorical = ["#", "Mode", "Percentage"]
        for (var p in fileds_categorical) {
            header_categorical += "<th>" + fileds_categorical[p] + "</th>";
        }

        data.columns.forEach(column => {
            const key = column.replace(/\s/g, '').replace(/[^\w-]/g, '_');
            const type = document.getElementById(key).value
            if (type !== FeatureCategories.Numerical) {
                const category_info = this.data_parser.getCategoricalMode(data.column(key).values)
                let row = "";
                row += "<td>" + column + "</td>";
                row += "<td>" + category_info['mode'] + "</td>";
                row += "<td>" + ((category_info[category_info['mode']] / category_info['total']) * 100).toFixed(2) + "</td>";
                tbody_categorical += "<tr>" + row + "</tr>";
            }
        });
        document.getElementById("categorical_features").innerHTML =
            '<table class="table is-bordered is-striped is-narrow is-hoverable"><thead>' +
            header_categorical +
            "</thead><tbody>" +
            tbody_categorical +
            "</tbody></table>"
            ;
    }

}
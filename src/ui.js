import Plotly from 'plotly.js-dist';
import * as ss from 'simple-statistics';
import Bulma from '@vizuaalog/bulmajs';

import { FeatureCategories, Settings } from "../feature_types.js";
export default class UI {
    constructor(parser) {
        this.data_parser = parser
    }
    drawTargetPieChart(values, labels, containerId) {
        var data = [{
            values: values,
            labels: labels,
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
            const myClass = this
            $('#props').empty()
            $('#features').empty()
            $('#props').append(`<div class="column is-6"><button id ="feature_selection_modal" class="button is-warning" >Select Features</button></div>`)
            document.querySelector('#feature_selection_modal').addEventListener('click', function (e) {
                var modalTwo = Bulma('#features_modal').modal();
                modalTwo.open();
            });
            document.querySelector('#close_modal').addEventListener('click', function (e) {
                var modalTwo = Bulma('#features_modal').modal();
                modalTwo.close();
            });
            const default_target = items.columns[items.columns.length - 1]
            items.columns.forEach(column => {
                let key = column.replace(/\s/g, '').replace(/[^\w-]/g, '_');
                $('#features').append(`
                <div class="column is-6">
                    <h4>${column}  ${key === default_target ? "(default target)" : ""}</h4>
                    <div class="select mb-1">
                        <select id="${key}">
                            <option value="${FeatureCategories.Numerical}">Numerical</option>
                            <option value="${FeatureCategories.Nominal}">Nominal</option>
                            <option value="${FeatureCategories.Ordinal}">Ordinal</option>
                        </select>
                    </div>
                    <label class="checkbox my-2">
                        <input id="${key + "-checkbox"}" type="checkbox" checked>
                        Ignore
                    </label>
                </div>
                `);
                $('#' + key).on('change', function (e) {
                    const type = e.target.value
                    $('#algorithm').empty()
                    if (type === 'Numerical') {
                        $('#algorithm').append(myClass.updateAlgorithmsSelect(1));
                    } else {
                        $('#algorithm').append(myClass.updateAlgorithmsSelect(2));
                    }
                });
                const id = column
                if (items.column(column).dtype !== 'string') {
                    $('#' + key).val(FeatureCategories.Numerical)
                } else {
                    $('#' + key).val(FeatureCategories.Nominal)
                }
            });

            $('#props').append(this.createTargetDropdown(items))

            if (items.column(default_target).dtype !== 'string') {
                $('#props').append(this.createAlgorithmsSelect(1));
            } else {
                $('#props').append(this.createAlgorithmsSelect(2));
            }
            $("#props").append(`
            <div class="column is-2">
            <button class="button is-success" id="config_modal_button">
            <span class="icon is-small">
            <i class="fas fa-cog"></i>
            </span>
            </button>
            </div>
            <div class="column is-12" id="settings" style="display:none">
            </div>`)
            $(document).on('change', '#' + default_target, function (e) {
                $("#algorithm").empty();
                $("#algorithm").append(myClass.updateAlgorithmsSelect(e.target.value == 1 ? 1 : 2))
            });
            $("#model_options").empty();
            $('#algorithm').on('change', function () {
                $("#model_options").empty();
            });
            document.querySelector('#config_modal_button').addEventListener('click', function (e) {
                let model_name = document.getElementById('model_name').value;
                model_name = model_name.replace(/\s+/g, '_').toLowerCase();
                var model = Settings.classification[model_name];
                var options_modal_content = document.getElementById("settings");
                if (window.getComputedStyle(options_modal_content).display !== "none") {
                    options_modal_content.style.display = "none"
                    return
                }
                options_modal_content.innerHTML = ""
                for (const key in model.options) {
                    options_modal_content.style.display = "block"
                    if (Object.hasOwnProperty.call(model.options, key)) {
                        const element = model.options[key]["type"]
                        if (element === "number") {
                            $('#settings').append(`
                            <div class="column is-12">
                                <div class="field is-horizontal">
                                    <div class="field-label is-small">
                                    <label class="label">${key}</label>
                                    </div>
                                    <div class="field-body">
                                    <div class="control">
                                        <input id="${key + "_" + model_name}" class="input is-small" type="number">
                                    </div>
                                    </div>
                                </div>
                            </div>
                            `)
                            document.getElementById(key + "_" + model_name).value = model.options[key]["default"]
                        } else if (element === "select") {
                            let result = ""
                            let options = model.options[key]["values"]
                            result = `
                            <div class="column is-12">
                                <div class="field is-horizontal">
                                    <div class="field-label is-small">
                                       <label class="label mr-1">${key}</label>
                                    </div>
                                    <div class="field-body">
                                        <div class="select is-small">
                                            <select id="${key + "_" + model_name}">
                                    </div>
                            `
                            for (let i = 0; i < options.length; i++) {
                                result += `<option>${options[i]}</option>`
                            }
                            result += "</select></div></div></div>"
                            $('#settings').append(result)
                        }
                    }
                }
            });
            $('#props').append(`
            <div class="column is-12">
                <div class="label">Imputation</div>
                <div class="select mb-1">
                    <select id="imputation">
                        <option value="1">Delete rows</option>
                        <option value="2">Mean and Mode</option>
                        <option value="3">Linear regression</option>
                        <option value="4">random forest</option>
                    </select>
                </div>
            </div>
            `)
            $('#props').append(`
            <div class="column is-12">
                <div class="label">Data Transformation</div>
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
            <div class="column is-12">
                <div class="label">Cross Validation</div>
                <div class="select mb-1">
                    <select id="cross_validation">
                        <option value="1">70 % training - 30 % test</option>
                        <option value="2">No</option>
                        <option value="3">K-fold</option>
                    </select>
                </div>
            </div>
            `)
            $('#target').val(default_target)
            $('#props').append(`<div class="column is-6"><button class="button is-primary mt-2" id="visualize">EDA</button></div>`);
            $('#props').append(`<div class="column is-6"><button class="button is-info mt-2" id="train-button">train</button></div>`);

            //modle options
            $('#algorithm').on('change', function (e) {
                const model_type = items.column(default_target).dtype !== 'string' ? 1 : 2;
                const label = model_type == 1 ? "regression" : "classification"
                for (const key in Settings[label]) {
                    if (Settings.hasOwnProperty.call(Settings[label], key)) {
                        const item = Settings[label][key];
                        if (item.label === e.target.value) {
                            console.log(item?.options);

                        }
                    }
                }
            });
            $('#target').on('change', function (e) {
                const type = document.getElementById(e.target.value).value
                $('#algorithm').empty()
                if (type === 'Numerical') {
                    $('#algorithm').append(myClass.updateAlgorithmsSelect(1));
                } else {
                    $('#algorithm').append(myClass.updateAlgorithmsSelect(2));
                }
            });
            $('#target').on('change', function (e) {
                const type = document.getElementById(e.target.value).value
                $('#algorithm').empty()
                if (type === 'Numerical') {
                    $('#algorithm').append(myClass.updateAlgorithmsSelect(1));
                } else {
                    $('#algorithm').append(myClass.updateAlgorithmsSelect(2));
                }
            });

        } catch (error) {
            throw error
        }

        // $('#kde_select').append(this.createFeaturesDropdown(rowMetadata))
    }
    get_settings() {

    }
    createAlgorithmsSelect(category) {
        let result = '<div id="algorithm" class="column is-10"><div class="select mb-1"> <select id="model_name" class="select">'
        const label = category == 1 ? "regression" : "classification"
        for (const key in Settings[label]) {
            if (Settings.hasOwnProperty.call(Settings[label], key)) {
                const item = Settings[label][key];
                result += `<option value="${item.label}">${item.label}</option>`
            }
        }
        result += '</select></div></div>'
        return result
    }
    updateAlgorithmsSelect(category) {
        let result = '<div class="select mb-1"> <select id="model_name" class="select">'
        const label = category == 1 ? "regression" : "classification"
        for (const key in Settings[label]) {
            if (Settings.hasOwnProperty.call(Settings[label], key)) {
                const item = Settings[label][key];
                result += `<option value="${item.label}">${item.label}</option>`
            }
        }
        result += '</select></div>'
        return result
    }

    find_selected_columns(columns, get_all = false) {
        const selected_columns = []
        columns.forEach(column => {
            let key = column.replace(/\s/g, '').replace(/[^\w-]/g, '_');
            if (!document.getElementById(key + '-checkbox').checked || get_all)
                selected_columns.push(column)
        });
        return selected_columns
    }
    find_selected_columns_types(columns) {
        const target = document.getElementById("target").value;
        columns = columns.filter(column => column !== target)
        const column_types = []
        columns.forEach(column => {
            let key = column.replace(/\s/g, '').replace(/[^\w-]/g, '_');
            column_types.push({
                name: column,
                type: document.getElementById(key).value
            })
        });
        return column_types
    }
    createTargetDropdown(items) {
        let result = '<div  class="column is-12"><div class="label">Target</div><div class="select mb-1"> <select class="select" id="target">'
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
        const fileds = ["#", "Min", "Max", "Median", "Mean", "Standard deviation"]
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
        const fileds_categorical = ["#", "Shape", "Mode", "Percentage"]
        for (var p in fileds_categorical) {
            header_categorical += "<th>" + fileds_categorical[p] + "</th>";
        }

        data.columns.forEach(column => {
            const key = column.replace(/\s/g, '').replace(/[^\w-]/g, '_');
            const type = document.getElementById(key).value
            if (type !== FeatureCategories.Numerical) {
                const shape = [...new Set(data.column(key).values)];
                const category_info = this.data_parser.getCategoricalMode(data.column(key).values)
                let row = "";
                row += "<td>" + column + "</td>";
                row += "<td>" + shape.length + "</td>";
                row += "<td>" + category_info['mode'] + "</td>";
                row += "<td>" + ((category_info[category_info['mode']] / category_info['total'])).toFixed(2) + "</td>";
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
    reset(ids, tables) {
        tables.forEach(table => {
            if ($.fn.DataTable.isDataTable('#' + table)) {
                $('#' + table).DataTable().destroy();
                document.getElementById(table).innerHTML = ""
            }

        });

        ids.forEach(id => {
            document.getElementById(id).innerHTML = ""
        });
    }
    init_upload_button(upoad_handler) {
        $('#upload').append(`<div class="column is-12">
        <div class="file is-info">
            <label class="file-label">
                <input class="file-input is-info" id="parseCVS" type="file" name="resume">
                <span class="file-cta">
                    <span class="file-icon">
                        <i class="fas fa-upload"></i>
                    </span>
                    <span class="file-label">
                        Upload file
                    </span>
                </span>
            </label>
        </div>
        <p class="help is-danger">CSV file is supported for now</p>
    </div>`)
        document.getElementById("parseCVS").addEventListener("change", upoad_handler)
    }
    start_loading() {
        document.getElementById("train-button").classList.add("is-loading")
    }
    stop_loading() {
        document.getElementById("train-button").classList.remove("is-loading")
    }
    show_error_message(message = "Something went wrong", background = "#7E191B", duration = 3000) {
        Toastify({
            text: message,
            duration: duration,
            close: true,
            style: {
                background: background,
            },
        }).showToast();
    }
}
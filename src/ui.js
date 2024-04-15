import Plotly from 'plotly.js-dist';
import Bulma from '@vizuaalog/bulmajs';
import { MinMaxScaler, DataFrame } from 'danfojs/dist/danfojs-base';

import { FeatureCategories, Settings } from "../feature_types.js";
export default class UI {
    constructor(parser, chart_controller) {
        this.data_parser = parser
        this.chart_controller = chart_controller
    }

    get_model_settings() {
        let model_settings = {};
        let model_name = document.getElementById('model_name').value;
        const target = document.getElementById("target").value;
        let is_classification = document.getElementById(target).value !== FeatureCategories.Numerical;
        if (is_classification) {
            for (const model in Settings.classification) {
                if (Settings.classification[model].label === model_name) {
                    model_name = model
                }
            }
        } else {
            for (const model in Settings.regression) {
                if (Settings.regression[model].label === model_name) {
                    model_name = model
                }
            }
        }
        let model = Settings.classification[model_name];
        for (const option in model?.options) {
            let option_element = document.getElementById(option + "_" + model_name);
            if (model.options[option].type === "select") {
                let option_value = document.getElementById(option + "_" + model_name)?.value;
                model_settings[option] = option_value ?? model.options[option].default
            } else {
                if (model.options[option].type === "number") {
                    let option_value = document.getElementById(option + "_" + model_name)?.value;
                    model_settings[option] = !option_value ? model.options[option].default : parseFloat(option_value)
                } else {
                    let option_value = document.getElementById(option + "_" + model_name)?.value;
                    model_settings[option] = option_value ?? model.options[option].default
                }

            }
        }

        return model_settings
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
    scale_data(dataset, column, normalization_type) {
        switch (normalization_type) {
            case "1":
                let scaler = new MinMaxScaler()
                scaler.fit(dataset[column])
                dataset.addColumn(column, scaler.transform(dataset[column]), { inplace: true })
                break;
            case "2":

                break;
            default:
                break;
        }

    }
    createDatasetPropsDropdown(items) {
        try {
            const myClass = this
            //feature selection
            $('#props').empty()
            $('#features-selection').empty()
            $('#features').empty()
            $('#props').append(this.createTargetDropdown(items))
            $('#features-selection').append(`<div class="column is-6"><button id ="feature_selection_modal" class="button is-warning is-small" >Select Features</button></div>`)
            document.querySelector('#feature_selection_modal').addEventListener('click', function (e) {
                var features_dropdown = document.getElementById("config_modal")
                var props = document.getElementById("props")
                var props_feature_selection_button = document.getElementById("feature_selection_modal")

                if (window.getComputedStyle(features_dropdown).display !== "none") {
                    props_feature_selection_button.innerText = "Select Features"
                    features_dropdown.style.display = "none"
                    props.style.display = "block"
                    const redraw_plots_data_analysis = new CustomEvent("update_graphs");
                    props_feature_selection_button.dispatchEvent(redraw_plots_data_analysis)
                    return
                }
                props_feature_selection_button.innerText = "Config model"
                features_dropdown.style.display = "block"
                props.style.display = "none"

            });
            $("#features-selection").append(`

                <div id="config_modal" style="display:none;overflow-y:scroll;max-height: 600px;">
                    <table class="table is-narrow is-size-7" 
                    <thead>
                    <tr>
                      <th><input id="select_all" value="1" name="selectall" type="checkbox" checked="checked" /></th>
                      <th>Name</th>
                      <th>Scale</th>
                    </tr>
                  </thead>
                  <tbody id="features">
                  </tbody>
                    </table>
                </div>
            </div>
            `)
            document.querySelector('#select_all').addEventListener('click', function (e) {
                if ($("#select_all").prop('checked')) {
                    $('.features-filter').prop('checked', true);
                } else {
                    $('.features-filter').prop('checked', false);
                }
            });
            const default_target = items.columns[items.columns.length - 1]
            items.columns.forEach(column => {
                let key = column.replace(/\s/g, '').replace(/[^\w-]/g, '_');
                $('#features').append(`
                <tr>
                    <td>
                    <label class="checkbox my-2">
                    <input id="${key + "-checkbox"}" type="checkbox" value="1" class="features-filter" checked="checked">
                    </label>
                    </td>
                    <td class="mt-1">
                    ${column}
                    </td>
                    <td>
                    <div class="select is-small mb-1">
                        <select id="${key}">
                            <option value="${FeatureCategories.Numerical}">Numerical</option>
                            <option value="${FeatureCategories.Nominal}">Nominal</option>
                            <option value="${FeatureCategories.Ordinal}">Ordinal</option>
                        </select>
                    </div>
                    </td>
                </tr>
                `);
                $('#' + key).on('change', function (e) {
                    const type = e.target.value
                    if (key === document.getElementById("target").value) {
                        $('#algorithm').empty()
                        if (type === 'Numerical') {
                            $('#algorithm').append(myClass.updateAlgorithmsSelect(1));
                        } else {
                            $('#algorithm').append(myClass.updateAlgorithmsSelect(2));
                        }
                    }
                });
                const id = column
                if (items.column(column).dtype !== 'string') {
                    $('#' + key).val(FeatureCategories.Numerical)
                } else {
                    $('#' + key).val(FeatureCategories.Nominal)
                }
            });






            // $(document).on('change', '#' + default_target, function (e) {
            //     $("#algorithm").empty();
            //     $("#algorithm").append(myClass.updateAlgorithmsSelect(e.target.value == 1 ? 1 : 2))
            // });
            $("#model_options").empty();
            $('#algorithm').on('change', function () {
                $("#model_options").empty();
            });
            $('#props').append(`
            <div class="column is-12">
                <div class="label is-size-7">Imputation
                    <span id="imputation_help" class="icon has-text-success">
                        <i class="fas fa-info-circle"></i>
                    </span>
                </div>
                <div class="select is-small mb-1">
                    <select id="imputation">
                        <option value="1">Delete rows</option>
                        <option value="2">Mean and Mode</option>
                        <option value="3">Linear regression</option>
                        <option value="4">random forest</option>
                    </select>
                </div>
            </div>
            `);
            items.columns.forEach(column => {
                let key = column.replace(/\s/g, '').replace(/[^\w-]/g, '_');
                $('#normalizations').append(`
                <div class="column is-3">
                    <div class="field">
                    <label class="label is-size-7">${key}</label>
                        <div class="control">
                            <div class="select is-small mb-1">
                                <select id="${key + '--normal'}">
                                    <option value="1">No</option>
                                    <option value="2">Scale</option>
                                    <option value="3">Normal</option>
                                    <option value="4">x^2</option>
                                    <option value="5">ln(x)</option>
                                </select>
                            </div>
                        </div>
                    </div>
                </div>
                `);
                document.getElementById(key + '--normal').addEventListener('change', function () {
                    const target = document.getElementById("target").value;
                    let is_classification = document.getElementById(target).value !== FeatureCategories.Numerical;
                    let data = items.loc({ columns: [key, target] });
                    let normalization_type = document.getElementById(key + '--normal').value
                    myClass.scale_data(data, key, normalization_type)
                    data.dropNa({ axis: 1, inplace: true })
                    myClass.chart_controller.redraw_kde(data, key, target, "nrd", is_classification, true);
                });
            });
            $('#props').append(`
            <div class="column is-10">
                <div class="label is-size-7">Cross Validation
                <span id="cv_help" class="icon has-text-success">
                    <i class="fas fa-info-circle"></i>
                </span>
                </div>
                <div class="select is-small mb-1">
                    <select id="cross_validation">
                        <option value="1">70 % training - 30 % test</option>
                        <option value="2">No</option>
                        <option value="3">K-fold</option>
                    </select>
                </div>
            </div>
            `)
            $('#target').val(default_target)


            $('#target').on('change', function (e) {
                const type = document.getElementById(e.target.value).value
                $('#algorithm').empty()
                if (type === 'Numerical') {
                    $('#algorithm').append(myClass.updateAlgorithmsSelect(1));
                } else {
                    $('#algorithm').append(myClass.updateAlgorithmsSelect(2));
                }
            });




            //modle options
            $('#algorithm').on('change', function (e) {
                const model_type = items.column(default_target).dtype !== 'string' ? 1 : 2;
                const label = model_type == 1 ? "regression" : "classification"
                for (const key in Settings[label]) {
                    if (Settings.hasOwnProperty.call(Settings[label], key)) {
                        const item = Settings[label][key];
                    }
                }
            });
            if (items.column(default_target).dtype !== 'string') {
                $('#props').append(this.createAlgorithmsSelect(1));
            } else {
                $('#props').append(this.createAlgorithmsSelect(2));
            }
            $("#props").append(`
            <div class="column is-3">
            <button class="button is-small is-success" id="config_modal_button">
            <span class="icon is-small">
            <i class="fas fa-cog"></i>
            </span>
            </button>
            </div>
            <div class="column is-12" id="settings" style="display:none">
            </div>`)
            $("#model_name").on("change", () => {
                document.getElementById("settings").innerHTML = ""
                document.getElementById("settings").style.display = "none";

            })
            document.querySelector('#config_modal_button').addEventListener('click', function (e) {
                let model_name = document.getElementById('model_name').value;
                model_name = model_name.replace(/\s+/g, '_').toLowerCase();
                var model = Settings.classification[model_name];
                var options_modal_content = document.getElementById("settings");
                if (window.getComputedStyle(options_modal_content).display !== "none") {
                    options_modal_content.innerHTML = ""
                    options_modal_content.style.display = "none"
                    return
                }
                options_modal_content.innerHTML = ""
                for (const key in model.options) {
                    options_modal_content.style.display = "block"
                    if (Object.hasOwnProperty.call(model.options, key)) {
                        const option_type = model.options[key]["type"]
                        const placeholder = model.options[key]["placeholder"]
                        if (option_type === "number" || option_type === "text") {
                            $('#settings').append(`
                            <div class="column is-12">
                                <div class="field is-horizontal">
                                    <div class="field-label is-small">
                                    <label class="label is-size-7">${key}</label>
                                    </div>
                                    <div class="field-body">
                                    <div class="control">
                                        <input id="${key + "_" + model_name}" class="input is-small" type="${option_type}" placeholder="${placeholder ?? ""}">
                                    </div>
                                    </div>
                                </div>
                            </div>
                            `)
                            if (!!model.options[key]["default"]) {
                                document.getElementById(key + "_" + model_name).value = model.options[key]["default"]
                            }
                        } else if (option_type === "select") {
                            console.log(model.options[key]["for"]);

                            let result = ""
                            let options = model.options[key]["values"]
                            result = `
                            <div class="column is-12">
                                <div class="field is-horizontal">
                                    <div class="field-label is-small">
                                       <label class="label is-size-7 mr-1">${key}</label>
                                    </div>
                                    <div class="field-body">
                                        <div class="select is-small">
                                            <select id="${key + "_" + model_name}">
                                    </div>
                            `
                            for (let i = 0; i < options.length; i++) {
                                result += `<option value="${options[i]?.value}">${options[i].label}</option>`
                            }
                            result += "</select></div></div></div>"
                            $('#settings').append(result)

                        }
                    }
                }
            });
            $('#props').append(`<div class="column is-6"><button class="button is-info mt-2" id="train-button">train</button></div>`);


        } catch (error) {
            throw error
        }

        // $('#kde_select').append(this.createFeaturesDropdown(rowMetadata))
    }

    createAlgorithmsSelect(category) {
        let result = '<div id="algorithm" class="column is-9"><div class="select is-small mb-1"> <select id="model_name" class="select">'
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
        let result = '<div class="select is-small mb-1"> <select id="model_name" class="select">'
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
        const selected_columns = [];
        columns.forEach(column => {
            let key = column.replace(/\s/g, '').replace(/[^\w-]/g, '_');
            if (document.getElementById(key + '-checkbox').checked || get_all) {
                selected_columns.push(column);
            }
        });
        return selected_columns;
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
        let result = '<div  class="column is-12"><div class="label is-size-7">Target</div><div class="select is-small mb-1"> <select id="target">'
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
        let limit = 10;
        //build numerical feature table table
        document.getElementById("stats").style.display = "block"
        var header = "";
        var tbody = "";
        const fileds = ["#", "Min", "Max", "Median", "Mean", "std", "# NAs"]
        for (var p in fileds) {
            header += "<th>" + fileds[p] + "</th>";
        }
        for (let i = 0; i < data.columns.length; i++) {
            if (i < limit) {
                const column = data.columns[i];
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
                    row += "<td>" + data.column(column).isNa().sum() + "</td>";
                    tbody += "<tr>" + row + "</tr>";
                }
            }

        }

        document.getElementById("output").innerHTML =
            '<div class="table-container"><table class="table is-fullwidth is-bordered is-striped is-narrow is-hoverable is-size-7"><thead>' +
            header +
            "</thead><tbody>" +
            tbody +
            "</tbody></table></div>"
            ;
        document.getElementById("data_details_div").innerHTML = '<h2 class="subtitle "> Data shape : (' + data.shape[0] + ',' + data.shape[1] + ')</h2>'
        //build categorical feature table table
        var header_categorical = "";
        var tbody_categorical = "";
        const fileds_categorical = ["#", "Shape", "Mode", "Percentage", "# NAs"]
        for (var p in fileds_categorical) {
            header_categorical += "<th>" + fileds_categorical[p] + "</th>";
        }

        data.columns.forEach((column) => {
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
                row += "<td>" + data.column(column).isNa().sum() + "</td>";

                tbody_categorical += "<tr>" + row + "</tr>";
            }

        });
        document.getElementById("categorical_features").innerHTML =
            '<div class="table-container"><table class="table is-fullwidth is-bordered is-striped is-narrow is-hoverable is-size-7"><thead>' +
            header_categorical +
            "</thead><tbody>" +
            tbody_categorical +
            "</tbody></table></div>"
            ;
    }
    reset(ids, tables, plots) {
        tables.forEach(table => {
            if ($.fn.DataTable.isDataTable('#' + table)) {
                $('#' + table).DataTable().destroy();
                document.getElementById(table).innerHTML = ""
            }

        });
        ids.forEach(id => {
            document.getElementById(id).innerHTML = ""
        });
        // plots.forEach((plot) => Plotly.purge(plot));
    }
    init_upload_button(upoad_handler) {
        $('#upload').append(`
        <div class="file is-small">
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
        <progress class="progress is-small is-primary my-1" max="100" id="progress" style="display:none;">15%</progress>
        <p class="help is-danger">CSV file is supported for now</p>`)
        document.getElementById("parseCVS").addEventListener("change", upoad_handler)
    }
    start_loading() {
        document.getElementById("train-button").classList.add("is-loading")
    }
    stop_loading() {
        document.getElementById("train-button")?.classList.remove("is-loading")
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
    get_numeric_columns(dataset, filter) {
        let selected_columns = this.find_selected_columns(dataset.columns, !filter)
        let selected_columns_types = this.find_selected_columns_types(selected_columns);
        selected_columns = selected_columns.filter(column => {
            let i = selected_columns_types.findIndex(col => col.name === column)
            if (selected_columns_types[i]?.type === FeatureCategories.Numerical) {
                return true;
            }
            return false;
        })
        let numericColumns = []
        dataset.columns.forEach(column => {
            if (dataset.column(column).dtype !== 'string' && column !== "Id" && selected_columns.includes(column)) {
                numericColumns.push(column)
            }
        });
        return numericColumns
    }
    get_categorical_columns(dataset, filter) {
        let selected_columns = this.find_selected_columns(dataset.columns, !filter)
        let selected_columns_types = this.find_selected_columns_types(selected_columns);
        selected_columns = selected_columns.filter(column => {
            let i = selected_columns_types.findIndex(col => col.name === column)
            if (selected_columns_types[i]?.type !== FeatureCategories.Numerical) {
                return true;
            }
            return false;
        })
        let categorical_columns = []
        dataset.columns.forEach(column => {
            if (dataset.column(column).dtype === 'string' && column !== "Id" && selected_columns.includes(column)) {
                categorical_columns.push(column)
            }
        });
        return categorical_columns
    }
    async visualize(dataset, len, file_name) {
        try {
            const myClass = this
            this.renderDatasetStats(dataset);
            let numericColumns = this.get_numeric_columns(dataset, false)
            let categorical_columns = this.get_categorical_columns(dataset, false)
            const target = document.getElementById("target").value;
            const index = numericColumns.findIndex(m => m === target)
            if (index === -1) {
                numericColumns.push(target)
            }
            let columns = [...new Set(numericColumns.concat(categorical_columns))];

            const filterd_dataset = dataset.loc({ columns: columns })
            filterd_dataset.dropNa({ axis: 1, inplace: true })
            numericColumns = numericColumns.filter(m => m !== target)
            let is_classification = document.getElementById(target).value !== FeatureCategories.Numerical;
            //draw kdes
            let limit = 0
            if (numericColumns.length > 0 && limit < 10) {
                document.getElementById("container").innerHTML = "";
                numericColumns.forEach(col => {
                    this.chart_controller.draw_kde(filterd_dataset, col, target, "nrd", is_classification);
                });
                limit++;
            }
            limit = 0
            //draw categories barplot
            if (categorical_columns.length > 0 && limit < 10) {
                document.getElementById("categories_barplots").innerHTML = "";
                categorical_columns.forEach(col => {
                    if (col !== target) {
                        this.chart_controller.draw_categorical_barplot(filterd_dataset.loc({ columns: [col] }).values, target, col);
                    }
                });
                limit++;
            }
            if (is_classification) {
                let labels = dataset.column(target).values;
                let unique_labels = [...new Set(labels)];
                let counts = [];
                for (let i = 0; i < unique_labels.length; i++) {
                    counts.push(labels.filter(m => m === unique_labels[i]).length);
                }
                this.chart_controller.classification_target_chart(counts, unique_labels, file_name, "y_pie_chart", target);
            }
        } catch (error) {
            throw error
        }
    }

    async createSampleDataTable(dataset) {
        try {
            let cols = []
            let column_names = dataset.columns
            for (let i = 0; i < column_names.length; i++) {
                cols.push({ title: column_names[i] })
            }
            new DataTable('#sample_data_table', {
                responsive: true,
                columns: cols,
                data: dataset.head(10).values,
                info: false,
                search: false,
                ordering: false,
                searching: false,
                paging: false,
                bDestroy: true,
            });

        } catch (error) {
            throw error
        }
    }
    toggle_loading_progress(show = false) {
        let element = document.getElementById("progress");
        if (!show) {
            element.style.display = "block";
        } else {
            element.style.display = "none";
        }

    }

}
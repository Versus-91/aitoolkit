import Plotly from 'plotly.js-dist';
import { MinMaxScaler, StandardScaler } from 'danfojs/dist/danfojs-base';
import { calculateRSquared, calculateMSE, encode_name } from './utils';
import { toJSON } from 'danfojs/dist/danfojs-browser/src/index';

import { FeatureCategories, Settings } from "../feature_types.js";
export default class UI {
    constructor(parser, chart_controller) {
        this.data_parser = parser
        this.chart_controller = chart_controller
    }

    get_model_settings() {
        let model_settings = {};
        let model_name = parseInt(document.getElementById('model_name').value);
        const target = document.getElementById("target").value;
        let is_classification = document.getElementById(target).value !== FeatureCategories.Numerical;
        var model;
        if (is_classification) {
            for (const m in Settings.classification) {
                if (Settings.classification[m].value === model_name) {
                    model_name = m
                    model_settings.name = Settings.classification[m].label
                    model = Settings.classification[model_name];
                    break;
                }
            }
        } else {
            for (const m in Settings.regression) {
                if (Settings.regression[m].value === model_name) {
                    model_name = m
                    model_settings.name = Settings.regression[m].label
                    model = Settings.regression[model_name];
                    break;
                }
            }
        }
        model_name = parseInt(document.getElementById('model_name').value);
        for (const option in model?.options) {
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
            case "1": {
                let scaler = new MinMaxScaler()
                scaler.fit(dataset[column])
                dataset.addColumn(column, scaler.transform(dataset[column]), { inplace: true })
                break;
            }
            case "2":
                dataset.addColumn(column, dataset[column].apply((x) => x * x), { inplace: true })
                break;
            case "3":
                dataset.addColumn(column, dataset[column].apply((x) => Math.log(x)), { inplace: true })
                break;
            case "4": {
                let scaler = new StandardScaler()
                scaler.fit(dataset[column])
                dataset.addColumn(column, scaler.transform(dataset[column]), { inplace: true })
                break;
            }
            default:
                break;
        }

    }


    createDatasetPropsDropdown(items) {
        const myClass = this
        //feature selection
        $('#props').empty()
        $('#normalizations').empty()
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
                <div id="config_modal" style="display:none;overflow-y:scroll;max-height: 600px;height:500px">
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
            let key = encode_name(column)
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
                    <div class="select is-small is-fullwidth mb-1">
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
                $('#' + key).val(FeatureCategories.Ordinal)
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
                <div class="label is-size-7">Seed</div>
                <input
                id="seed"
                required
                min="0"
                class="input is-info is-small"
                type="number"
                placeholder="Seed"
                value = "123"
                />
            </div>`);
        $('#props').append(`
            <div class="column is-12">
                <div class="label is-size-7">Imputation
                    <span id="imputation_help" class="icon has-text-success">
                        <i class="fas fa-info-circle"></i>
                    </span>
                </div>
                <div class="select is-small is-fullwidth mb-1">
                    <select id="imputation">
                        <option value="1">Delete rows</option>
                        <option value="2">Mean and Mode</option>
                        <option value="3">Linear regression</option>
                        <option value="4">random forest</option>
                    </select>
                </div>
            </div>
            `);
        $('#props').append(`
            <div class="column is-12">
                <div class="label is-size-7">Cross Validation
                <span id="cv_help" class="icon has-text-success">
                    <i class="fas fa-info-circle"></i>
                </span>
                </div>
                <div class="select is-fullwidth is-small mb-1">
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
            const redraw_plots_data_analysis = new CustomEvent("update_graphs");
            var props_feature_selection_button = document.getElementById("feature_selection_modal")
            props_feature_selection_button.dispatchEvent(redraw_plots_data_analysis)
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
            const target = document.getElementById("target").value;
            let is_classification = document.getElementById(target).value !== FeatureCategories.Numerical;
            var model;
            if (is_classification) {
                for (const key in Settings.classification) {
                    if (Object.hasOwnProperty.call(Settings.classification, key)) {
                        const element = Settings.classification[key];
                        if (element.value === parseInt(model_name)) {
                            model = Settings.classification[key];
                        }
                    }
                }
            } else {
                for (const key in Settings.regression) {
                    if (Object.hasOwnProperty.call(Settings.regression, key)) {
                        const element = Settings.regression[key];
                        if (element.value === parseInt(model_name)) {
                            model = Settings.regression[key];
                        }
                    }
                }
            }
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
                        if (model.options[key]["default"]) {
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

        // $('#kde_select').append(this.createFeaturesDropdown(rowMetadata))
    }

    createAlgorithmsSelect(category) {
        let result = '<div id="algorithm" class="column is-9"><div class="select is-small mb-1"> <select id="model_name" class="select">'
        const label = category == 1 ? "regression" : "classification"
        for (const key in Settings[label]) {
            if (Settings.hasOwnProperty.call(Settings[label], key)) {
                const item = Settings[label][key];
                result += `<option value="${item.value}">${item.label}</option>`
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
                result += `<option value="${item.value}">${item.label}</option>`
            }
        }
        result += '</select></div>'
        return result
    }

    find_selected_columns(columns, get_all = false) {
        const selected_columns = [];
        columns.forEach(column => {
            let key = encode_name(column)
            if (document.getElementById(key + '-checkbox').checked || get_all) {
                selected_columns.push(column);
            }
        });
        return selected_columns;
    }
    find_selected_columns_types(columns, include_target = true) {
        if (include_target === false) {
            const target = document.getElementById("target").value;
            columns = columns.filter(column => column !== target)
        }
        const column_types = []
        columns.forEach(column => {
            let key = encode_name(column)
            column_types.push({
                name: column,
                type: document.getElementById(key).value
            })
        });
        return column_types
    }
    createTargetDropdown(items) {
        let result = '<div  class="column is-12"><div class="label is-size-7">Target</div><div class="select is-fullwidth is-small mb-1"> <select id="target">'
        items.columns.forEach(column => {
            let key = encode_name(column)
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
                const key = encode_name(column)
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
            '<h2 class="subtitle">Continuous features:</h2><div class="table-container"><table class="table is-fullwidth is-bordered is-striped is-narrow is-hoverable is-size-7"><thead>' +
            header +
            "</thead><tbody>" +
            tbody +
            "</tbody></table></div>"
            ;
        document.getElementById("data_details_div").innerHTML = '<h2 class="subtitle"> Data shape : (' + data.shape[0] + ',' + data.shape[1] + ')</h2>'
        //build categorical feature table table
        var header_categorical = "";
        var tbody_categorical = "";
        const fileds_categorical = ["#", "Shape", "Mode", "Percentage", "# NAs"]
        for (var f in fileds_categorical) {
            header_categorical += "<th>" + fileds_categorical[f] + "</th>";
        }

        data.columns.forEach((column) => {
            const key = encode_name(column)
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
            '<h2 class="subtitle">Categorical features:</h2><div class="table-container"><table class="table is-fullwidth is-bordered is-striped is-narrow is-hoverable is-size-7"><thead>' +
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
        <div class="column is-12">
            <div class="select is-small is-fullwidth">
                <select id="decimal_separator">
                    <option value="0">Decimal</option>
                    <option value="1">.</option>
                    <option value="2">,</option>
                </select>
            </div>
        </div>
        <div class="column is-12">
            <label class="checkbox"><input id="header_checkbox" type="checkbox" checked/>  Has header</label>
        </div>
        <div class="column is-12">
            <div class="select is-small is-fullwidth">
                <select id="items_separator">
                    <option value="0">Separator</option>
                    <option value="1">,</option>
                    <option value="2">;</option>
                    <option value="3">space</option>
                </select>
            </div>
        </div>
        <progress class="progress is-small is-primary my-1" max="100" id="progress" style="display:none;">15%</progress>
        <div class="column is-12">
            <div class="select is-small is-fullwidth">
                <select id="sample_data_select">
                <option vlaue="0">Sample data selection</option>
                <option>Iris</option>
                <option>Wine</option>
                <option>Diabetes</option>
                </select>
            </div>
        </div>
        <div class="column is-12">
            <p class="help is-danger">CSV, XLSX, and TXT files are supported for now</p>
        </div>
        `)

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
            if (i !== -1 && selected_columns_types[i]?.type !== FeatureCategories.Numerical) {
                return true;
            }
            return false;
        })
        let categorical_columns = []
        dataset.columns.forEach(column => {
            if (column !== "Id" && selected_columns.includes(column)) {
                categorical_columns.push(column)
            }
        });
        return categorical_columns
    }
    column_types(columns) {
        let selected_columns = this.find_selected_columns(columns, false)
        return this.find_selected_columns_types(selected_columns);
    }
    async visualize(dataset, len, file_name) {
        const current_class = this
        this.renderDatasetStats(dataset);
        let numericColumns = this.get_numeric_columns(dataset, true)
        let categorical_columns = this.get_categorical_columns(dataset, true)
        const target = document.getElementById("target").value;
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
                if (col !== target) {
                    this.chart_controller.draw_kde(filterd_dataset, col, target, "nrd", is_classification);
                }
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
            this.chart_controller.classification_target_chart(counts, unique_labels, file_name, "target_chart", target);
        } else {
            this.chart_controller.regression_target_chart(dataset.column(target).values, "target_chart", target);
        }
        let features = []

        numericColumns = this.get_numeric_columns(dataset, true)
        categorical_columns = this.get_categorical_columns(dataset, true)
        features = Object.values(numericColumns).concat(Object.values(categorical_columns))

        dataset = this.data_parser.handle_missing_values(dataset)
        this.chart_controller.ScatterplotMatrix(dataset.loc({ columns: features }).values, features, dataset.column(target).values, categorical_columns.length,
            is_classification, numericColumns, categorical_columns, dataset).then(() => {
                document.getElementById('splom_update').addEventListener('click', async function (e) {
                    console.log('update');
                    Plotly.purge('scatterplot_mtx');
                    $('#scatterplot_mtx').empty()
                    await current_class.visualize(dataset);
                });
            })

    }


    async createSampleDataTable(dataset) {
        let cols = []
        let column_names = dataset.columns
        for (let i = 0; i < column_names.length; i++) {
            cols.push({ title: column_names[i] })
        }
        new DataTable('#sample_data_table', {
            responsive: true,
            columns: cols,
            data: dataset.head(5).values,
            info: false,
            search: false,
            ordering: false,
            dom: '<"my-class">',
            initComplete: function (settings, json) {
                $('.my-class').html('Sample Data');
            },
            searching: false,
            paging: false,
            bDestroy: true,
        });

    }
    toggle_loading_progress(show = false) {
        let element = document.getElementById("progress");
        if (!show) {
            element.style.display = "block";
        } else {
            element.style.display = "none";
        }

    }
    show_settings(settings, numeric_columns, categorical_columns, target, dataset_name, i) {
        let columns = numeric_columns.concat(categorical_columns)
        let column_types = [];
        for (let i = 0; i < columns.length; i++) {
            const column = columns[i];
            column_types.push({ column: column, type: document.getElementById(column + '--normal') })
        }
        let columns_transformation = '';
        for (let i = 0; i < columns.length; i++) {
            const column_name = encode_name(columns[i]);
            let normalization_type = document.getElementById(column_name + '--normal')?.value;
            if (normalization_type && normalization_type !== "0") {
                columns_transformation += columns[i] + ': ' + normalization_type + ' '
            }
        }

        let content = `
        <div class="column is-12">
        <div class="notification">
        <p class="title my-1 is-5">${settings.name}</p>
        <div class="columns is-multiline is-gapless">`;
        for (const key in settings) {
            if (key !== 'name') {
                if (Object.hasOwnProperty.call(settings, key)) {
                    const element = settings[key];
                    content += `<div class="column is-12 "><p><strong>${key}</strong>: ${element}</p></div>`
                }
            }

        }
        content += `<div class="column is-12 "><p><strong>Dataset name :</strong> ${dataset_name}</p></div>`
        content += `<div class="column is-12 "><p><strong>Target :</strong> ${target}</p></div>`
        content += `<div class="column is-12 "><p><strong>Continious featues :</strong> ${numeric_columns}</p></div>`
        content += `<div class="column is-12 "><p><strong>Categorical featues :</strong> ${categorical_columns}</p></div>`
        content += `<div class="column is-12 "><p><strong>Transformations :</strong> ${columns_transformation}</p></div>`
        content += `<div class="column is-12 ">
        <button class="button is-danger" id="remove_${i}"> Remove results </button>
        </div>`

        content += `</div></div></div>`
        $("#tabs_info li[data-index='" + i + "'] #results_" + i + "").append(content);

        document.getElementById("remove_" + i).addEventListener('click', () => {
            $('#' + 'tab_' + i).remove();
            if (document.getElementById(target).value !== FeatureCategories.Numerical) {
                Plotly.purge('pca_results_' + i)
                $('#predictions_table_' + i).DataTable().destroy();
                $('#' + 'info_' + i).remove();
            } else {
                Plotly.purge('regression_y_yhat_' + i)
                $('#predictions_table_' + i).DataTable().destroy();
                $('#' + 'info_' + i).remove();
            }

        });
    }
    create_model_result_tab(index) {
        $("#tabs_content").append(`
        <li data-index="${index}" id="tab_${index}">
           <a>${index}</a>
        </li>`)
        $("#tabs_info").append(`
        <li data-index="${index}" id="info_${index}"  class=" tabs-li">
        <div id="results_${index}" class="columns is-multiline"></div>
        </li>`)
        $("#tabs_content li").removeClass("is-active");
        $("#tabs_info li").removeClass("is-active");
        $("#tabs_info li[data-index='" + index + "']").addClass("is-active");
        $("#tabs_content li[data-index='" + index + "']").addClass("is-active");
    }
    init_regression_results_tab_linear_regression(index) {
        let content = `
        <div class="column is-7">
            <div class="table-container">
            <table
                class="table has-text-centered nowrap is-striped is-bordered is-narrow is-hoverable is-size-7"
                id="metrics_table_${index}" >
                        <thead>
            <tr>
                <th colspan="1"></th>
                <th colspan="3">OLS</th>
                <th colspan="3">lambda min</th>
                <th colspan="3">lambda 1se</th>
            </tr>
            <tr>
                <th>name</th>
                <th>coef</th>
                <th>st.d.</th>
                <th><i>p</i>-value</th>
                <th>coef</th>
                <th>st.d.</th>
                <th><i>p</i>-value</th>
                <th>coef</th>
                <th>st.d.</th>
                <th><i>p</i>-value</th>
            </tr>
        </thead>
          <tfoot class="has-text-centered" style=" font-weight: normal">
    <tr>
      <th></th>
      <th colspan="3"></th>
      <th colspan="3"></th>
      <th colspan="3"></th>

    </tr>
  </tfoot>
            </table>
           </div>
        </div>        
        <div class="column is-5 mt-4" style="height:400px;" id="parameters_plot_${index}">
        </div>
        <div class="column is-12" id="metrics_${index}">
        </div>

        <div class="column is-3">
            <div id="errors_${index}" width="100%" style="height:200px"></div>
        </div>
        <div class="column is-3">
            <div id="regression_y_yhat_${index}" width="100%" style="height:200px">
           </div>
        </div>
        <div class="column is-3">
            <div id="regression_y_yhat_min_${index}" width="100%" style="height:200px">
           </div>
        </div>
        <div class="column is-3">
            <div id="regression_y_yhat_1se_${index}" width="100%" style="height:200px">
           </div>
        </div>
        <div class="column is-3">
            <div id="regularization_${index}" width="100%" style="height:200px"></div>
        </div>
        <div class="column is-3">
            <div id="regression_residual_${index}" width="100%" style="height:200px">
           </div>
        </div>
        <div class="column is-3">
            <div id="regression_residual_min_${index}" width="100%" style="height:200px">
           </div>
        </div>
        <div class="column is-3">
            <div id="regression_residual_1se_${index}" width="100%" style="height:200px">
           </div>
        </div>
        <div class="column is-3">
        </div>
        <div class="column is-3">
            <div id="qqplot_ols_${index}" width="100%" style="height:200px">
           </div>
        </div>
        <div class="column is-3">
            <div id="qqplot_min_${index}" width="100%" style="height:200px">
           </div>
        </div>
        <div class="column is-3">
            <div id="qqplot_1se_${index}" width="100%" style="height:200px">
           </div>
        </div>
`
        $("#tabs_info li[data-index='" + index + "'] #results_" + index + "").append(content);
    }
    init_regression_results_tab(index) {
        let content = `
        <div class="column is-12">
            <div id="metrics_${index}">
           </div>
        </div>
        <div class="column is-6">
            <div id="regression_y_yhat_${index}" width="100%" style="height:200px">
           </div>
        </div>
`
        $("#tabs_info li[data-index='" + index + "'] #results_" + index + "").append(content);
    }
    init_tooltips(tippy) {
        tippy('#kde_help', {
            interactive: true,
            popperOptions: {
                positionFixed: true,
            },
            content: 'Default bandwidth method :Silverman’s rule of thumb',
        });
        tippy('#normalization_help', {
            interactive: true,
            popperOptions: {
                positionFixed: true,
            },
            content: '<p>not functional yet</p><p>standard scaler uses z = (x - u) / s</p><p>Transform features by scaling each feature to a given range</p>',
            allowHTML: true,
        });
        tippy('#imputation_help', {
            interactive: true,
            popperOptions: {
                positionFixed: true,
            },
            content: 'currently we are just deleting rows with missing values',
            allowHTML: true,
        });
        tippy('#cv_help', {
            interactive: true,
            popperOptions: {
                positionFixed: true,
            },
            content: 'option 1 and 2 are working',
            allowHTML: true,
        });
    }

    init_tabs_events() {
        $("#tabs_content").on("click", "li", function () {
            var index = $(this).data("index");
            $("#tabs_content li").not(this).removeClass("is-active");
            $("#tabs_info li").removeClass("is-active");
            $("#tabs_info li[data-index='" + index + "']").addClass("is-active");
            $(this).toggleClass("is-active ");
        });
        $(".tabs ul li").click(function () {
            window.dispatchEvent(new Event('resize'));
        });
    }
    predictions_table_regression(x, y, predictions, tab_index) {
        let content = `
        <div class="column is-12">
            <table id="predictions_table_${tab_index}" class="table is-bordered is-hoverable is-narrow display is-size-7" width="100%">
           </table>
        </div>`
        $("#tabs_info li[data-index='" + tab_index + "'] #results_" + tab_index + "").append(content);
        let table_columns = [];
        x.addColumn("y", y, { inplace: true });
        x.addColumn("predictions: ", predictions, { inplace: true });
        x.columns.forEach(element => {
            table_columns.push({ title: element });
        });
        new DataTable('#predictions_table_' + tab_index, {
            pageLength: 10,
            responsive: true,
            paging: true,
            columnDefs: [
                {
                    render: function (data, type, row) {
                        return data.toFixed(2);
                    },
                    targets: "_all",
                }
            ],
            "bPaginate": true,
            columns: table_columns,
            data: x.values,
            bDestroy: true,
        });
    }
    predictions_table(x, y, encoder, predictions, probs = null, tab_index = 0) {
        let content = `
        <div class="column is-12">
            <table id="predictions_table_${tab_index}" class="table is-bordered is-hoverable is-narrow display is-size-7" width="100%">
           </table>
        </div>`
        $("#tabs_info li[data-index='" + tab_index + "'] #results_" + tab_index + "").append(content);
        let table_columns = [];
        if (probs !== null) {
            x.addColumn("probs", probs, { inplace: true });
        }
        x.addColumn("y", y, { inplace: true });
        x.addColumn("predictions: ", encoder.inverseTransform(predictions), { inplace: true });
        x.columns.forEach(element => {
            table_columns.push({ title: element });
        });
        new DataTable('#predictions_table_' + tab_index, {
            pageLength: 10,
            responsive: true,
            paging: true,
            "bPaginate": true,
            columns: table_columns,
            data: x.values,
            bDestroy: true,
            columnDefs: [
                {
                    render: function (data, type, row) {
                        for (let i = 0; i < data.length; i++) {
                            data[i] = data[i].toFixed(2);
                        }
                        return data
                    },
                    targets: [-3]
                },
                {
                    render: function (data, type, row) {
                        return data.toFixed(2);
                    },
                    targets: [...Array(table_columns.length - 3).keys()]
                }
            ],
            rowCallback: function (row, data, index) {
                var prediction = data[table_columns.length - 1];
                var y = data[table_columns.length - 2];
                if (prediction !== y) {
                    $(row).addClass('is-danger');
                }
            }
        });
    }
    regression_metrics_display(y_test, predictions, tab_index) {
        let r2 = calculateRSquared(y_test.values, predictions);
        let mse = calculateMSE(y_test.values, predictions);
        let content = `
                        <p>R squared : ${r2.toFixed(2)}</p>
                        <p>Mean Squared Error: ${mse.toFixed(2)}</p>
                     `;
        $("#metrics_" + tab_index).append(content);
    }
}
import { FeatureCategories, Settings } from "./feature_types.js";
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
    createDatasetPropsDropdown(items) {
        let rowMetadata = this.data_parser.findDataTypes(items);
        let header = "";
        const lastProperty = Object.keys(items[0])[Object.keys(items[0]).length - 1];
        for (const key in rowMetadata) {
            let options = ""
            $('#props').append(`
            <div class="column is-4">
                <h4>${this.insertSpaces(key)} - ${key === lastProperty ? "Output" : "Input"}</h4>
                <div class="select mb-1">
                    <select id="${key === lastProperty ? key + "-y" : key}">
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
            const id = key === lastProperty ? key + "-y" : key
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
        $('#props').append(this.createTargetDropdown(rowMetadata))
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
                row += "<td>" + ss.median(formattedData) + "</td>";
                row += "<td>" + ss.mean(formattedData) + "</td>";
                row += "<td>" + ss.standardDeviation(formattedData) + "</td>";
                row += "<td>" + "NA" + "</td>";
                tbody += "<tr>" + row + "</tr>";
            }
        }

        //build a table
        document.getElementById("output").innerHTML =
            '<table class="table is-bordered"><thead>' +
            header +
            "</thead><tbody>" +
            tbody +
            "</tbody></table>"
            ;
    }
}
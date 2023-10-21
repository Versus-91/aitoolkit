export default class UI {
    constructor(parser) {
        data_parser = parser
    }
    drawPieChart() {
        var data = [{
            values: [19, 26, 55],
            labels: ['Residential', 'Non-Residential', 'Utility'],
            type: 'pie'
        }];

        var layout = {
            height: 400,
            width: 500
        };

        Plotly.newPlot('myDiv', data, layout);
    }
    createDatasetPropsDropdown(items) {
        let rowMetadata = data_parser.findDataTypes(items);
        let header = "";
        const lastProperty = Object.keys(items[0])[Object.keys(items[0]).length - 1];
        for (const key in rowMetadata) {
            let options = ""
            $('#props').append(`
            <div class="column is-4">
                <h4>${insertSpaces(key)} - ${key === lastProperty ? "Output" : "Input"}</h4>
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
            $('#props').append(createAlgorithmsSelect(1));
        } else if (rowMetadata[lastProperty] === FeatureCategories.Categorical) {
            $('#props').append(createAlgorithmsSelect(2));
        }
        $(document).on('change', '#' + lastProperty + '-y', function (e) {
            $("#algorithm").remove();
            $("#props").append(createAlgorithmsSelect(e.target.value == 1 ? 1 : 2))
        });
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
    insertSpaces(string) {
        string = string.replace(/([a-z])([A-Z])/g, '$1 $2');
        string = string.replace(/([A-Z])([A-Z][a-z])/g, '$1 $2')
        return string;
    }
}
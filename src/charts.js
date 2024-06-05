import Plotly from 'plotly.js-dist';
import PCA from './dimensionality-reduction/pca';
import { binarize } from './utils'
import * as ss from "simple-statistics"
import { schemeCategory10 } from 'd3-scale-chromatic';
import { FeatureCategories } from "../feature_types.js";
import { metrics } from './utils.js';
import * as tfvis from '@tensorflow/tfjs-vis';
import { scale_data } from './utils';

export default class ChartController {
    constructor(data_processor) {
        this.data_processor = data_processor
        this.color_scheme = schemeCategory10;
    }

    classification_target_chart(values, labels, name, container, title = "") {
        var uniqueLabels = [...new Set(labels)];
        var colorIndices = labels.map(label => this.indexToColor(uniqueLabels.indexOf(label)));
        var data = [];
        data.push({
            name: "Count",
            data: values.map((item, i) => ({ y: item, color: colorIndices[i] }))
        })

        Highcharts.chart(container, {
            credits: {
                enabled: false
            },
            title: {
                text: ""
            },
            chart: {
                type: 'column'
            },
            xAxis: {
                categories: uniqueLabels,
            },
            yAxis: {
                min: 0,
            },
            plotOptions: {
                column: {
                    pointPadding: 0.1,
                    borderWidth: 0
                }
            },
            colors: colorIndices,
            series: data
        });
    }
    regression_target_chart(items, container, name) {
        let kde_data = [];
        let ys = [];
        let bandwidth = this.nrd(items.map(item => item)).toFixed(2);
        let items_range = items
        let minValue = Math.min(...items_range);
        let maxValue = Math.max(...items_range);
        items_range.push(minValue - parseFloat(bandwidth))
        items_range.push(maxValue + parseFloat(bandwidth))
        var breaks = ss.equalIntervalBreaks(items_range, 100);
        let kde = ss.kernelDensityEstimation(items, 'gaussian', 'nrd');
        breaks.forEach((item) => {
            ys.push(kde(item, 'nrd'));
            kde_data.push([item, ys[ys.length - 1]]);
        });


        Highcharts.chart(container, {
            credits: {
                enabled: false
            },
            legend: {
                enabled: false,
                verticalAlign: 'top',
            },
            chart: {
                height: '300',
                type: "spline",
                animation: true,
            },
            title: {
                text: name // Assuming `column` is defined elsewhere
            },
            yAxis: {
                title: { text: null }
            },
            tooltip: {
                valueDecimals: 3
            },
            plotOptions: {
                series: {
                    marker: {
                        enabled: false
                    },
                    dashStyle: "shortdot",
                    area: true
                }
            },
            series: [{
                type: 'area',
                dashStyle: "solid",
                lineWidth: 2,
                data: kde_data
            }]
        });
    }
    draw_categorical_barplot(column_values, target, title) {
        const key = title + "- barplot";
        $("#categories_barplots").append(`<div class="column is-4" style="height:40vh;" id="${key}"></div>`)
        const countOccurrences = column_values.reduce((acc, val) => {
            acc[val] = (acc[val] || 0) + 1;
            return acc;
        }, {});
        const countArray = Object.entries(countOccurrences).map(([value, count]) => ({ value: value, count }));
        countArray.sort((a, b) => b.count - a.count);
        const top5 = countArray.slice(0, 5);
        new Highcharts.Chart({
            chart: {
                renderTo: key,
                type: 'column'
            },
            xAxis: {
                categories: top5.map(m => m.value),
            },
            title: {
                text: title
            },
            yAxis: {
                min: 0,
                labels: {
                    overflow: 'justify'
                }
            },
            credits: {
                enabled: false
            },
            plotOptions: {
                bar: {
                    dataLabels: {
                        enabled: true
                    }
                }
            },
            series: [{
                showInLegend: false,
                name: title,
                data: top5.map(m => m.count)
            }]
        });

    }
    roc_chart(container, true_positive_rates, false_positive_rates) {
        var trace = {
            x: false_positive_rates,
            y: true_positive_rates,
            type: 'scatter',
            mode: 'lines',
            name: 'ROC Curve',
        };
        var trace2 = {
            x: [0, 1],
            y: [0, 1],
            type: 'scatter',
            name: 'diagonal',
        };
        var layout = {
            title: 'ROC Curve',
            xaxis: { title: 'False Positive Rate' },
            yaxis: { title: 'True Positive Rate' },
        };

        var data = [trace, trace2];

        Plotly.newPlot(container, data, layout);
    }
    falsePositives(yTrue, yPred) {
        return tf.tidy(() => {
            const one = tf.scalar(1);
            const zero = tf.scalar(0);
            return tf.logicalAnd(yTrue.equal(zero), yPred.equal(one))
                .sum()
                .cast('float32');
        });
    }
    indexToColor(index) {
        return this.color_scheme[index + 1 % this.color_scheme.length];
    }
    reshape(array, shape) {
        if (shape.length === 0) return array[0];

        const [size, ...restShape] = shape;
        const result = [];
        const restSize = restShape.reduce((a, b) => a * b, 1);
        console.log(restSize);

        for (let i = 0; i < size; i++) {
            result.push(this.reshape(array.slice(i * restSize, (i + 1) * restSize), restShape));
        }

        return result;
    }
    async plot_tsne(data, labels, regression_labels) {
        document.getElementById("dimensionality_reduction_panel_tsne").style.display = "block"
        console.assert(Array.isArray(data));
        // Create some data
        // const items = tf.randomUniform([2000, 10]);

        // Get a tsne optimizer
        const tsneOpt = tsne.tsne(window.tensorflow.tensor2d(data));

        // Compute a T-SNE embedding, returns a promise.
        // Runs for 1000 iterations by default.
        await tsneOpt.compute();
        // tsne.coordinate returns a *tensor* with x, y coordinates of
        // the embedded data.
        const coordinates = tsneOpt.coordinates();
        // let model = new TSNE();
        // var Y = await model.train(data)
        const items = coordinates.dataSync()
        const Y = this.reshape(items, coordinates.shape)
        let x = []
        let traces = []
        if (labels.length > 0) {
            labels = labels.flat()
            var uniqueLabels = [...new Set(labels)];
            let points_labled = Y.map(function (item, i) {
                return {
                    label: labels[i],
                    'x': item[0],
                    'y': item[1]
                }
            }
            )
            uniqueLabels.forEach((label, i) => {
                var items_for_label = points_labled.filter(m => m.label === label)
                traces.push({
                    x: items_for_label.map(m => m.x),
                    y: items_for_label.map(m => m.y),
                    mode: 'markers',
                    type: 'scatter',
                    name: label,
                    marker: {
                        size: 4,
                        color: this.indexToColor(i),
                    }
                })
            })
        } else {
            let points = Y.map(function (item, i) {
                x.push(regression_labels[i][0])
                return {
                    'x': item[0],
                    'y': item[1]
                }
            })
            traces.push({
                x: x,
                y: points.map(m => m.y),
                mode: 'markers+text',
                type: 'scatter',
                marker: {
                    size: 4,
                    colorscale: 'YlOrRd',
                    color: x,
                    colorbar: {
                        title: 'Color Scale Legend',
                        titleside: 'right'
                    }
                },
            })

        }

        var layout = {
            showlegend: true,
            margin: {
                l: 20,
                r: 20,
                b: 20,
                t: 20,
                pad: 5
            },
            legend: {
                x: 1,
                xanchor: 'right',
                y: 1
            },
        };
        Plotly.newPlot('tsne', traces, layout, { responsive: true, modeBarButtonsToRemove: ['resetScale2d', 'select2d', 'resetViews', 'sendDataToCloud', 'hoverCompareCartesian', 'lasso2d', 'drawopenpath '] });

    }
    trueNegatives(yTrue, yPred) {
        return tf.tidy(() => {
            const zero = tf.scalar(0);
            return tf.logicalAnd(yTrue.equal(zero), yPred.equal(zero))
                .sum()
                .cast('float32');
        });
    }

    // TODO(cais): Use tf.metrics.falsePositiveRate when available.
    falsePositiveRate(yTrue, yPred) {
        return tf.tidy(() => {
            const fp = this.falsePositives(yTrue, yPred);
            const tn = this.trueNegatives(yTrue, yPred);
            return fp.div(fp.add(tn));
        });
    }
    drawROC(targets, probs) {

        return tf.tidy(() => {
            const thresholds = [
                0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
                0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0
            ];
            const tprs = [];  // True positive rates.
            const fprs = [];  // False positive rates.
            let area = 0;
            for (let i = 0; i < thresholds.length; ++i) {
                const threshold = thresholds[i];
                const threshPredictions = binarize(probs, threshold).as1D();

                const fpr = this.falsePositiveRate(targets, threshPredictions).dataSync()[0];
                const tpr = tf.metrics.recall(targets, threshPredictions).dataSync()[0];

                fprs.push(fpr);
                tprs.push(tpr);
                // Accumulate to area for AUC calculation.
                if (i > 0) {
                    area += (tprs[i] + tprs[i - 1]) * (fprs[i - 1] - fprs[i]) / 2;
                }
            }
            return [area, fprs, tprs];
        });
    }
    nrd(x) {
        let s = ss.standardDeviation(x);
        const iqr = ss.interquartileRange(x);
        if (typeof iqr === "number") {
            s = Math.min(s, iqr / 1.34);
        }
        return 1.06 * s * Math.pow(x.length, -0.2);
    }
    hexToRgb(hex) {
        var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16),
            a: 0.5
        } : null;
    }
    kernelFunctions = {
        gaussian: function (u) {
            return Math.exp(-0.5 * u * u) / Math.sqrt(2 * Math.PI);
        },
        uniform: function (x) {
            return Math.abs(x) <= 1 ? 0.5 : 0;
        },
        triangular: function (x) {
            return Math.abs(x) <= 1 ? 1 - Math.abs(x) : 0;
        },
        biweight: function (x) {
            return Math.abs(x) <= 1 ? 15 / 16 * Math.pow(1 - x * x, 2) : 0;
        },
        triweight: function (x) {
            return Math.abs(x) <= 1 ? 35 / 32 * Math.pow(1 - x * x, 3) : 0;
        },
        Epanechnikov: function (x) {
            return Math.abs(x) <= 1 ? 0.75 * (1 - x * x) : 0;
        }
    };

    draw_kde(dataset, column, target_name, bandwidth = "nrd", is_classification = false, redrawing = false) {
        let items = dataset.column(column).values;
        let default_bandwidth = this.nrd(items).toFixed(2);
        let raw_values = dataset.loc({ columns: [column, target_name] });
        let uniqueLabels = [...new Set(raw_values.column(target_name).values)];
        let column_values = raw_values.values;
        let subsets = [];
        var colorIndices = uniqueLabels.map(label => this.indexToColor(uniqueLabels.indexOf(label)));
        if (!is_classification) {
            subsets.push(dataset[column].values);
        } else {
            for (let i = 0; i < uniqueLabels.length; i++) {
                const label = uniqueLabels[i];
                let subset = [];
                for (let i = 0; i < column_values.length; i++) {
                    const item = column_values[i];
                    if (item[1] === label) {
                        subset.push(item[0])
                    }
                }
                subsets.push(subset);
            }
        }

        document.getElementById("kde_panel").style.display = "block";

        var newColumn = document.createElement("div");
        newColumn.className = "column is-3";
        newColumn.setAttribute("id", column + '-kde-plot');
        if (!redrawing) {
            // let key = column.replace(/\s/g, '').replace(/[^\w-]/g, '_');
            let key = column;

            $("#container").append(
                `<div class="column is-4 is-size-6-tablet my-1">
                <div class="columns is-multiline">
                <div class="column is-12" >
                    <div id="${column + '-kde-plot'}"> </div>
                    <div id="${column + '-boxplot'}" style="height:20vh;width: 100%">
                    </div>
                    <div class="field has-addons has-addons-centered my-1">
                    <div class="control">
                    <span class="select is-small">
                      <select id="${column + '-kernel_type'}">
                      <option value="gaussian">gaussian</option>
                        <option value="uniform">uniform</option>
                        <option value="triangular">triangular</option>
                        <option value="biweight">biweight</option>
                        <option value="triweight">triweight</option>
                        <option value="Epanechnikov">Epanechnikov</option>
                      </select>
                    </span>
                    <p class="help is-success">Kernel</p>
                  </div>
                  <div class="control">
                        <div class="select is-small">
                            <select id="${key + '--normal'}">
                                <option value="0">No</option>
                                <option value="1">Scale</option>
                                <option value="2">x^2</option>
                                <option value="3">ln(x)</option>
                                <option value="4">Standardize </option>
                            </select>
                        </div>
                    <p class="help is-success">Normalization</p>
                    </div>
                        <div class="control">
                            <input class="input is-small" type="number"  min="0" id="${column + '-kde'}" value="${default_bandwidth}">
                            <p class="help is-success">Bandwidth</p>
                        </div>
                        <p class="control">
                            <a class="button is-success is-small" id="${column + '-kde-button'}">
                                Apply
                            </a>
                        </div>
                    </div>
                  </div>
                </div>`
            );
            document.getElementById(key + '--normal').addEventListener('change', function () {
                const target = document.getElementById("target").value;
                let is_classification = document.getElementById(target).value !== FeatureCategories.Numerical;
                let data = dataset.loc({ columns: [key, target] });
                let normalization_type = document.getElementById(key + '--normal').value
                scale_data(data, key, normalization_type)
                data.dropNa({ axis: 1, inplace: true })
                var newBandwidth = parseFloat(document.getElementById(column + '-kde').value);
                current_class.draw_kde(data, key, target, newBandwidth, is_classification, true);
            });
        }
        var current_class = this;
        document.getElementById(column + '-kde-button').addEventListener("click", function () {
            const target = document.getElementById("target").value;
            let is_classification = document.getElementById(target).value !== FeatureCategories.Numerical;
            let data = dataset.loc({ columns: [column, target] });
            let normalization_type = document.getElementById(column + '--normal').value
            scale_data(data, column, normalization_type)
            var newBandwidth = parseFloat(document.getElementById(column + '-kde').value);
            data.dropNa({ axis: 1, inplace: true })
            current_class.draw_kde(data, column, target, newBandwidth, is_classification, true);
        });
        let container_id = column + '-kde-plot';
        let items_range = raw_values.column(column).values
        let minValue = Math.min(...items_range);
        let maxValue = Math.max(...items_range);
        items_range.push(minValue - parseFloat(default_bandwidth))
        items_range.push(maxValue + parseFloat(default_bandwidth))
        var breaks = ss.equalIntervalBreaks(items_range, 100);
        let allData = [];
        let kernel_type = document.getElementById(column + "-kernel_type")?.value ?? "gaussian"
        // Loop through subsets to generate data for all subsets
        let traces = []
        let kde;
        if (is_classification) {
            for (let i = 0; i < subsets.length; i++) {
                if (subsets[i].length > 2) {
                    let ys = [];
                    kde = ss.kernelDensityEstimation(subsets[i], this.kernelFunctions[kernel_type], bandwidth);
                    let data = [];
                    breaks.forEach((item) => {
                        ys.push(kde(item, bandwidth));
                        data.push([item, ys[ys.length - 1]]);
                    });
                    allData.push(data);
                } else {
                    allData.push([]);
                }
                traces.push({
                    name: uniqueLabels[i],
                    x: subsets[i],
                    marker: {
                        color: colorIndices[i]
                    },
                    type: 'box',
                })
            }
        } else {
            for (let i = 0; i < subsets.length; i++) {
                if (subsets[i].length > 2) {
                    let ys = [];
                    kde = ss.kernelDensityEstimation(subsets[i], this.kernelFunctions[kernel_type], bandwidth);
                    let data = [];
                    breaks.forEach((item) => {
                        ys.push(kde(item, bandwidth));
                        data.push([item, ys[ys.length - 1]]);
                    });
                    allData.push(data);
                } else {
                    allData.push([]);
                }
            }
            traces.push({
                name: column,
                x: items,
                type: 'box',
            })
        }

        let animationDuration = 4000;

        var layout = {

            yaxis: {
                visible: false,
            },
            showlegend: false,
            margin: {
                l: 20,
                r: 10,
                b: 60,
                t: 10,
            },
            legend: {
                x: 1,
                xanchor: 'right',
                y: 1
            },
        };
        Plotly.newPlot(column + '-boxplot', traces, layout, { autosize: true, responsive: true, modeBarButtonsToRemove: ['pan', 'resetScale2d', 'select2d', 'resetViews', 'sendDataToCloud', 'hoverCompareCartesian', 'lasso2d', 'drawopenpath '] });
        Highcharts.chart(container_id, {
            credits: {
                enabled: false
            },
            legend: {
                enabled: is_classification ? true : false, align: 'right',
                verticalAlign: 'top',
            },
            chart: {
                height: '300',
                type: "spline",
                animation: true,
            },
            title: {
                text: column // Assuming `column` is defined elsewhere
            },
            yAxis: {
                title: { text: null }
            },
            tooltip: {
                valueDecimals: 3
            },
            plotOptions: {
                series: {
                    marker: {
                        enabled: false
                    },
                    dashStyle: "shortdot",
                    color: colorIndices,
                    animation: {
                        duration: animationDuration
                    },
                    area: true
                }
            },
            series: allData.map((data, index) => ({
                type: 'area',
                name: uniqueLabels[index],
                dashStyle: "solid",
                lineWidth: 2,
                color: colorIndices[index],
                data: data
            }))
        });
        window.dispatchEvent(new Event('resize'));
    }

    async draw_classification_pca(dataset, labels, missclassifications, uniqueLabels, index) {

        const pca = new PCA(dataset, { center: true, scale: true });
        var colorIndices = labels.map(label => this.indexToColor(uniqueLabels.indexOf(label)));
        const pca_data = await pca.predict(dataset, { nComponents: 2 })
        let x = []
        let y = []
        let x_error = []
        let y_error = []
        pca_data[0].forEach((element, i) => {
            if (missclassifications.includes(i)) {
                x_error.push(element[0])
                y_error.push(element[1])
            } else {
                x.push(element[0])
                y.push(element[1])
            }

        });
        var trace1 = {
            x: x,
            y: y,
            name: 'Predictions',
            text: labels,
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 4,
                color: colorIndices,
                symbol: 'circle'
            },
        };
        var trace2 = {
            name: 'Missclassifications',
            x: x_error,
            y: y_error,
            text: labels,
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 7,
                color: colorIndices,
                symbol: 'cross'
            },
        };
        var data = [trace1, trace2];

        var chart_container = `<div class="column is-6" style="height: 40vh" id="pca_results_${index}"></div>`
        $("#tabs_info li[data-index='" + index + "'] #results_" + index + "").append(chart_container);

        Plotly.newPlot('pca_results_' + index, data, {
            showlegend: true,
            margin: {
                l: 40,
                r: 20,
                b: 40,
                t: 20,
            },
            legend: {
                x: 1,
                xanchor: 'right',
                y: 1
            },
            xaxis: {

                title: 'PC1'
            },
            yaxis: {
                title: 'PC2'
            }
        }, { responsive: true, modeBarButtonsToRemove: ['resetScale2d', 'select2d', 'resetViews', 'sendDataToCloud', 'hoverCompareCartesian', 'lasso2d', 'drawopenpath '] });

    }
    async draw_pca(dataset, labels, regression_labels) {
        document.getElementById("dimensionality_reduction_panel_pca").style.display = "block"
        document.getElementById("pca-1").innerHTML = ""
        const pca = new PCA(dataset, { center: true, scale: true });

        labels = labels.flat()
        var uniqueLabels = [...new Set(labels)];
        var colorIndices = labels.map(label => this.indexToColor(uniqueLabels.indexOf(label)));

        const pca_x = await pca.predict(dataset, { nComponents: 3 })
        const pca_data = pca_x[0]

        let x = []
        let pc1 = []
        let x_axis = document.getElementById("pca_x").value;
        let y_axis = document.getElementById("pca_y").value;
        pca_data.forEach((element, i) => {
            pc1.push({
                x: element[x_axis - 1],
                y: element[y_axis - 1],
                label: labels[i]
            })
            x.push(regression_labels[i][0])
        });
        let traces1 = []
        if (uniqueLabels.length !== 0) {
            uniqueLabels.forEach((label, i) => {
                var items_for_label = pc1.filter(m => m.label === label)
                traces1.push({
                    x: items_for_label.map(m => m.x),
                    y: items_for_label.map(m => m.y),
                    mode: 'markers',
                    type: 'scatter',
                    name: label,
                    marker: {
                        size: 4,
                        color: this.indexToColor(i),
                    }
                })
            })
        } else {
            traces1.push({
                x: pc1.map(m => m.x),
                y: pc1.map(m => m.y),
                mode: 'markers',
                type: 'scatter',
                marker: {
                    color: x,
                    colorscale: 'YlOrRd',
                    size: 4,
                    colorbar: {
                        title: 'Color Scale Legend',
                        titleside: 'right'
                    }
                },

            })
        }
        let cumulatedExplainedVaraince = []
        let sum = 0
        pca_x[2].forEach(element => {
            sum = sum + element
            cumulatedExplainedVaraince.push(sum)
        });
        Highcharts.chart('scree_plot', {
            credits: {
                enabled: false
            },
            title: {
                text: '',
            },
            yAxis: {
                min: 0,
                max: 1,
                title: {
                    text: 'Explained variance'
                },
                plotLines: [{
                    value: 0.9,
                    dashStyle: 'shortdash',
                    color: 'grey',
                    width: 1,
                    zIndex: 4,
                    label: {
                        text: '0.9', align: "right",
                    }
                }, {
                    value: 0.8,
                    dashStyle: 'shortdash',
                    color: 'darkgrey',
                    width: 1,
                    zIndex: 4,
                    label: {
                        text: '0.8', align: "right",
                    }
                }]

            },
            xAxis: {
                labels: {
                    enabled: true,
                    formatter: function () {
                        return this.value + 1;
                    }
                },

                title: {
                    text: 'Number of components'
                },
            },
            series: [{
                name: 'Propotional',
                color: "blue",
                data: pca_x[2]
            },
            {
                name: 'Cumulative',
                color: "red",
                data: cumulatedExplainedVaraince
            }],

        });
        Plotly.newPlot('pca-1', traces1, {
            autosize: true,
            showlegend: true,
            margin: {
                l: 30,
                r: 20,
                b: 20,
                t: 20,
                pad: 5
            },
            legend: {
                x: 1,
                xanchor: 'right',
                y: 1
            },
            xaxis: {
                title: 'PC' + x_axis
            },
            yaxis: {
                title: 'PC' + y_axis
            }
        }, { responsive: true });
    }
    drawStackedHorizontalChart(categories, lable) {
        var trace1 = {
            x: [20, 14, 23],
            y: ['giraffes', 'orangutans', 'monkeys'],
            name: 'SF Zoo',
            orientation: 'h',
            marker: {
                color: 'rgba(55,128,191,0.6)',
                width: 1
            },
            type: 'bar'
        };

        var trace2 = {
            x: [12, 18, 29],
            y: ['giraffes', 'orangutans', 'monkeys'],
            name: 'LA Zoo',
            orientation: 'h',
            type: 'bar',
            marker: {
                color: 'rgba(255,153,51,0.6)',
                width: 1
            }
        };

        var data = [trace1, trace2];

        var layout = {
            title: 'Colored Bar Chart',
            barmode: 'stack'
        };

        Plotly.newPlot('myDiv', data, layout);

    }
    regularization_plot(xs, ys, labels) {
        const traces = []
        labels.forEach((element, i) => {
            traces.push({
                x: xs,
                y: ys.map(m => m[i]),
                type: 'scatter',
                name: element,
                mode: 'line'
            })
        });
        var layout = {
            colorway: ['#f3cec9', '#e7a4b6', '#cd7eaf', '#a262a9', '#6f4d96', '#3d3b72', '#182844'],
            title: 'Lasso Coefficients as Alpha varies',
            xaxis: {
                type: 'log',
                title: 'Alpha (Regularization Strength)'
            },
            yaxis: {
                title: 'Coefficient Value'
            }
        };
        Plotly.newPlot('lasso_plot', traces, layout);
    }
    argmax(array) {
        return array.reduce((maxIndex, currentValue, currentIndex, array) => {
            return currentValue > array[maxIndex] ? currentIndex : maxIndex;
        }, 0);
    }
    probabilities_boxplot(probs, labels, true_labels, index) {
        // Map labels to colors for consistent coloring in plots
        var colorIndices = labels.map((label, i) => this.indexToColor(i));
        const num_columns = probs[0].length;
        let traces = [];

        // Create subsets of probabilities based on the true labels
        let subsets = {};
        true_labels.forEach((true_label, i) => {
            if (!(true_label in subsets)) {
                subsets[true_label] = [];
            }
            subsets[true_label].push(probs[i]);
        });

        // Generate box plots for each true label class
        for (let true_label in subsets) {
            let subset = subsets[true_label];
            for (let j = 0; j < num_columns; j++) {
                let data = subset.map(item => item[j]);
                traces.push({
                    type: 'box',
                    name: `${true_label}`,
                    marker: {
                        color: colorIndices[j]
                    },
                    y: data
                });
            }
        }

        // Create a div for the plot
        let content = `
            <div class="column is-6" id="probs_box_plot_${index}" style="height: 450px;">
            </div>
        `;
        $("#tabs_info li[data-index='" + index + "'] #results_" + index + "").append(content);

        // Plot the box plots using Plotly
        Plotly.newPlot("probs_box_plot_" + index, traces, {
            yaxis: {
                zeroline: false
            },
            boxmode: 'group'
        });
    }
    probablities_violin_plot(probs, classes, labels) {
        var colorIndices = labels.map((label, i) => this.indexToColor(i));
        const arrayColumn = (arr, n) => arr.map(x => x[n]);
        const num_columns = probs[0].length
        let traces = []
        for (let i = 0; i < num_columns; i++) {
            traces.push({
                name: classes[i],
                type: 'violin',
                y: arrayColumn(probs, i),
                points: 'none',
                box: {
                    visible: true
                },
                boxpoints: false,
                line: {
                    color: colorIndices[i]
                },
                fillcolor: colorIndices[i],
                opacity: 0.6,
                meanline: {
                    visible: true
                },

            });
        }
        var layout = {
            title: "Violin Plot",
            yaxis: {
                zeroline: false
            }
        }

        Plotly.newPlot('probs_violin_plot', traces, layout);
    }
    async plot_confusion_matrix(y, predictedLabels, labels, uniqueClasses, tab_index) {
        const confusionMatrix = await tfvis.metrics.confusionMatrix(y, predictedLabels);
        let div = document.createElement('div');
        div.classList.add('column');
        div.classList.add('is-12');
        div.setAttribute("id", "result_number_" + tab_index);
        let metric = await metrics(y.arraySync(), predictedLabels.arraySync(), uniqueClasses)
        let info = metric[0]
        console.log(labels);
        console.log(confusionMatrix);
        let len = confusionMatrix[0].length
        let preceissions = [];
        let supports = [];
        let recalls = [];
        let f1s = [];
        for (let j = 0; j < len; j++) {
            preceissions.push(parseFloat(info[0][j].toFixed(2)))
        }
        for (let j = 0; j < len; j++) {
            recalls.push(parseFloat(info[1][j].toFixed(2)))
        }
        for (let j = 0; j < len; j++) {
            f1s.push(parseFloat(info[2][j].toFixed(2)))
        }
        for (let j = 0; j < len; j++) {
            supports.push(parseFloat(info[3][j].toFixed(2)))
        }
        div.innerHTML =
            `<div class="column is-12">

            <h5 class="subtitle mb-1">Accuracy: ${metric[3].toFixed(2)}</h5>
            <span class="subtitle mr-2">F1 micro: ${metric[1].toFixed(2)}</span>
            <span class="subtitle">F1 macro: ${metric[2].toFixed(2)}</span>
            </div>`
            ;
        $("#tabs_info li[data-index='" + tab_index + "'] #results_" + tab_index + "").append(div);
        $("#tabs_info li[data-index='" + tab_index + "'] #results_" + tab_index + "").append(`
        <div class="column is-6" id="confusion_matrix_${tab_index}" style="height:50vh">
        </div>
        `);
        window.tensorflow.dispose(y)
        window.tensorflow.dispose(predictedLabels)
        const metric_labels = ["Precession", "Recall", "F1 score", "Support"]
        labels.push("Precession")
        labels.push("Recall")
        labels.push("F1 score")
        // labels.push("Support")
        confusionMatrix.push(preceissions)
        confusionMatrix.push(recalls)
        confusionMatrix.push(f1s)
        // confusionMatrix.push(supports)
        let items_labels = labels.filter(x => !metric_labels.includes(x))
        // Substring template helper for the responsive labels
        Highcharts.Templating.helpers.substr = (s, from, length) =>
            s.substr(from, length);
        let formatted_matrix = []
        for (let i = 0; i < confusionMatrix.length; i++) {
            const element = confusionMatrix[i];
            for (let j = 0; j < element.length; j++) {
                const item = element[j];
                formatted_matrix.push([j, i, item])
            }
        }
        console.log(confusionMatrix);
        console.log(labels);

        // Create the chart
        Highcharts.chart("confusion_matrix_" + tab_index, {
            credits: {
                enabled: false
            },
            exporting: {
                enabled: true
            },
            chart: {
                type: 'heatmap',
                plotBorderWidth: 1
            },
            title: {
                text: '',
                style: {
                    fontSize: '1em'
                }
            },

            xAxis: [{
                categories: items_labels,
                title: {
                    text: 'Predicted Class'
                }
            }, {
                linkedTo: 0,
                opposite: true,
                tickLength: 0,
                labels: {
                    formatter: function () {
                        var chart = this.chart,
                            each = Highcharts.each,
                            series = chart.series[0],
                            sum = 0,
                            x = this.value;

                        each(series.options.data, function (p, i) {
                            if (p[0] === x) {
                                if (p[1] < uniqueClasses.length) {
                                    sum += p[2];
                                }
                            }
                        });

                        return sum;
                    }
                }
            }],
            yAxis: [{
                categories: labels,
                title: {
                    text: 'Actual Class'
                },
                reversed: true, endOnTick: false
            }, {
                linkedTo: 0,
                opposite: true,
                tickLength: 0,
                labels: {
                    formatter: function () {
                        var chart = this.chart,
                            each = Highcharts.each,
                            series = chart.series[0],
                            sum = 0,
                            x = this.value;
                        each(series.options.data, function (p, i) {
                            if (p[1] === x) {
                                if (p[1] < uniqueClasses.length) {
                                    sum += p[2];
                                }

                            }
                        });
                        return sum;
                    }
                },
                title: null
            }],
            colorAxis: {
                min: 0,
                minColor: '#FFFFFF',
                maxColor: Highcharts.getOptions().colors[0]
            },

            legend: {
                align: 'right',
                layout: 'vertical',
                margin: 0,
                verticalAlign: 'top',
                y: 25,
                symbolHeight: 280
            },

            tooltip: {
                formatter: function () {
                    var totalCount = this.series.data.reduce(function (acc, cur) {
                        return acc + cur.value;
                    }, 0);
                    var count = this.point.value;
                    var percentage = ((count / totalCount) * 100).toFixed(2);
                    return '<b>Actual: </b>' + this.series.yAxis.categories[this.point.y] +
                        '<br><b>Predicted: </b>' + this.series.xAxis.categories[this.point.x] +
                        '<br><b>Count: </b>' + count +
                        '<br><b>Percentage: </b>' + percentage + '%';
                }
            },
            series: [{
                name: 'Sales per employee',
                borderWidth: 1,
                data: formatted_matrix,
                dataLabels: {
                    enabled: true,
                    useHTML: true,
                    color: '#000000',
                    formatter: function () {
                        var totalCount = this.series.data.reduce(function (acc, cur) {
                            return acc + cur.value;
                        }, 0);
                        var count = this.point.value;
                        var skip = this.point.index >= this.series.data.length - (3 * uniqueClasses.length);

                        if (!skip) {
                            var percentage = ((count / totalCount) * 100).toFixed(2);
                            return '<p style="margin:auto; text-align:center;">' + count + '<br/>(' + percentage + '%)</p> ';
                        } else {
                            return '<p style="margin:auto; text-align:center;">' + count + '</p>';
                        }
                    }
                }
            }],
            responsive: {
                rules: [{
                    condition: {
                        maxWidth: 500
                    },
                    chartOptions: {
                        yAxis: {
                            labels: {
                                format: '{substr value 0 1}'
                            }
                        }
                    }
                }]
            }
        });
        return confusionMatrix
    }
    plot_regularization(weights, alphas, names, tab_index) {
        let content = `
                    <div class="column is-6" id="regularization_${tab_index}" style="height: 40vh;">
                    </div>
    `
        $("#tabs_info li[data-index='" + tab_index + "'] #results_" + tab_index + "").append(content);

        let serieses = []
        for (let i = 0; i < names.length; i++) {
            serieses.push({
                name: names[i],
                data: weights.map(m => m[i])
            })
        }
        const alphas_formatted = [];
        for (let i = 0; i < alphas.length; i++) {
            alphas_formatted.push(alphas[i].toFixed(2));
        }
        Highcharts.chart("regularization_" + tab_index, {

            title: {
                text: '',
            },
            yAxis: {
                title: {
                    text: 'Coefficients'
                }
            },
            xAxis: {
                title: {
                    text: 'penalty weight'
                },
                categories: alphas_formatted,
            },
            legend: {
                layout: 'vertical',
                align: 'right',
                verticalAlign: 'middle'
            },

            plotOptions: {
                series: {
                    label: {
                        connectorAllowed: false
                    },
                }
            },
            series: serieses,
            responsive: {
                rules: [{
                    condition: {
                        maxWidth: 500
                    },
                    chartOptions: {
                        legend: {
                            layout: 'horizontal',
                            align: 'center',
                            verticalAlign: 'bottom'
                        }
                    }
                }]
            }
        });
    }
    yhat_plot(y_test, predictions, tab_index) {
        Plotly.newPlot('regression_y_yhat_' + tab_index, [{
            x: y_test,
            y: predictions,
            type: 'scatter',
            name: "y",
            mode: 'markers',
        }, {
            x: y_test,
            y: y_test,
            mode: 'lines',
            type: 'scatter',
            line: { color: 'red', dash: 'dash' },
            name: 'y=x '
        }], {
            title: "y vs y&#770;",
            margin: {
                l: 20,
                r: 20,
                b: 20,
                t: 30,
                pad: 5
            }, plot_bgcolor: "#E5ECF6"
        }, { responsive: true });
    }

    scatterplot_matrix_display(dataset, columns, labels) {
        const target = document.getElementById("target").value;
        console.log(target);
        let is_classification = document.getElementById(target).value !== FeatureCategories.Numerical;
        if (is_classification) {
            let unique_labels = [...new Set(labels)];
            if (unique_labels.length == 2) {
                labels = labels.map(label => label[0].toString() === "0" ? 'positive' : 'negative')
            } else {
                labels = labels.map(label => 'class.' + label[0].toString())
            }
            var data = {
                "y": {
                    "data": dataset,
                    "smps": columns,

                },
                "z": {
                    "Classes": labels,
                }
            }


            var config = {
                "legendColumns": labels.length,
                "legendHorizontalJustification": null,
                "legendPosition": "top",
                "broadcast": "true",
                "colorBy": "Classes",
                "graphType": "Scatter2D",
                "layoutAdjust": "true",
                "scatterPlotMatrix": true,
                "scatterPlotMatrixType": "all",
                "theme": "CanvasXpress"
            }
        } else {
            var data = {
                "y": {
                    "data": dataset,
                    "smps": columns,
                },
                "z": {
                }
            };
            data['z'][target] = labels
            var config = {
                autoScaleFont: true,
                fontSize: 8,
                "broadcast": "true",
                "colorBy": target,
                "graphType": "Scatter2D",
                "layoutAdjust": "true",
                "scatterPlotMatrix": true,
                "scatterPlotMatrixType": "lower",
                "theme": "CanvasXpress"
            }
        }
        new CanvasXpress("canvasId", data, config);
    }
}
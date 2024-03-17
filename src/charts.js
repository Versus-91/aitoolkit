import {
    getNumbers, getDataset, getClasses
} from 'ml-dataset-iris';
import Plotly from 'plotly.js-dist';
import { PCA } from 'ml-pca';
import { binarize } from './utils'
import * as tfvis from '@tensorflow/tfjs-vis';
import * as ss from "simple-statistics"
import { schemeAccent, schemeCategory10 } from 'd3-scale-chromatic';
import { scaleLinear, } from 'd3-scale';

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
    draw_categorical_barplot(column_values, target, title) {
        const key = title + "- barplot";
        $("#categories_barplots").append(`<div class="column is-6"style="height:40vh;" id="${key}"></div>`)
        const countOccurrences = column_values.reduce((acc, val, i) => {
            acc[val] = (acc[val] || 0) + 1;
            return acc;
        }, {});
        const countArray = Object.entries(countOccurrences).map(([value, count]) => ({ value: value, count }));
        countArray.sort((a, b) => b.count - a.count);
        const top5 = countArray.slice(0, 5);
        console.log(top5);
        var top_categories_trace = {
            x: top5.map(m => m.value),
            y: top5.map(m => m.count),
            width: 0.3,
            name: 'SF Zoo',
            type: 'bar'
        };
        var data = [top_categories_trace];
        var layout = { barmode: 'stack', title: title, bargap: 0.05 };

        Plotly.newPlot(key, data, layout);

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
    plot_tsne(data, labels) {
        document.getElementById("dimensionality_reduction_panel").style.display = "block"
        console.assert(Array.isArray(data));

        var opt = {}
        opt.epsilon = 10;
        opt.perplexity = 30;
        opt.dim = 2;
        var tsne = new window.tsnejs.tSNE(opt);
        tsne.initDataRaw(data);
        for (var k = 0; k < 500; k++) {
            tsne.step(); // 
        }
        var Y = tsne.getSolution();
        let traces = []
        if (labels.length > 0) {
            labels = labels.flat()
            var uniqueLabels = [...new Set(labels)];
            var colorIndices = labels.map(label => uniqueLabels.indexOf(label));
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
                return {
                    'x': item[0],
                    'y': item[1]
                }
            })
            traces.push({
                x: points.map(m => m.x),
                y: points.map(m => m.y),
                mode: 'markers+text',
                type: 'scatter',
                marker: { size: 4 }
            })
        }

        var layout = {
            showlegend: true,
            legend: { "orientation": "h" },
            title: 't-SNE plot'
        };
        Plotly.newPlot('tsne', traces, layout, { responsive: true });
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

    draw_kde(dataset, column, target_name, bandwidth = "nrd", is_classification = false, redrawing = false) {
        let items = dataset.column(column).values;
        var kde = ss.kernelDensityEstimation(items, "gaussian", bandwidth);
        let default_bandwidth = this.nrd(items).toFixed(2);
        let raw_values = dataset.loc({ columns: [column, target_name] });
        let uniqueLabels = [...new Set(raw_values.column(target_name).values)];
        let column_values = raw_values.values;
        let subsets = [];
        var colorIndices = uniqueLabels.map(label => this.indexToColor(uniqueLabels.indexOf(label)));
        if (!is_classification) {
            subsets.push(raw_values.column(column).values);
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
        newColumn.className = "column is-4";
        newColumn.setAttribute("id", column + '-kde-plot');
        if (!redrawing) {
            $("#container").append(
                `<div class="column is-4 " >
                    <div id="${column + '-kde-plot'}"> </div>
                    <div class="field has-addons my-1">
                        <div class="control">
                            <input class="input is-small" type="number"  min="0" id="${column + '-kde'}" value="${default_bandwidth}">
                            </div>
                            <div class="control">
                            <a class="button is-success is-small" id="${column + '-kde-button'}">
                                Apply
                            </a>
                        </div>
                    </div>
                </div>`
            );
        }

        var current_class = this;
        document.getElementById(column + '-kde-button').addEventListener("click", function () {
            var newBandwidth = document.getElementById(column + '-kde').value;
            current_class.draw_kde(dataset, column, target_name, parseFloat(newBandwidth), is_classification = true, redrawing = true)
        });
        let container_id = column + '-kde-plot';
        let items_range = raw_values.column(column).values
        let minValue = Math.min(...items_range);
        let maxValue = Math.max(...items_range);
        items_range.push(minValue - parseFloat(default_bandwidth))
        items_range.push(maxValue + parseFloat(default_bandwidth))
        var breaks = ss.equalIntervalBreaks(items_range, 100);

        let allData = [];

        // Loop through subsets to generate data for all subsets
        for (let i = 0; i < subsets.length; i++) {
            let ys = [];
            var kde = ss.kernelDensityEstimation(subsets[i], "gaussian", bandwidth);
            let data = [];
            breaks.forEach((item) => {
                ys.push(kde(item, bandwidth));
                data.push([item, ys[ys.length - 1]]);
            });
            allData.push(data);
        }

        let animationDuration = 4000;

        Highcharts.chart(container_id, {
            credits: {
                enabled: false
            },
            chart: {
                type: "spline",
                animation: true
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

    }

    draw_classification_pca(dataset, labels, missclassifications, uniqueLabels, size = 4, color_scale = "Jet") {
        const pca = new PCA(dataset, { center: true, scale: true });
        var colorIndices = labels.map(label => this.indexToColor(uniqueLabels.indexOf(label)));
        const pca_data = pca.predict(dataset, { nComponents: 2 })
        let x = []
        let y = []
        let x_error = []
        let y_error = []
        pca_data.data.forEach((element, i) => {
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
                size: size,
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
        Plotly.newPlot('pca-results', data, {
            showlegend: true,
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
        }, { responsive: true });

    }
    draw_pca(dataset, labels, size = 4, color_scale = "Jet") {
        document.getElementById("dimensionality_reduction_panel").style.display = "block"
        document.getElementById("pca-1").innerHTML = ""
        document.getElementById("pca-2").innerHTML = ""
        document.getElementById("pca-3").innerHTML = ""
        const pca = new PCA(dataset, { center: true, scale: true });

        labels = labels.flat()
        var uniqueLabels = [...new Set(labels)];
        var colorIndices = labels.map(label => this.indexToColor(uniqueLabels.indexOf(label)));

        const pca_data = pca.predict(dataset, { nComponents: 3 })

        let x = []
        let y = []
        let x1 = []
        let y1 = []
        let x2 = []
        let y2 = []

        pca_data.data.forEach(element => {
            x.push(element[0])
            y.push(element[1])
            x1.push(element[0])
            y1.push(element[2])
            x2.push(element[1])
            y2.push(element[2])
        });

        var trace1 = []
        var trace2 = []
        var trace2 = []

        for (let i = 0; i < labels.length; i++) {
            const label = labels[i];
            trace1.push({
                x: x,
                y: y,
                text: labels,
                mode: 'markers',
                type: 'scatter',
                marker: {
                    size: size,
                    color: colorIndices,
                },
            });
            trace2.push({
                x: x1,
                y: y1,
                text: labels,
                mode: 'markers',
                type: 'scatter',
                marker: {
                    size: size,
                    color: colorIndices,
                },
            });
            trace3.push({
                x: x2,
                y: y2,
                text: labels,
                mode: 'markers',
                type: 'scatter',
                marker: {
                    size: size,
                    color: colorIndices,
                    colorscale: color_scale,
                },
            });
        }

        Plotly.newPlot('pca-1', [trace1], {
            showlegend: true,
            legend: { "orientation": "h" },
            xaxis: {
                title: 'PC1'
            },
            yaxis: {
                title: 'PC2'
            }
        }, { responsive: true });
        Plotly.newPlot('pca-2', [trace2], {
            showlegend: true,
            legend: { "orientation": "h" },
            xaxis: {
                title: 'PC1'
            },
            yaxis: {
                title: 'PC3'
            }
        }, { responsive: true });
        Plotly.newPlot('pca-3', [trace3], {
            showlegend: true,
            legend: { "orientation": "h" },
            xaxis: {
                title: 'PC2'
            },
            yaxis: {
                title: 'PC3'
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
    probablities_boxplot(probs, classes, labels) {

        var colorIndices = labels.map((label, i) => this.indexToColor(i));
        const num_columns = probs[0].length
        let traces = [];
        let subsets = {};
        if (labels.length > 1) {
            probs.forEach((prob) => {
                let index = this.argmax(prob);
                if (!(index in subsets)) {
                    subsets[index] = [];
                    subsets[index].push(prob);
                } else {
                    subsets[index].push(prob);
                }
            });
        }
        for (let i = 0; i < num_columns; i++) {
            let subset = subsets[i]
            for (let j = 0; j < num_columns; j++) {
                if (!!subset) {
                    traces.push({
                        type: 'box',
                        name: classes[i],
                        marker: {
                            color: colorIndices[j]
                        },
                        y: subset.map(item => item[j])
                    })
                }
            }
        }

        Plotly.newPlot("probs_box_plot", traces, {
            yaxis: {
                zeroline: false
            }, boxmode: 'group'
        })
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
}
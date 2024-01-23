import {
    getNumbers, getDataset, getClasses
} from 'ml-dataset-iris';
import Plotly from 'plotly.js-dist';
import { PCA } from 'ml-pca';
import { binarize } from './utils'
import * as tfvis from '@tensorflow/tfjs-vis';
import * as ss from "simple-statistics"
import { schemeAccent, schemeCategory10 } from 'd3-scale-chromatic';
export default class ChartController {
    constructor(data_processor) {
        this.data_processor = data_processor
        this.color_scheme = schemeCategory10;
    }

    classification_target_chart(values, labels, name, container, title = "") {
        var uniqueLabels = [...new Set(labels)];
        var colorIndices = labels.map(label => this.indexToColor(uniqueLabels.indexOf(label)));
        var trace2 = {
            y: values,
            x: labels,
            type: 'bar',
            width: 0.3,
            xaxis: 'x2',
            yaxis: 'y2',
            marker: {
                color: colorIndices,
                line: {
                    color: colorIndices,
                    width: 1.5
                }
            }
        };

        var data = [trace2];

        var layout = {
            title: title,
            font: {
                family: 'Raleway, sans-serif'
            },
            showlegend: false,
            xaxis: {
                tickangle: -45
            },
            yaxis: {
                zeroline: false,
                gridwidth: 1
            },
            bargap: 0.05
        };

        Plotly.newPlot(container, data, layout);
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
            legend: {
                y: 0.5,
                yref: 'paper',
                font: {
                    family: 'Arial, sans-serif',
                    size: 20,
                    color: 'grey',
                }
            },
            title: 't-SNE plot'
        };
        Plotly.newPlot('tsne', traces, layout);
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
    redraw_kde(dataset, column, bandwidth) {
        document.getElementById(container_id).innerHTML = "";

        Plotly.purge(container_id);

        let traces = [];

        let items = dataset.column(column).values;

        var kde = ss.kernelDensityEstimation(items, "gaussian", bandwidth);
        let default_bandwidth = this.nrd(items).toFixed(2);

        var breaks = ss.equalIntervalBreaks(items, 100);

        let ys = breaks.map((item) => kde(item, bandwidth));

        traces.push({
            x: breaks,
            y: ys,
            fill: 'tozeroy',
            type: 'scatter',
            xaxis: 'x',
            yaxis: 'y',
            name: column
        });
        var layout = {
            title: column,
            showlegend: false,
            height: 400,
        };
        Plotly.newPlot(container_id, traces, layout);

        // Append input and button elements to the container if they exist
        if (inputElement) {
            document.getElementById(container_id).appendChild(inputElement);
        }
        if (buttonElement) {
            document.getElementById(container_id).appendChild(buttonElement);
        }
    }
    draw_kde(dataset, column, bandwidth = "nrd") {
        let current_class = this;
        let traces = [];
        let items = dataset.column(column).values;
        var kde = ss.kernelDensityEstimation(items, "gaussian", bandwidth);
        let default_bandwidth = this.nrd(items).toFixed(2);

        document.getElementById("kde_panel").style.display = "block";
        var newColumn = document.createElement("div");
        newColumn.className = "column is-4";
        newColumn.setAttribute("id", column + '-kde-clomun');
        var container = document.getElementById("container");
        container.appendChild(newColumn);
        let container_id = column + '-kde-clomun';
        // Create an input element
        var inputElement = document.createElement("input");
        inputElement.setAttribute("class", "input is-small is-2");
        inputElement.setAttribute("placeholder", "bandwidth");
        inputElement.setAttribute("id", column + '-kde');
        inputElement.setAttribute("type", "number");
        inputElement.value = default_bandwidth
        var buttonElement = document.createElement("button");
        buttonElement.setAttribute("class", "button is-primary is-small");
        buttonElement.textContent = "Apply";

        buttonElement.addEventListener("click", function () {
            var newBandwidth = document.getElementById(column + '-kde').value;
            current_class.redraw_kde(dataset, column, parseFloat(newBandwidth));
        });


        var breaks = ss.equalIntervalBreaks(items, 100);
        let ys = [];

        breaks.forEach((item) => {
            ys.push(kde(item, bandwidth));
        });
        traces.push({
            x: breaks,
            y: ys,
            fill: 'tozeroy',
            type: 'scatter',
            xaxis: 'x',
            yaxis: 'y',
        });
        var layout = {
            showlegend: false, height: 400,
            title: column,
            plot_bgcolor: "#E5ECF6"
        };
        Plotly.newPlot(container_id, traces, layout);
        document.getElementById(container_id).appendChild(inputElement);
        document.getElementById(container_id).appendChild(buttonElement);
    }
    draw_classification_pca(dataset, labels, missclassifications, size = 4, color_scale = "Jet") {
        const pca = new PCA(dataset, { center: true, scale: true });
        var uniqueLabels = [...new Set(labels)];
        var colorIndices = labels.map(label => uniqueLabels.indexOf(label));
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
                colorscale: color_scale,
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
                colorscale: color_scale,
                symbol: 'cross'
            },
        };
        var data = [trace1, trace2];
        Plotly.newPlot('pca-results', data, {
            xaxis: {
                title: 'PCA component 1'
            },
            yaxis: {
                title: 'PCA component 2'
            }
        });

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
        var trace1 = {
            x: x,
            y: y,
            text: labels,
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: size,
                color: colorIndices,
                // colorscale: color_scale,
            },
        };
        var trace2 = {
            x: x1,
            y: y1,
            text: labels,
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: size,
                color: colorIndices,
                // colorscale: color_scale,
            },
        };
        var trace3 = {
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
        };
        Plotly.newPlot('pca-1', [trace1], {
            xaxis: {
                title: 'PCA component 1'
            },
            yaxis: {
                title: 'PCA component 2'
            }
        });
        Plotly.newPlot('pca-2', [trace2], {
            xaxis: {
                title: 'PCA component 1'
            },
            yaxis: {
                title: 'PCA component 3'
            }
        });
        Plotly.newPlot('pca-3', [trace3], {
            xaxis: {
                title: 'PCA component 2'
            },
            yaxis: {
                title: 'PCA component 3'
            }
        });
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

    probablities_boxplot(probs, classes) {
        const num_columns = probs[0].length
        let traces = []
        for (let i = 0; i < num_columns; i++) {
            traces.push({
                type: 'box',
                name: classes[i],
                y: probs.map(item => item[i])
            })
        }
        Plotly.newPlot("probs_box_plot", traces)
    }
    probablities_violin_plot(probs, classes) {
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
                    color: 'black'
                },
                fillcolor: '#8dd3c7',
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
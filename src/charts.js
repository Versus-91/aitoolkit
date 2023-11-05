import {
    getNumbers, getDataset, getClasses
} from 'ml-dataset-iris';
import Plotly from 'plotly.js-dist';
import { PCA } from 'ml-pca';
import { binarize } from './utils'
import * as tfvis from '@tensorflow/tfjs-vis';

export default class ChartController {
    constructor(data_processor) {
        this.data_processor = data_processor
    }
    roc_chart(container, tprs, fprs) {
        var trace = {
            x: fprs,
            y: tprs,
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
    plot_tsne(data, lables) {
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
        if (lables.length > 0) {
            var uniqueLabels = [...new Set(lables.map(m => m[0]))];
            let points_labled = Y.map(function (item, i) {
                return {
                    lable: lables[i][0],
                    'x': item[0],
                    'y': item[1]
                }
            }
            )
            uniqueLabels.forEach((lable, i) => {
                var items_for_lable = points_labled.filter(m => m.lable === lable)
                traces.push({
                    x: items_for_lable.map(m => m.x),
                    y: items_for_lable.map(m => m.y),
                    mode: 'markers+text',
                    type: 'scatter',
                    name: lable,
                    marker: { size: 5 }
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
                marker: { size: 5 }
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
            title: 'T-sne plot'
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
    draw_kde(items, container) {
        d3.json("./faithful.json", function (error, faithful) {
            if (error) throw error;

            var svg = d3.select("svg")

            let min = Math.min(...faithful) - 1
            let max = Math.max(...faithful) + 1

            const width = +svg.attr("width")
            const height = +svg.attr("height")

            const margin = { top: 20, right: 30, bottom: 30, left: 40 };
            var n = faithful.length
            var x = d3.scaleLinear()
                .domain([min, max])
                .range([margin.left, width - margin.right]);
            density = kernelDensityEstimator(kernelEpanechnikov(0.1), x.ticks(10))(faithful);
            var y = d3.scaleLinear()
                .domain([0, d3.max(density, d => d[1])])
                .range([height - margin.bottom, margin.top]);


            svg.append("g")
                .attr("class", "axis axis--x")
                .attr("transform", "translate(0," + (height - margin.bottom) + ")")
                .call(d3.axisBottom(x))
                .append("text")
                .attr("x", width - margin.right)
                .attr("y", -6)
                .attr("fill", "#000")
                .attr("text-anchor", "end")
                .attr("font-weight", "bold");

            svg.append("g")
                .attr("class", "axis axis--y")
                .attr("transform", "translate(" + margin.left + ",0)")
                .call(d3.axisLeft(y));
            svg.insert("g", "*")
                .attr("fill", "#bbb")
                .selectAll("rect")
                .enter().append("rect")
                .attr("x", function (d) { return x(d.x0) + 1; })
                .attr("y", function (d) { return y(d.length / n); })
                .attr("width", function (d) { return x(d.x1) - x(d.x0) - 1; })
                .attr("height", function (d) { return y(0) - y(d.length / n); });

            svg.append("path")
                .datum(density)
                .attr("fill", "none")
                .attr("stroke", "#000")
                .attr("stroke-width", 1.5)
                .attr("stroke-linejoin", "round")
                .attr("d", d3.line()
                    .curve(d3.curveBasis)
                    .x(function (d) { return x(d[0]); })
                    .y(function (d) { return y(d[1]); }));
        });

        function kernelDensityEstimator(kernel, X) {
            return function (V) {
                return X.map(function (x) {
                    return [x, d3.mean(V, function (v) { return kernel(x - v); })];
                });
            };
        }

        function kernelEpanechnikov(k) {
            return function (v) {
                return Math.abs(v /= k) <= 1 ? 0.75 * (1 - v * v) / k : 0;
            };
        }
    }
    draw_classification_pca(dataset, labels, missclassifications) {
        console.log(dataset);
        const pca = new PCA(dataset, { center: true, scale: true });
        var uniqueLabels = [...new Set(labels)];
        var colorscale = uniqueLabels.map((label, index) => {
            var hue = (360 * index) / uniqueLabels.length;
            return `hsl(${hue}, 100%, 50%)`;
        });
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
            text: labels,
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 10,
                color: colorIndices,
                colorscale: [colorscale],
                symbol: 'circle'
            },
        };
        var trace2 = {
            x: x_error,
            y: y_error,
            text: labels,
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 10,
                color: colorIndices,
                colorscale: [colorscale],
                symbol: 'cross'
            },
        };
        var data = [trace1, trace2];
        Plotly.newPlot('pca-1', data, {
            xaxis: {
                title: 'PCA component 1'
            },
            yaxis: {
                title: 'PCA component 2'
            }
        });

    }
    draw_pca(dataset, labels) {
        const pca = new PCA(dataset, { center: true, scale: true });
        var uniqueLabels = [...new Set(labels)];
        var colorscale = uniqueLabels.map((label, index) => {
            var hue = (360 * index) / uniqueLabels.length;
            return `hsl(${hue}, 100%, 50%)`;
        });
        var colorIndices = labels.map(label => uniqueLabels.indexOf(label));
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
                size: 10,
                color: colorIndices,
                colorscale: [colorscale],
            },
        };
        var trace2 = {
            x: x1,
            y: y1,
            text: labels,
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 10,
                color: colorIndices,
                colorscale: [colorscale],
            },
        };
        var trace3 = {
            x: x2,
            y: y2,
            text: labels,
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 10,
                color: colorIndices,
                colorscale: [colorscale],
            },
        };
        var data = [trace1];
        Plotly.newPlot('pca-1', data, {
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
}
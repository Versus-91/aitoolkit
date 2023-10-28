import {
    getNumbers, getDataset
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
                if (i === 0) {
                    console.log("probs", probs.dataSync());
                    console.log("thresh preds", threshPredictions.arraySync());
                    console.log("targets", targets.arraySync());
                    console.log("true positive rate", tpr, fpr);
                }
                // Accumulate to area for AUC calculation.
                if (i > 0) {
                    area += (tprs[i] + tprs[i - 1]) * (fprs[i - 1] - fprs[i]) / 2;
                }
            }
            return [area, fprs, tprs];
        });
    }
    draw_kde(points, items, container) {
        console.log(container);
        let dataSource = items;
        let xiData = [];
        let animationDuration = 4000;
        items.forEach(element => {
            console.log(points.next(element));
        });
        Highcharts.chart("roc2", {
            chart: {
                type: "spline",
                animation: true
            },
            title: {
                text: "Gaussian Kernel Density Estimation (KDE)"
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
                    color: "#ff8d1e",
                    pointStart: xiData[0],
                    animation: {
                        duration: animationDuration
                    }
                }
            },
            series: [
                {
                    name: "KDE",
                    dashStyle: "solid",
                    lineWidth: 2,
                    color: "#1E90FF",
                    data: data
                }
            ]
        });
    }
    draw_pca() {
        const dataset = getNumbers();
        const pca = new PCA(dataset);
        /*
        [ 0.9246187232017269,
        0.05306648311706785,
        0.017102609807929704,
        0.005212183873275558 ]
        */
        const newPoints = [
            [4.9, 3.2, 1.2, 0.4],
            [5.4, 3.3, 1.4, 0.9],
        ];
        let rows = pca.getEigenvectors().data.map((eigenvectorForPCs, variableIndex) => {
            const variable = Object.keys(dataset[0])[variableIndex];
            const row = {
                Variable: variable
            };
            eigenvectorForPCs.forEach((value, pcIndex) => {
                row[`PC${pcIndex + 1}`] = value;
            });
            return row;
        })
        console.log(rows)

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
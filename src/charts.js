import * as tfvis from "@tensorflow/tfjs-vis";
import { Matrix } from 'ml-matrix';
import { PCA } from 'ml-pca';
let viz = tfvis
export default class ChartController {
    constructor(data_processor) {
        this.data_processor = data_processor
    }
    draw_kde(data, bandwidth = 0.5) {
        const kde = this.data_processor.kernelDensityEstimation(data, bandwidth);

        // Generate x and y values for the KDE plot
        const xValues = [];
        const yValues = [];
        for (let x = Math.min(...data); x <= Math.max(...data); x += 0.1) {
            xValues.push(x);
            yValues.push(kde(x));
        }

        // Create a Plotly trace for the KDE plot
        const trace = {
            x: xValues,
            y: yValues,
            mode: 'lines',
            type: 'scatter',
        };

        // Create the layout for the plot
        const layout = {
            title: 'Kernel Density Estimation (KDE) Plot',
            xaxis: { title: 'X' },
            yaxis: { title: 'Density' },
        };

        // Create the Plotly plot
        Plotly.newPlot('kde-plot', [trace], layout);
    }
    draw_pca() {
        // Sample data
        const data = new Matrix([
            [2, 3, 4, 5],
            [4, 1, 5, 8],
            [7, 6, 9, 8],
            [10, 12, 11, 9],
            [13, 14, 16, 11],
        ]);

        // Perform PCA
        const pca = new PCA(data);
        const scores = pca.getCenteredData();

        // Extract the first two principal components (PC1 and PC2)
        const pc1 = scores.getColumn(0);
        const pc2 = scores.getColumn(1);

        // Create a scatter plot using Plotly
        const trace = {
            x: pc1,
            y: pc2,
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 10,
                color: 'blue',
            },
        };

        const layout = {
            xaxis: { title: 'PC1' },
            yaxis: { title: 'PC2' },
            title: 'PCA Plot',
        };

        const dataToPlot = [trace];

        Plotly.newPlot('pca-plot', dataToPlot, layout);
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
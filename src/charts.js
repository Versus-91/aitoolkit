import {
    getNumbers, getDataset
} from 'ml-dataset-iris';
import { PCA } from 'ml-pca';
export default class ChartController {
    constructor(data_processor) {
        this.data_processor = data_processor
    }
    draw_kde(data) {
        let dataSource = [93, 93, 96, 100, 101, 102, 102];
        let xiData = [];
        let animationDuration = 4000;
        let range = 20,
            startPoint = 88;
        for (i = 0; i < range; i++) {
            xiData[i] = startPoint + i;
        }
        let data = [];

        function GaussKDE(xi, x) {
            return (1 / Math.sqrt(2 * Math.PI)) * Math.exp(Math.pow(xi - x, 2) / -2);
        }

        let N = dataSource.length;

        for (i = 0; i < xiData.length; i++) {
            let temp = 0;
            for (j = 0; j < dataSource.length; j++) {
                temp = temp + GaussKDE(xiData[i], dataSource[j]);
            }
            data.push([xiData[i], (1 / N) * temp]);
        }

        Highcharts.chart("container", {
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
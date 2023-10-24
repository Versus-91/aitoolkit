
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
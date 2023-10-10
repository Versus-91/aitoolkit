"use strict";
function printPapaObject(papa) {
    var header = "";
    var tbody = "";
    for (var p in papa.meta.fields) {
        header += "<th>" + papa.meta.fields[p] + "</th>";
    }
    for (var i = 0; i < papa.data.length; i++) {
        var row = "";
        for (var z in papa.data[i]) {
            row += "<td>" + papa.data[i][z] + "</td>";
        }
        tbody += "<tr>" + row + "</tr>";
    }
    //build a table
    $("output").html(
        '<table class="table"><thead>' +
        header +
        "</thead><tbody>" +
        tbody +
        "</tbody></table>"
    );
}

function handleFileSelect(evt) {
    var file = evt.target.files[0];

    Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        complete: function (results) {
            console.log(results);
            printPapaObject(results);
        }
    });
}
function scatterplot() {
    cars = res.data.map((car) => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower
    }))
    const values = cars.map((d) => ({
        x: d.horsepower,
        y: d.mpg,
    }));
    tfvis.render.scatterplot(
        { name: 'Horsepower v MPG' },
        { values: values },
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        }
    );
}
function printPapaObject(papa) {
    var header = "";
    var tbody = "";
    for (var p in papa.meta.fields) {
        header += "<th>" + papa.meta.fields[p] + "</th>";
    }
    for (var i = 0; i < papa.data.length; i++) {
        var row = "";
        for (var z in papa.data[i]) {
            row += "<td>" + papa.data[i][z] + "</td>";
        }
        tbody += "<tr>" + row + "</tr>";
    }
    //build a table
    document.querySelector("output").innerHTML =
        '<table class="table is-bordered"><thead>' +
        header +
        "</thead><tbody>" +
        tbody +
        "</tbody></table>"
        ;
}

function handleFileSelect(evt) {
    var file = evt.target.files[0];
    Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        complete: function (results) {
            console.log(results.data);
            printPapaObject(results);
            renderChart("chart", results.data, "PetalLengthCm", {
                title: "Overall material and finish quality (0-10)",
                xLabel: "Species"
            });
        }
    });

}
const createDataSets = (data, features, categoricalFeatures, testSize) => {
    const X = data.map(r =>
        features.flatMap(f => {
            if (categoricalFeatures.has(f)) {
                return oneHot(!r[f] ? 0 : r[f], VARIABLE_CATEGORY_COUNT[f]);
            }
            return !r[f] ? 0 : r[f];
        })
    );

    const X_t = normalize(tf.tensor2d(X));

    const y = tf.tensor(data.map(r => (!r.SalePrice ? 0 : r.SalePrice)));

    const splitIdx = parseInt((1 - testSize) * data.length, 10);

    const [xTrain, xTest] = tf.split(X_t, [splitIdx, data.length - splitIdx]);
    const [yTrain, yTest] = tf.split(y, [splitIdx, data.length - splitIdx]);

    return [xTrain, xTest, yTrain, yTest];
};
async function trainNewModel() {
    this.linearmodel = tf.sequential();
    this.linearmodel.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    this.linearmodel.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

    const xs = tf.tensor1d([1, 2]);
    const ys = tf.tensor1d([2, 4]);
    await this.linearmodel.fit(xs, ys);
    console.log('training is complete');
}
function renderChart(container, data, column, config) {
    const columnData = data.map(r => r[column]);

    const columnTrace = {
        name: column,
        x: columnData,
        type: "histogram",
        opacity: 0.7,
        marker: {
            color: "dodgerblue"
        }
    };
    Plotly.newPlot(container, [columnTrace], {
        xaxis: {
            title: config.xLabel,
            range: config.range
        },
        yaxis: { title: "Count" },
        title: config.title
    });
};
function linear_test(data) {
    // Inputs
    const xs = tf.tensor([-1, 0, 1, 2, 3, 4]); 1
    // Answers we want from inputs
    const ys = tf.tensor([-4, -2, 0, 2, 4, 6]);
    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: 1,
        units: 1
    }));
    model.compile({
        optimizer: "sgd",
        loss: "meanSquaredError"
    });

    let log = model.summary();
    // Train
    model.fit(xs, ys, { epochs: 300 }).then(history => {
        5
        const inputTensor = tf.tensor([10]);
        const answer = model.predict(inputTensor); 6
        console.log(`10 results in ${Math.round(answer.dataSync())}`);
        // cleanup
        tf.dispose([xs, ys, model, answer, inputTensor]); 7
    });
}
document.getElementById("parseCVS").addEventListener("change", handleFileSelect)
document.getElementById("test").addEventListener("click", linear_test)


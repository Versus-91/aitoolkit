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
            //printPapaObject(results);
            renderDatasetStats(results.data)
            renderChart("chart", results.data, "PetalLengthCm", {
                title: "",
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

function renderDatasetStats(data) {
    var header = "";
    var tbody = "";
    const fileds = ["Metric", "Min", "Max", "Median", "Standard deviation", "p-value"]
    for (var p in fileds) {
        header += "<th>" + fileds[p] + "</th>";
    }
    const invalidColumns = ["Id", "Species"];
    for (const key in data[0]) {
        if (!invalidColumns.includes(key)) {
            let row = "";
            const formattedData = data.map(row => {
                return row[key]
            }).filter(function (item) {
                return typeof item === "number"
            });

            console.log(formattedData);
            const min = Math.min(...formattedData)
            console.log(min, key);
            const max = Math.max(...formattedData)
            row += "<td>" + key + "</td>";
            row += "<td>" + min + "</td>";
            row += "<td>" + max + "</td>";
            row += "<td>" + ss.mean(formattedData) + "</td>";
            row += "<td>" + ss.standardDeviation(formattedData) + "</td>";
            row += "<td>" + min + "</td>";
            tbody += "<tr>" + row + "</tr>";
        }
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
        const inputs = [10, 10, 15]
        const inputTensor = tf.tensor(inputs);
        const answer = model.predict(inputTensor);
        const answers = answer.dataSync();
        answers.forEach((element, i) => {
            console.log(inputs[i] + ` results in ${Math.round(answers[i])}`);
        });
        const labels = tf.tensor1d(inputs);
        const predictions = tf.tensor1d(answers);
        tfvis.metrics.confusionMatrix(labels, predictions).then((res) => {
            const container = document.getElementById("confusion-matrix")
            tfvis.render.confusionMatrix(container, {
                values: res,
                tickLabels: ["Healthy", "Diabetic"],
            })
            console.log(JSON.stringify(res, null, 2))
        });
        tf.dispose([xs, ys, model, answer, inputTensor]);
    });
}
document.getElementById("parseCVS").addEventListener("change", handleFileSelect)
document.getElementById("test").addEventListener("click", linear_test)


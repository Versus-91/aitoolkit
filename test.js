const dfd = require("danfojs-node")
let data = {
    "Name": ["Apples", "Mango", "Banana", undefined],
    "Count": [NaN, 5, NaN, 10],
    "Price": [200, 300, 40, 250]
}

let df = new dfd.DataFrame(data)

const missingValuesCount = [];

// Loop through each column and count missing values
df.columns.forEach((col) => {
    const count = df.column(col).values.length
    missingValuesCount.push({ column: col, count: count });
});

// Display missing values count for each column
console.log("Missing values count in each column:");
console.log(missingValuesCount);

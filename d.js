const dfd = require("danfojs-node")
const tf = dfd.tensorflow //Reference to the exported tensorflowjs libraryasync function load_process_data() {

// dfd.readCSV("https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv").then((df) => {
//     //label Encode Name feature
//     let encoder = new dfd.OneHotEncoder()
//     let cols = ["Sex", "Name"]
//     cols.forEach(col => {
//         encoder.fit(df[col])
//         enc_val = encoder.transform(df[col])
//         console.log(enc_val);
//         df.addColumn(col, enc_val, { inplace: true })
//     })

//     df.head().print()
// })

let data = {
    fruits: ['pear', 'mango', "pawpaw", "mango", "bean"],
    Count: [20, 30, 89, 12, 30],
    Country: ["NG", "NG", "GH", "RU", "RU"]
}

let df = new dfd.DataFrame(data)

let dum_df = dfd.getDummies(df, { columns: "fruits", prefix: "fruits_" })
for (let index = 0; index < dum_df.columns.length; index++) {
    const element = dum_df.columns[index];
    if (element.includes("__")) {
        dum_df.drop({ columns: element, inplace: true })
        console.log(element);
        break;
    }
}
dum_df.print()
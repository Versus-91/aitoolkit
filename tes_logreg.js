const dfd = require("danfojs-node")

let data = {
    "Name": ["Apples", "Mango", "Banana", "Pear"],
    "Count": [21, 5, 30, 10],
    "Price": [200, 300, 40, 250]
}

let df = new dfd.DataFrame(data)
let s_df = df.head(2)
s_df.print()
console.log(s_df.values);
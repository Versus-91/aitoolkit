const dfd = require("danfojs-node")

arr_data = [["bval1", 10, 1.2, "test"],
["bval2", 20, 3.45, "train"],
["bval3", 30, 60.1, "train"],
["bval4", 35, 3.2, "test"]]

df = new dfd.DataFrame(arr_data)
console.log(df.columns);
console.log(df.head(2).values)
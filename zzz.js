const dfd = require("danfojs-node")


let datasf = ['pear', 'mango', "pawpaw", "mango", "bean"]
let sf1 = new dfd.Series(datasf)
let dum_df = dfd.getDummies(sf1, { prefix: "fruit" })
dum_df.drop({ columns: [dum_df.columns.find(m => m.includes("fruit_"))], inplace: true })
dum_df.print()
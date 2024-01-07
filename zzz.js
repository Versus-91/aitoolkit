const MLR =  require("ml-regression-multivariate-linear")

const x = [
  [0, 0],
  [1, 2],
  [2, 3],
  [3, 4]
];
// Y0 = X0 * 2, Y1 = X1 * 2, Y2 = X0 + X1
const y = [
  [0,],
  [2],
  [4],
  [6]
];
const mlr = new MLR(x, y);
console.log(mlr.predict([[4, 5],[1 ]]));
console.log(mlr.toJSON());

// [6, 6, 6]
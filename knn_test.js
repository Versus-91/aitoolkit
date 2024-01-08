

var KNN = require("ml-knn")
function squaredEuclidean(p, q) {
  let d = 0;
  for (let i = 0; i < p.length; i++) {
    if (typeof p[i] === "string" && typeof p[i] === "string") {
      if (p[i] === q[i]) {
        d += 0
      } else {
        d += 1
      }
    } else {
      d += (p[i] - q[i]) * (p[i] - q[i]);
    }
    console.log(d)
  }
  return d;
}
function manhatan_distance(p, q) {
  let d = 0;
  for (let i = 0; i < p.length; i++) {
    if (typeof p[i] === "string" && typeof p[i] === "string") {
      if (p[i] === q[i]) {
        d += 0
      } else {
        d += 1
      }
    } else {
      d += Math.abs(p[i] - q[i])
    }
    console.log(d);
  }
  return d;
}
function euclidean(p, q) {
  return Math.sqrt(squaredEuclidean(p, q));
}
var train_dataset = [
  [0, 0, 0, "a"],
  [0, 1, 1, "a"],
  [1, 1, 0, "a"],
  [2, 2, 2, "b"],
  [1, 2, 2, "b"],
  [2, 1, 2, "a"],
];
var train_labels = [0, 0, 0, 1, 1, 1];
var knn = new KNN(train_dataset, train_labels, { k: 2, distance: manhatan_distance }); // consider 2 nearest neighbors
var test_dataset = [
  [0.9, 0.9, 0.9, "a"],
];

var ans = knn.predict(test_dataset);

console.log(ans);

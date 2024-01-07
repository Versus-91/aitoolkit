import KNN from "ml-knn"

import { KNeighborsClassifier } from 'scikitjs'
export default class KNNModel {

    constructor() {
        this.model = null
    }
    async train(x_train, y_train, k = 3, distance = manhatan_distance) {
        this.model = new KNN(x_train, y_train, { k: k, distance: distance });
    }
    predict(x_test) {
        if (this.model === null || this.model === undefined) {
            throw "model not found."
        }
        return this.model.predict(x_test)
    }
}
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
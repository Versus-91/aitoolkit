import KNN from 'ml-knn';

export default class KNNModel {
    constructor() {
        this.model = null

    }
    train(x_train, y_train, k = 3) {
        this.model = new KNN(x_train, y_train, { k: k });
    }
    evaluate(x_test) {
        if (this.model === null || this.model === undefined) {
            throw "model not found."
        }
        var ans = this.model.predict(x_test);
        return ans
    }
}
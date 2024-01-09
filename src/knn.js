import KNN from 'ml-knn';

import { KNeighborsClassifier } from 'scikitjs'
export default class KNNModel {
    constructor(options) {
        this.options = options
        this.model = null

    }
    async train(x_train, y_train, k = 3) {
        console.log(this.options.metric);
        this.model = new KNeighborsClassifier({ nNeighbors: k, metric: this.options.metric })
        await this.model.fit(x_train, y_train);
    }
    predict(x_test) {
        if (this.model === null || this.model === undefined) {
            throw "model not found."
        }
        var ans = window.tf.tidy(() => {
            let results = this.model.predict(x_test);
            return Array.from(results.dataSync())
        })
        return ans
    }
}
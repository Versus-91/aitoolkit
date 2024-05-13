
import { KNeighborsRegressor } from 'scikitjs'
export default class KNNRegressor {
    constructor(options) {
        this.options = options
        this.model = null

    }
    async train(x_train, y_train, k = 3) {
        return new Promise((resolve, reject) => {
            try {
                setTimeout(async () => {
                    this.model = new KNeighborsRegressor({ nNeighbors: k, metric: this.options.metric })
                    await this.model.fit(x_train, y_train);
                    resolve()
                }, 1000)
            } catch (error) {
                reject(error)
            }
        })
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
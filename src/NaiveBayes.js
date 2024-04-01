import { GaussianNB } from 'scikitjs'
export default class NaiveBayes {
    constructor(options) {
        this.options = options
        this.model = null
    }
    async train(x_train, y_train) {
        const priors = this.options.priors?.split(',').map((m) => parseFloat(m))
        this.model = new GaussianNB({ priors: priors })
        await this.model.fit(x_train, y_train)

    }
    predict(x_test) {
        const result = this.model.predict(x_test);
        return result.dataSync()
    }
}
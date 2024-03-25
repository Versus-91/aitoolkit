import { GaussianNB } from 'scikitjs'
export default class NaiveBayes {
    constructor(options) {
        this.model = null
    }
    async train(x_train, y_train) {

        this.model = new GaussianNB()
        await this.model.fit(x_train, y_train)

    }
    predict(x_test) {
        const result = this.model.predict(x_test);
        return result.dataSync()
    }
}
import xgb from '../xgboost/index';
export default class BoostingRegression {
    constructor(options) {
        this.options = options
    }
    async init(options) {
        const XGBoost = await xgb;
        this.model = new XGBoost(options);
    }
    async train(x_train, y_train) {
        if (!this.model) {
            await this.init(this.options);
        }
        return new Promise((resolve, reject) => {
            try {
                setTimeout(async () => {
                    this.model.train(x_train, y_train);
                    resolve()
                }, 1000)
            } catch (error) {
                reject(error)
            }
        })
    }
    async predict(x_test) {
        if (!this.model) {
            await this.init(this.options);
        }
        const result = this.model.predict(x_test);
        return result
    }
}

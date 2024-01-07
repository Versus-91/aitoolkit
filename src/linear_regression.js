import MLR from "ml-regression-multivariate-linear"

export default class LinearRegression {

    train(x, y) {
        this.model = new MLR(x, y);
    }
    predict(x) {
        return this.model.predict(x)
    }
    stats() {
        return this.model.toJSON()
    }
}
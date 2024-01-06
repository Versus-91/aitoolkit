import MLR from "ml-regression-multivariate-linear"

export default class LinearRegression {

    fit(x, y) {
        this.model = new MLR(x, y);
    }
    predict(x) {
        return this.model.predict(x)
    }
    stats() {
        return model.toJSON()
    }
}
import { Settings } from "../feature_types";
import LogisticRegression from "./logistic_regression";
import LinearRegression from "./linear_regression";
import KNNModel from './knn';

export var ModelFactory = function () {
    this.createModel = function (modelName, ChartController = null, ...params) {
        var model;
        if (modelName === Settings.classification.logistic_regression) {
            model = new LogisticRegression(ChartController)
        } else if (modelName === Settings.classification.k_nearest_neighbour) {
            model = new KNNModel()
        } else if (modelName === Settings.classification.random_forest) {

        } else if (modelName === Settings.classification.support_vectore_machine) {

        } else if (modelName === Settings.classification.boosting) {

        } else if (modelName === Settings.classification.discriminant_analysis) {

        } else if (modelName === Settings.regression.linear_regression) {
            model = new LinearRegression()
        } else {
            throw "model not supported."
        }
        model.name = modelName;
        model.train = model.train
        model.evaluate = model.evaluate
        return model;
    }
}
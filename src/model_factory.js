import { Settings } from "../feature_types";
import { LogisticRegression } from "./LogisticRegression";
import LinearRegression from "./linear_regression";
import SupportVectorMachine from './svm';
import Boosting from './boosting';
import KNNModel from './knn';
import RandomForest from "./random_forest";
import NaiveBayes from "./NaiveBayes";


export var ModelFactory = function () {
    this.createModel = function (modelName, ChartController = null, options) {
        var model;
        if (modelName === Settings.classification.logistic_regression) {
            model = new LogisticRegression(options)
        } else if (modelName === Settings.classification.k_nearest_neighbour) {
            model = new KNNModel()
        } else if (modelName === Settings.classification.random_forest) {
            model = new RandomForest(options)

        } else if (modelName === Settings.classification.support_vectore_machine) {
            model = new SupportVectorMachine(options)
        } else if (modelName === Settings.classification.boosting) {
            model = new Boosting(options)
        } else if (modelName === Settings.classification.discriminant_analysis) {

        } else if (modelName === Settings.regression.linear_regression) {
            model = new LinearRegression()
        } else if (modelName === Settings.classification.naive_bayes) {
            model = new NaiveBayes()
        } else {
            throw "model not supported."
        }
        model.name = modelName;
        model.train = model.train
        model.evaluate = model.evaluate
        return model;
    }
}
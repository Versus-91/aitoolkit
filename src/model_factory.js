import { Settings } from "../feature_types";
import LogisticRegression from "./logistic_regression";
import LinearRegression from "./regression/linear_regression";
import SupportVectorMachine from './svm';
import Boosting from './boosting';
import BoostingRegression from './regression/boosting';

import KNNModel from './knn';
import KNNRegressor from './regression/knn_regressor';
import RandomForest from "./random_forest";
import RandomForestRegressor from "./regression/random_forest";
import NaiveBayes from "./NaiveBayes";
import DiscriminantAnalysis from "./lda";
import PolynomialRegression from "./regression/polynomial_regression";



export var ModelFactory = function () {
    this.createModel = (modelName, options) => {
        var model;
        if (modelName.value === Settings.classification.logistic_regression.value) {
            model = new LogisticRegression(options);
        } else if (modelName.value === Settings.classification.k_nearest_neighbour.value) {
            model = new KNNModel(options);
        } else if (modelName.value === Settings.classification.random_forest.value) {
            model = new RandomForest(options);
        } else if (modelName.value === Settings.classification.support_vector_machine.value) {
            model = new SupportVectorMachine(options);
        } else if (modelName.value === Settings.classification.boosting.value) {
            model = new Boosting(options);
        } else if (modelName.value === Settings.classification.discriminant_analysis.value) {
            model = new DiscriminantAnalysis(options);
        } else if (modelName.value === Settings.regression.linear_regression.value) {
            model = new LinearRegression(options);
        } else if (modelName.value === Settings.classification.naive_bayes.value) {
            model = new NaiveBayes(options);
        }
        else if (modelName.value === Settings.regression.k_nearest_neighbour.value) {
            model = new KNNRegressor(options);
        } else if (modelName.value === Settings.regression.support_vector_machine.value) {
            model = new SupportVectorMachine(options);
        } else if (modelName.value === Settings.regression.boosting.value) {
            model = new BoostingRegression(options);
        } else if (modelName.value === Settings.regression.random_forest.value) {
            model = new RandomForestRegressor(options);
        }
        else if (modelName.value === Settings.regression.polynomial_regression.value) {
            model = new PolynomialRegression(options);
        } else {
            throw "model not supported.";
        }
        model.name = modelName.value;
        model.train = model.train;
        model.evaluate = model.evaluate;
        return model;
    }
}
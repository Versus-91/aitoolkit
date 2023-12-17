export const FeatureCategories = {
    Numerical: "Numerical",
    Nominal: "Nominal",
    Ordinal: "Ordinal",
};
export const Settings = {
    "classification": {
        "logistic_regression": {
            "lable": "logistic regression",
            "value": 1,
            "options":{
                "regularization":{
                    "label":"regulrization",
                    "type" :"select",
                    "values":["no","lasso","ridge"]
                }
            }
        },
        "discriminant_analysis": {
            "lable": "discriminant analysis",
            "value": 2
        },
        "k_nearest_neighbour": {
            "lable": "k nearest neighbour",
            "value": 3,
            "options": {
                "min": {
                    type: "number",
                    default: 3
                },
                "max": {
                    type: "number",
                    default: 9
                },
                "metric": {
                    type: "select",
                    default: "euclidean",
                    values: ["euclidean", "minkowski", "manhattan"]
                },
            },
        },
        "support_vectore_machine": {
            "lable": "support vector machine",
            "value": 4,
            "options": {
                "kernel": {
                    type: "select",
                    values: []
                },
                "metric": {
                    type: "select",
                    default: "euclidean",
                    values: ["euclidean", "minkowski", "manhattan"]
                },
            },
        },
        "random_forest": {
            "lable": "Random forest",
            "value": 5,
            "options": {
                "estimators": {
                    type: "number",
                    default: 10
                },
                "features": {
                    type: "number",
                    default: "square-p"
                },
                "depth": {
                    type: "number",
                    default: 5
                }
            },
            "entropy": ["class_error", "gini", "Bernoulli"]
        },
        "boosting": {
            "lable": "boosting",
            "value": 6
        },
        "naive_bayes": {
            "lable": "Naive Bayes",
            "value": 7,
            "options": {
                "laplace": {
                    type: "number",
                    default: 0.05
                },
                "priors": {
                    type: "number",
                    default: 0.05
                },
                "laplace": {
                    type: "number",
                    default: 0.05
                },
                "type": {
                    type: "select",
                    values: ["Gaussian", "Multinomial", "Bernoulli"]
                }
            }

        },
    },
    "regression": {
        "linear_regression": {
            "lable": "Linear Regression",
            "value": 8,
            "feature_selection": ["no", "Lasso", "ridge"],
            "criteria": ["AIC", "BIC", "AR2",],
        },
    },
};
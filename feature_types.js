export const FeatureCategories = {
    Numerical: "Numerical",
    Nominal: "Nominal",
    Ordinal: "Ordinal",
};
export const Settings = {
    "classification": {
        "logistic_regression": {
            "lable": "logistic regression",
            "value": 1
        },
        "discriminant_analysis": {
            "lable": "discriminant analysis",
            "value": 2
        },
        "k_nearest_neighbour": {
            "lable": "k nearest neighbor",
            "value": 3,
            "options": {
                "K range": {
                    type: "number",
                    default: [3, 9]
                },
                "distance metric": {
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
                    type: "number",
                    default: [3, 9]
                },
                "distance metric": {
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
                "Number of Decission trees": {
                    type: "number",
                    default: 10
                },
                "features selcetion length": {
                    type: "number",
                    default: "square-p"
                },
                "decission_tree_depth": {
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
            "laplace": "no",
            "class_priors": "no",
            "feature_selection": ["no", "Lasso", "ridge"],
            "type": ["Gaussian", "Multinomial", "Bernoulli"]
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
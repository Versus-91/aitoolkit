export const FeatureCategories = {
    Numerical: "Numerical",
    Nominal: "Nominal",
    Ordinal: "Ordinal",
};
export const Settings = {
    "classification": {
        "logistic_regression": {
            "label": "Logistic Regression",
            "value": 1,
            "options": {
                "regularization": {
                    "label": "regulrization",
                    "type": "select",
                    default: "no",
                    "values": [{ label: "No", value: "No" }, { label: "lasso", value: "lasso" }, { label: "ridge", value: "ridge" }]
                }
            }
        },
        "discriminant_analysis": {
            "label": "Discriminant Analysis",
            "value": 2,
            "options": {
                "type": {
                    "label": "type",
                    "type": "select",
                    default: "linear",
                    "values": [{ label: "linear", value: "linear" }, { label: "quadratic", value: "quadratic" }]
                },
                "priors": {
                    type: "text",
                    placeholder: "comma separated priors"
                },
            }
        },
        "k_nearest_neighbour": {
            "label": "k nearest neighbour",
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
                    default: "manhattan",
                    values: [{ label: "euclidean", value: "euclidean" }, { label: "manhattan", value: "manhattan" }]
                },
            },
        },
        "support_vector_machine": {
            "label": "Support vector machine",
            "value": 4,
            "options": {
                "kernel": {
                    type: "select",
                    default: "RBF",
                    values: [{ label: "RBF", value: "RBF" }, { label: "Linear", value: "Linear" }, { label: "Polynomial", value: "Polynomial" }
                        , { label: "Sigmoid", value: "Sigmoid" }]
                },
                "gamma": {
                    type: "number",
                    for: ["RBF", "Sigmoid", "Polynomial"],
                    default: 1
                },
                "bias": {
                    type: "number",
                    for: ["Sigmoid", "Sigmoid"],
                    default: 0
                },
                "degree": {
                    type: "number",
                    for: ["Polynomial"],
                    default: 3
                },
            },
        },
        "random_forest": {
            "label": "Random forest",
            "value": 5,
            "options": {
                "estimators": {
                    type: "number",
                    default: 100
                },
                "features": {
                    type: "number",
                    default: "sqrt"
                },
                "depth": {
                    type: "number",
                    default: 5
                },
                "criteria": {
                    type: "select",
                    default: "gini",
                    "values": [{ label: "gini", value: "gini" }, { label: "log loss", value: "log_loss" },
                    { label: "entropy", value: "entropy" }]
                }
            },
        },
        "boosting": {
            "label": "Boosting",
            "value": 6,
            "options": {
                "booster": {
                    type: "select",
                    default: "gbtree",
                    values: [{ label: "gbtree", value: "gbtree" }, { label: "gblinear", value: "gblinear" }, { label: "dart", value: "dart" }]
                },
                "eta": {
                    type: "number",
                    default: 0.3
                },
                "iterations": {
                    type: "number",
                    default: 200
                },
                "depth": {
                    type: "number",
                    default: 5
                },

            },
        },
        "naive_bayes": {
            "label": "Naive Bayes",
            "value": 7,
            "options": {
                "laplace": {
                    type: "number",
                    default: 0.05
                },
                "priors": {
                    type: "text",
                    placeholder: "comma separated priors"
                },
                "laplace": {
                    type: "number",
                    default: 0.05
                },
                "type": {
                    type: "select",
                    default: "Gaussian",
                    values: [{ label: "Gaussian", value: "Gaussian" }, { label: "Multinomial", value: "Multinomial" }, { label: "Bernoulli", value: "Bernoulli" }]
                }
            }

        },
    },
    "regression": {
        "linear_regression": {
            "label": "Linear Regression",
            "value": 8,
            "feature_selection": ["no", "Lasso", "ridge"],
            "criteria": ["AIC", "BIC", "AR2",],
        },
        "k_nearest_neighbour": {
            "label": "k nearest neighbour Regression",
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
                    values: ["euclidean", "manhattan"]
                },
            },
        },
        "boosting": {
            "label": "Boosting Regression",
            "value": 6,
            "options": {
                "booster": {
                    type: "select",
                    default: "gbtree",
                    values: [{ label: "gbtree", value: "gbtree" }, { label: "gblinear", value: "gblinear" }, { label: "dart", value: "dart" }]
                },
                "eta": {
                    type: "number",
                    default: 0.3
                },
                "iterations": {
                    type: "number",
                    default: 200
                },
                "depth": {
                    type: "number",
                    default: 5
                },

            },
        },
        "support_vector_machine": {
            "label": "Support vector machine Regression",
            "value": 4,
            "options": {
                "kernel": {
                    type: "select",
                    default: "RBF",
                    values: [{ label: "RBF", value: "RBF" }, { label: "Linear", value: "Linear" }, { label: "Polynomial", value: "Polynomial" }
                        , { label: "Sigmoid", value: "Sigmoid" }]
                },
                "gamma": {
                    type: "number",
                    for: ["RBF", "Sigmoid", "Polynomial"],
                    default: 1
                },
                "bias": {
                    type: "number",
                    for: ["Sigmoid", "Sigmoid"],
                    default: 0
                },
                "degree": {
                    type: "number",
                    for: ["Polynomial"],
                    default: 3
                },
            },
        },
        "random_forest": {
            "label": "Random forest Regression",
            "value": 5,
            "options": {
                "estimators": {
                    type: "number",
                    default: 100
                },
                "features": {
                    type: "number",
                    default: "sqrt"
                },
                "depth": {
                    type: "number",
                    default: 5
                },
                "criteria": {
                    type: "select",
                    default: "gini",
                    "values": [{ label: "gini", value: "gini" }, { label: "log loss", value: "log_loss" },
                    { label: "entropy", value: "entropy" }]
                }
            },
        },
    },
};
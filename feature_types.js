export const FeatureCategories = {
    Numerical: "Numerical",
    Nominal: "Nominal",
    Ordinal: "Ordinal",
};
export const Settings = {
    "classification": {
        "logistic_regression": {
            "label": "logistic regression",
            "value": 1,
            "options": {
                "regularization": {
                    "label": "regulrization",
                    "type": "select",
                    default: "no",
                    "values": ["no", "lasso", "ridge"]
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
                    "values": ["linear", "quadratic"]
                },
                "priors": {
                    type: "text",
                    default: 3
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
                    default: "euclidean",
                    values: ["euclidean", "manhattan"]
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
                    values: ["RBF", "Linear", "Polynomial", "Sigmoid"]
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
                    default: "square-p"
                },
                "depth": {
                    type: "number",
                    default: 5
                },
                "criteria": {
                    type: "select",
                    default: "gini",
                    "values": ["gini", "class_error", "entropy"]
                }
            },
        },
        "boosting": {
            "label": "Boosting",
            "value": 6,
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
                    type: "number",
                    default: 0.05
                },
                "laplace": {
                    type: "number",
                    default: 0.05
                },
                "type": {
                    type: "select",
                    default: "Gaussian",
                    values: ["Gaussian", "Multinomial", "Bernoulli"]
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
                    default: "euclidean",
                    values: ["euclidean", "manhattan"]
                },
            },
        },
    },
};
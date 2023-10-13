import { FeatureCategories } from "./feature_types.js";

export default class DataLoader {
    constructor() {

    }
    createDataSets(data, features, categoricalFeatures, testSize) {
        const X = data.map(r =>
            features.flatMap(f => {
                if (categoricalFeatures.has(f)) {
                    return oneHot(!r[f] ? 0 : r[f], VARIABLE_CATEGORY_COUNT[f]);
                }
                return !r[f] ? 0 : r[f];
            })
        );

        const X_t = normalize(tf.tensor2d(X));

        const y = tf.tensor(data.map(r => (!r.SalePrice ? 0 : r.SalePrice)));

        const splitIdx = parseInt((1 - testSize) * data.length, 10);

        const [xTrain, xTest] = tf.split(X_t, [splitIdx, data.length - splitIdx]);
        const [yTrain, yTest] = tf.split(y, [splitIdx, data.length - splitIdx]);

        return [xTrain, xTest, yTrain, yTest];
    };
    findDataTypes(items) {
        if (Array.isArray(items) && items.length == 0) {
            throw "input is not an array"
        }
        let result = {};
        for (const key in items[0]) {
            result[key] = "na"
            for (let index = 0; index < items.length; index++) {
                const element = items[index];
                if (Object.hasOwnProperty.call(element, key)) {
                    if (typeof element[key] == "number") {
                        result[key] = FeatureCategories.Numerical;
                        break;
                    } else if (typeof element[key] == "string") {
                        result[key] = FeatureCategories.Categorical;
                        break;
                    }
                }
            }
        }
        return result;
    }
    countCategoricalFeaturesOccurance(items, outputLable) {
        let result = {};
        for (let index = 0; index < items.length; index++) {
            
            const element = items[index];
        }
    }
}
import { FeatureCategories } from "./feature_types.js";

export default class DataLoader {
    constructor() {

    }
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

    createDataSets(data, features, testSize, batchSize) {
        const oneHot = outcome => Array.from(tf.oneHot(outcome, 2).dataSync());
        const X = data.map(r =>
            features.map(f => {
                const val = r[f];
                return val === undefined ? 0 : val;
            })
        );
        const y = data.map(r => {
            const outcome = r.Outcome === undefined ? 0 : r.Outcome;
            return oneHot(outcome);
        });

        const splitIdx = parseInt((1 - testSize) * data.length, 10);

        const ds = tf.data
            .zip({ xs: tf.data.array(X), ys: tf.data.array(y) })
            .shuffle(data.length, 42);
        console.log(ds);
        return [
            ds.take(splitIdx).batch(batchSize),
            ds.skip(splitIdx + 1).batch(batchSize),
            tf.tensor(X.slice(splitIdx)),
            tf.tensor(y.slice(splitIdx))
        ];
    };
}
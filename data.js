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
    findMissinValues(data) {
        if (!Array.isArray(data) || data.length == 0) {
            throw "input must be an array."
        }
        const result = {}
        for (const key in data[0]) {
            if (Object.hasOwnProperty.call(data[0], key)) {
                const element = data[0][key];
                result[key] = 0
            }
        }
        data.forEach(element => {
            for (const key in element) {
                if (Object.hasOwnProperty.call(element, key)) {
                    const item = element[key];
                    if (item === null || item === undefined) {
                        result[key]++
                    }
                }
            }
        });
        let itemsCount = data.length
        for (const key in result) {
            if (Object.hasOwnProperty.call(result, key)) {
                result[key] = (result[key] / itemsCount) * 100;
            }
        }
    }
    findTargetPercents(items, target) {
        if (!Array.isArray(items) || items.length == 0) {
            throw "input must be an array."
        }
        const result = { "count": 0 }
        items.forEach(element => {
            for (const key in element) {
                if (Object.hasOwnProperty.call(element, key)) {
                    if (key === target) {
                        result.count++
                        if (Object.hasOwnProperty.call(result, element[key])) {
                            result[element[key]]++
                        } else {
                            result[element[key]] = 1
                        }
                    }
                }
            }
        });
        for (const key in result) {
            if (Object.hasOwnProperty.call(result, key)) {
                if (key !== "count") {
                    result[key] = (result[key] / result.count) * 100;
                }
            }
        }
        return result
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
        return [
            ds.take(splitIdx).batch(batchSize),
            ds.skip(splitIdx + 1).batch(batchSize),
            tf.tensor(X.slice(splitIdx)),
            tf.tensor(y.slice(splitIdx))
        ];
    };
}
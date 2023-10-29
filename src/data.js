import { FeatureCategories } from "../feature_types.js";
import { DataFrame, LabelEncoder, Series, tensorflow, concat, OneHotEncoder, getDummies } from 'danfojs/dist/danfojs-base';
export default class DataLoader {
    kernelDensityEstimation(data, kernel, bandwidth) {
        return function (x) {
            return (1 / (data.length * bandwidth)) * data.reduce(function (sum, dataPoint) {
                return sum + kernel((x - dataPoint) / bandwidth);
            }, 0);
        };
    }


    // Gaussian kernel function
    gaussian(x) {
        return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
    }




    // Function to find the mode
    findMode(data) {
        const frequencyMap = {};
        let maxFrequency = 0;
        let mode = null;

        // Count the frequency of each value in the array
        data.forEach((value) => {
            frequencyMap[value] = (frequencyMap[value] || 0) + 1;
            if (frequencyMap[value] > maxFrequency) {
                maxFrequency = frequencyMap[value];
                mode = value;
            }
        });

        return mode;
    }

    findDataTypes(items) {
        if (Array.isArray(items) && items.length == 0) {
            throw "input is not an array"
        }
        let result = {};
        for (let key in items[0]) {
            key = key.replace(/\s/g, '').replace(/[^\w-]/g, '_');
            result[key] = "na"
            for (let index = 0; index < items.length; index++) {
                const element = items[index];
                if (Object.hasOwnProperty.call(element, key)) {
                    if (typeof element[key] === "number") {
                        result[key] = FeatureCategories.Numerical;
                        break;
                    } else if (typeof element[key] === "string") {
                        result[key] = FeatureCategories.Categorical;
                        break;
                    }
                    else {
                        console.log(element[key]);
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

    splitData(data, testRatio = 0.2) {
        // Shuffle the data randomly
        // this.shuffleArray(data);

        // Calculate the split point
        const splitIndex = Math.floor(data.length * (1 - testRatio));

        // Split the data into training and testing sets
        const trainingData = data.slice(0, splitIndex);
        const testingData = data.slice(splitIndex);
        return {
            "test_data": testingData,
            "training_data": trainingData
        }
        // Now you have trainingData and testingData
        // You can use these for machine learning tasks
    }

    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }
    perprocess_data(data_frame) {
        // to do normalization
        let cols = []
        data_frame.columns.forEach((item) => {
            if (data_frame.column(item).dtype === 'string') {
                cols.push(item)
            }
        })
        let encoder = new LabelEncoder()
        cols.forEach((column) => {
            encoder.fit(data_frame[column])
            let encoded_column = encoder.transform(data_frame[column])
            data_frame.addColumn(column, encoded_column, { inplace: true })
        })
        return data_frame
    }
    set_model() {

    }
}
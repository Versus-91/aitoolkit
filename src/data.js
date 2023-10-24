import { FeatureCategories } from "../feature_types.js";
export default class DataLoader {

    gaussianKernel = (u) => {
        return (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * u * u);
    }
    GaussKDE(xi, x) {
        return (1 / Math.sqrt(2 * Math.PI)) * Math.exp(Math.pow(xi - x, 2) / -2);
    }
    kernelDensityEstimation(data, bandwidth) {
        return (x) => {
            return (1 / (data.length * bandwidth)) * data.reduce((sum, dataPoint) => {
                return sum + this.gaussianKernel((x - dataPoint) / bandwidth);
            }, 0);
        };
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

    splitData(data, testRatio = 0.2) {
        // Shuffle the data randomly
        this.shuffleArray(data);

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
}
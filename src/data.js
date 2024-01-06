import { FeatureCategories, Settings } from "../feature_types.js";

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
    handle_missing_values(data_frame, impute = true) {
        // to do normalization
        if (impute) {
            let string_columns = []
            let numeric_columns = []
            let string_column_modes = []
            let numeric_column_means = []
            data_frame.columns.forEach((item) => {
                if (data_frame.column(item).dtype === 'string') {
                    string_columns.push(item)
                } else {
                    numeric_columns.push(item)
                }
            })
            string_columns.forEach(element => {
                let mode = this.getCategoricalMode(element).mode
                string_column_modes.push(mode)
            });
            numeric_columns.forEach(element => {
                let mean = data_frame.column(element).mean()
                numeric_column_means.push(mean)
            });
            data_frame = data_frame.fillNa(string_column_modes, { columns: string_columns })
            data_frame = data_frame.fillNa(numeric_column_means, { columns: numeric_columns })

        } else {
            data_frame.dropNa({ axis: 1, inplace: true })
        }
        return data_frame
    }
    encode_dataset(data_frame, columns_types, model) {
        let df = data_frame.copy()

        columns_types = columns_types.filter(column => column.type === FeatureCategories.Nominal || column.type === FeatureCategories.Ordinal)
        columns_types.forEach((column) => {
            if (column.type === FeatureCategories.Ordinal) {
                let encoder = new LabelEncoder()
                encoder.fit(df[column.name])
                let encoded_column = encoder.transform(df[column.name])
                df.addColumn(column.name, encoded_column.values, { inplace: true })
            } else {
                df = getDummies(df, { columns: [column.name] })
                if (model === Settings.classification.logistic_regression.label || model === Settings.regression.linear_regression.label) {
                    df.drop({ columns: [df.columns.find(m => m.includes(column.name + "_"))], inplace: true })
                }
            }
        })

        return df
    }
    getCategoricalMode(arr) {
        if (arr.length === 0) {
            return null;
        }

        const categoryCount = {};
        categoryCount['total'] = 0
        categoryCount['mode'] = ''
        for (let i = 0; i < arr.length; i++) {
            const category = arr[i];
            if (category === null || category === undefined) {
                continue
            }
            categoryCount['total']++
            if (category in categoryCount) {
                categoryCount[category]++;
            } else {
                categoryCount[category] = 1;
            }
        }

        let modeCategory = null;
        let modeCount = 0;
        for (const category in categoryCount) {
            if (category === 'total') {
                continue
            }
            if (categoryCount[category] > modeCount) {
                modeCategory = category;
                modeCount = categoryCount[category];
            }
        }
        categoryCount['mode'] = modeCategory;
        return categoryCount;
    }
}
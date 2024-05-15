import { asyncRun } from "./py-worker";
import * as Papa from 'papaparse';
async function parseCsv(data) {
    return new Promise(resolve => {
        data = data.map((row) => {
            return Object.keys(row).sort().map(key => parseFloat(row[key]));
        });
        resolve(data);
    });
};

/**
 * Downloads and returns the csv.
 *
 * @param {string} filename Name of file to be loaded.
 *
 * @returns {Promise.Array<number[]>} Resolves to parsed csv data.
 */
export async function loadCsv(filename) {
    return new Promise(resolve => {
        const url = `${BASE_URL}${filename}.csv`;

        console.log(`  * Downloading data from: ${url}`);
        Papa.parse(url, {
            download: true,
            header: true,
            complete: (results) => {
                resolve(parseCsv(results['data']));
            }
        })
    });
};

/**
 * Shuffles data and label using Fisher-Yates algorithm.
 */
export async function shuffle(data, label) {
    let counter = data.length;
    let temp = 0;
    let index = 0;
    while (counter > 0) {
        index = (Math.random() * counter) | 0;
        counter--;
        // data:
        temp = data[counter];
        data[counter] = data[index];
        data[index] = temp;
        // label:
        temp = label[counter];
        label[counter] = label[index];
        label[index] = temp;
    }
};

/**
 * Calculate the arithmetic mean of a vector.
 *
 * @param {Array} vector The vector represented as an Array of Numbers.
 *
 * @returns {number} The arithmetic mean.
 */
function mean(vector) {
    let sum = 0;
    for (const x of vector) {
        sum += x;
    }
    return sum / vector.length;
};

/**
 * Calculate the standard deviation of a vector.
 *
 * @param {Array} vector The vector represented as an Array of Numbers.
 *
 * @returns {number} The standard deviation.
 */
function stddev(vector) {
    let squareSum = 0;
    const vectorMean = mean(vector);
    for (const x of vector) {
        squareSum += (x - vectorMean) * (x - vectorMean);
    }
    return Math.sqrt(squareSum / (vector.length - 1));
};

/**
 * Normalize a vector by its mean and standard deviation.
 *
 * @param {Array} vector Vector to be normalized.
 * @param {number} vectorMean Mean to be used.
 * @param {number} vectorStddev Standard Deviation to be used.
 *
 * @returns {Array} Normalized vector.
 */
const normalizeVector = (vector, vectorMean, vectorStddev) => {
    return vector.map(x => (x - vectorMean) / vectorStddev);
};

/**
 * Normalizes the dataset
 *
 * @param {Array} dataset Dataset to be normalized.
 * @param {boolean} isTrainData Whether it is training data or not.
 * @param {Array} vectorMeans Mean of each column of dataset.
 * @param {Array} vectorStddevs Standard deviation of each column of dataset.
 *
 * @returns {Object} Contains normalized dataset, mean of each vector column,
 *                   standard deviation of each vector column.
 */
export function normalizeDataset(
    dataset, isTrainData = true, vectorMeans = [], vectorStddevs = []) {
    const numFeatures = dataset[0].length;
    let vectorMean;
    let vectorStddev;

    for (let i = 0; i < numFeatures; i++) {
        const vector = dataset.map(row => row[i]);

        if (isTrainData) {
            vectorMean = mean(vector);
            vectorStddev = stddev(vector);

            vectorMeans.push(vectorMean);
            vectorStddevs.push(vectorStddev);
        } else {
            vectorMean = vectorMeans[i];
            vectorStddev = vectorStddevs[i];
        }

        const vectorNormalized =
            normalizeVector(vector, vectorMean, vectorStddev);

        vectorNormalized.forEach((value, index) => {
            dataset[index][i] = value;
        });
    }

    return { dataset, vectorMeans, vectorStddevs };
};

/**
 * Binarizes a tensor based on threshold of 0.5.
 *
 * @param {tf.Tensor} y Tensor to be binarized.
 * @param {number} threshold (default: 0.5).
 * @returns {tf.Tensor} Binarized tensor.
 */
export function binarize(y, threshold) {
    if (threshold == null) {
        threshold = 0.5;
    }
    tf.util.assert(
        threshold >= 0 && threshold <= 1,
        `Expected threshold to be >=0 and <=1, but got ${threshold}`);

    return tf.tidy(() => {
        const condition = y.greater(tf.scalar(threshold));
        return tf.where(condition, tf.onesLike(y), tf.zerosLike(y));
    });
}
export function encode_name(str) {
    let str_encoded = key.replace(/\s/g, '').replace(/[^\w-]/g, '_');
    return str_encoded
}
export function calculatePrecision(classIndex, confusionMatrix) {
    let truePositive = confusionMatrix[classIndex][classIndex];
    let falsePositive = 0;

    for (let i = 0; i < confusionMatrix.length; i++) {
        if (i !== classIndex) {
            falsePositive += confusionMatrix[i][classIndex];
        }
    }

    if (truePositive === 0 && falsePositive === 0) {
        return 1;
    }

    return truePositive / (truePositive + falsePositive);
}

export function calculateRecall(classIndex, confusionMatrix) {
    let truePositive = confusionMatrix[classIndex][classIndex];
    let falseNegative = 0;
    for (let i = 0; i < confusionMatrix.length; i++) {
        falseNegative += confusionMatrix[classIndex][i];
    }
    falseNegative -= truePositive;
    if (truePositive === 0 && falseNegative === 0) {
        return 1;
    }
    return truePositive / (truePositive + falseNegative);
}


export function calculateF1Score(classIndex, confusionMatrix) {
    const precision = calculatePrecision(classIndex, confusionMatrix);
    const recall = calculateRecall(classIndex, confusionMatrix);
    return (2 * precision * recall) / (precision + recall);
}

export async function metrics(y, y_pred, labels) {
    const context = {
        y: y,
        y_pred: y_pred,
        labels: labels
    };
    const script = `
        from sklearn.metrics import precision_recall_fscore_support, classification_report, f1_score,accuracy_score
        from js import y_pred,y,labels       
        # Perform t-SNE dimensionality reduction
        precision_recall_support = precision_recall_fscore_support(y, y_pred,labels=labels)
        f1_micro = f1_score(y, y_pred, average='micro')
        f1_macro = f1_score(y, y_pred, average='macro')
        accuracy = accuracy_score(y, y_pred)
        (precision_recall_support,f1_micro,f1_macro,accuracy)
    `;
    try {
        const { results, error } = await asyncRun(script, context);
        if (results) {
            console.log(results);
            return results;
        } else if (error) {
            throw error
        }
    } catch (e) {
        throw ("Something went wrong", e)
    }
}




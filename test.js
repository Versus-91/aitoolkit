function calculatePrecision(classIndex, confusionMatrix) {
    let truePositive = confusionMatrix[classIndex][classIndex];
    let falsePositive = 0;
    let falseNegative = 0;

    for (let i = 0; i < confusionMatrix.length; i++) {
        falsePositive += confusionMatrix[i][classIndex];
        falseNegative += confusionMatrix[classIndex][i];
    }

    falsePositive -= truePositive;
    falseNegative -= truePositive;

    const precision = truePositive / (truePositive + falsePositive);
    return isNaN(precision) ? 0 : precision; // Check for NaN (division by zero)
}

function calculateRecall(classIndex, confusionMatrix) {
    let truePositive = confusionMatrix[classIndex][classIndex];
    let falseNegative = 0;

    for (let i = 0; i < confusionMatrix.length; i++) {
        falseNegative += confusionMatrix[classIndex][i];
    }

    falseNegative -= truePositive;

    const recall = truePositive / (truePositive + falseNegative);
    return isNaN(recall) ? 0 : recall; // Check for NaN (division by zero)
}

function calculateF1Score(classIndex, confusionMatrix) {
    const precision = calculatePrecision(classIndex, confusionMatrix);
    const recall = calculateRecall(classIndex, confusionMatrix);

    if (precision === 0 || recall === 0) {
        return 0; // Handle the case of precision or recall being zero
    }

    return 2 * ((precision * recall) / (precision + recall));
}

// Example confusion matrix for a 3-class scenario
const exampleMatrix = [
    [0, 0, 0], // Class 0
    [0, 0, 19], // Class 1
    [0, 0, 26]  // Class 2
];

// Calculate and display precision, recall, and F1-score for each class
for (let i = 0; i < exampleMatrix.length; i++) {
    console.log(`Class ${i} - Precision: ${calculatePrecision(i, exampleMatrix).toFixed(4)}, Recall: ${calculateRecall(i, exampleMatrix).toFixed(4)}, F1-score: ${calculateF1Score(i, exampleMatrix).toFixed(4)}`);
}

const val = "First sexual intercourse"; // Replace this with your input string
const processedString = val.replace(/[^a-zA-Z ]/g, "").trim();

const capitalizeAfterSpace = (str) => {
    const words = str.split(' ');
    for (let i = 1; i < words.length; i++) {
        words[i] = words[i].charAt(0).toUpperCase() + words[i].slice(1);
    }
    return words.join(' ');
};

const result = capitalizeAfterSpace(processedString);
console.log(result);
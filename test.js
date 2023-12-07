
function calculatePrecision(classIndex, confusionMatrix) {
    let truePositive = confusionMatrix[classIndex][classIndex];
    let falsePositive = 0;
    for (let i = 0; i < confusionMatrix.length; i++) {
        falsePositive += confusionMatrix[i][classIndex];
    }
    falsePositive -= truePositive;
    if (truePositive === 0 && falsePositive === 0) {
        return 1;
    }
    return truePositive / (truePositive + falsePositive);
}

function calculateRecall(classIndex, confusionMatrix) {
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


function calculateF1Score(classIndex, confusionMatrix) {
    const precision = calculatePrecision(classIndex, confusionMatrix);
    const recall = calculateRecall(classIndex, confusionMatrix);
    return (2 * precision * recall) / (precision + recall);
}

// Example confusion matrix for a 3-class scenario
const exampleMatrix = [
    [85, 10],  // true negatives, false positives
    [5, 100] // Class 2
];

// Calculate and display precision, recall, and F1-score for each class
for (let i = 0; i < exampleMatrix.length; i++) {
    console.log(`Class ${i} - Precision: ${calculatePrecision(i, exampleMatrix).toFixed(4)}, Recall: ${calculateRecall(i, exampleMatrix).toFixed(4)}, F1-score: ${calculateF1Score(i, exampleMatrix).toFixed(4)}`);
}

// Function to calculate precision, recall, and support for all classes
function calculateMetrics(trueLabels, predictedLabels) {
    const uniqueClasses = [...new Set([...trueLabels, ...predictedLabels])];
    const metrics = {};

    uniqueClasses.forEach((classLabel) => {
        let truePositives = 0;
        let falsePositives = 0;
        let falseNegatives = 0;

        for (let i = 0; i < trueLabels.length; i++) {
            if (predictedLabels[i] === classLabel) {
                if (trueLabels[i] === classLabel) {
                    truePositives++;
                } else {
                    falsePositives++;
                }
            } else {
                if (trueLabels[i] === classLabel) {
                    falseNegatives++;
                }
            }
        }

        const precision = truePositives / (truePositives + falsePositives);
        const recall = truePositives / (truePositives + falseNegatives);
        const support = truePositives + falseNegatives;

        metrics[classLabel] = {
            precision: isNaN(precision) ? 0 : precision,
            recall: isNaN(recall) ? 0 : recall,
            support: support,
        };
    });

    return metrics;
}

// Example usage:
const trueLabels = [0, 1, 2, 1, 2, 0]; // True labels (multi-class)
const predictedLabels = [0, 2, 1, 1, 2, 0]; // Predicted labels

// Calculate precision, recall, and support for all classes
const metrics = calculateMetrics(trueLabels, predictedLabels);

// Display metrics for each class
Object.keys(metrics).forEach((classLabel) => {
    console.log(`Class ${classLabel}:`);
    console.log(`Precision: ${metrics[classLabel].precision}`);
    console.log(`Recall: ${metrics[classLabel].recall}`);
    console.log(`Support: ${metrics[classLabel].support}`);
    console.log("--------------");
});
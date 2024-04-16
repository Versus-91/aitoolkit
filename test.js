function calculateMetrics(confusionMatrix) {
    // Input validation (optional):
    if (!Array.isArray(confusionMatrix) || !confusionMatrix.every(row => Array.isArray(row))) {
      throw new Error("Invalid confusion matrix format. It should be a 2D array.");
    }
  
    const numClasses = confusionMatrix.length;
    const metrics = {};
    let totalTruePositives = 0;
    let totalPredictedPositives = 0;
    let totalActualPositives = 0;
  
    // Calculate accuracy and metrics for each class
    for (let classIdx = 0; classIdx < numClasses; classIdx++) {
      const truePositives = confusionMatrix[classIdx][classIdx];
      const falsePositives = confusionMatrix[classIdx].reduce((sum, val, idx) => sum + (idx !== classIdx ? val : 0), 0);
      const falseNegatives = confusionMatrix.reduce((sum, row) => sum + (row[classIdx] !== classIdx ? row[classIdx] : 0), 0);
  
      // Handle division by zero (optional):
      const precision = truePositives / (truePositives + falsePositives) || 0;
      const recall = truePositives / (truePositives + falseNegatives) || 0;
  
      // F1-score (harmonic mean of precision and recall)
      const f1 = (2 * precision * recall) / (precision + recall) || 0;
  
      metrics[classIdx] = {
        precision,
        recall,
        f1
      };
  
      totalTruePositives += truePositives;
      totalPredictedPositives += confusionMatrix[classIdx].reduce((sum, val) => sum + val, 0);
      totalActualPositives += confusionMatrix.reduce((sum, row) => sum + row[classIdx], 0);
    }
  
    // Calculate overall accuracy
    const accuracy = (totalTruePositives / totalPredictedPositives) || 0;
  
    // Handle cases where there are no actual positives (optional):
    if (totalActualPositives === 0) {
      accuracy = 1; // Assuming all negatives are correctly classified
    }
  
    return {
      accuracy,
      ...metrics // Spread operator to include class-wise metrics
    };
  }
  
  // Example usage:
  const confusionMatrix = [
    [0, 0, 0],
    [0, 0, 8],
    [0, 0, 37]
  ];
  
  const results = calculateMetrics(confusionMatrix);
  
  console.log(results);
  
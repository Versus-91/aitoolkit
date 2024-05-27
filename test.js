function calculateMSE(actualValues, predictedValues) {
  if (actualValues.length !== predictedValues.length) {
    throw new Error("The lengths of actual values and predicted values must be the same.");
  }

  const n = actualValues.length;
  let sumSquaredError = 0;

  for (let i = 0; i < n; i++) {
    const squaredError = Math.pow(actualValues[i] - predictedValues[i], 2);
    sumSquaredError += squaredError;
  }

  const meanSquaredError = sumSquaredError / n;
  return meanSquaredError;
}
let y_true = [3, -0.5, 2, 7]
let y_pred = [2.5, 0.0, 2, 8]
console.log(calculateMSE(y_true, y_pred));
// Assuming you're using TensorFlow.js 0.14
const tf = require('@tensorflow/tfjs');

// Create a tensor with shape [2, 3]
const tensor = tf.tensor([[1, 3], [4, 6], [4, 6], [4, 6], [4, 6]]);

// Use dataSync() to get a flattened array
const data = tensor.dataSync(); // [1, 2, 3, 4, 5, 6]
// Get the shape of the original tensor
const shape = tensor.shape;

// A utility function to reshape the flattened array
function reshape(array, shape) {
    if (shape.length === 0) return array[0];

    const [size, ...restShape] = shape;
    const result = [];
    const restSize = restShape.reduce((a, b) => a * b, 1);
    console.log(restSize);

    for (let i = 0; i < size; i++) {
        result.push(reshape(array.slice(i * restSize, (i + 1) * restSize), restShape));
    }

    return result;
}

// Reshape the flattened array back to the original shape
const reshapedData = reshape(data, shape);

console.log(reshapedData); // Output: [[1, 2, 3], [4, 5, 6]]

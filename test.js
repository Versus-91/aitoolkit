// Import TensorFlow.js library
const tf = require('@tensorflow/tfjs');

async function pca(data, numComponents) {
    // Normalize the data
    const mean = tf.mean(data, 0);
    const centeredData = tf.sub(data, mean);
    const std = tf.sqrt(tf.mean(tf.square(centeredData), 0));
    const normalizedData = tf.div(centeredData, std);

    // Compute covariance matrix
    const covarianceMatrix = tf.matMul(normalizedData.transpose(), normalizedData).div(normalizedData.shape[0]);

    // Compute eigenvectors and eigenvalues
    const { values: eigenvalues, vectors: eigenvectors } = tf.linalg.eig(covarianceMatrix);
    
    // Sort eigenvalues and eigenvectors
    const sortedIndices = tf.argsort(eigenvalues, 'descend').squeeze();
    const sortedEigenvalues = tf.gather(eigenvalues, sortedIndices);
    const sortedEigenvectors = tf.gather(eigenvectors, sortedIndices, 1);

    // Select top k eigenvectors
    const principalComponents = sortedEigenvectors.slice([0, 0], [sortedEigenvectors.shape[0], numComponents]);

    // Project data onto new feature space
    const projectedData = tf.matMul(normalizedData, principalComponents);

    return { projectedData, principalComponents };
}

// Example usage
async function runPCA() {
    // Example data (replace this with your own data)
    const data = tf.tensor2d([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]);

    // Number of principal components to keep
    const numComponents = 2;

    // Perform PCA
    const { projectedData, principalComponents } = await pca(data, numComponents);
    
    console.log('Projected Data:');
    projectedData.print();
    console.log('Principal Components:');
    principalComponents.print();
}

// Run the PCA function
runPCA();

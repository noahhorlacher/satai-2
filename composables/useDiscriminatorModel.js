import * as tf from '@tensorflow/tfjs';

export default function useDiscriminatorModel() {
    // Input layer
    const inputs = tf.input({ shape: [64, 64, 1] });

    // First convolutional layer
    const conv1 = tf.layers.conv2d({
        filters: 8,
        kernelSize: 3,
        strides: 2,
        padding: 'same',
        activation: 'relu'
    }).apply(inputs);

    // Second convolutional layer (downsampling)
    const conv2 = tf.layers.conv2d({
        filters: 16,
        kernelSize: 3,
        strides: 2,
        padding: 'same',
        activation: 'relu'
    }).apply(conv1);

    // Third convolutional layer (further downsampling)
    const conv3 = tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        strides: 2,
        padding: 'same',
        activation: 'relu'
    }).apply(conv2);

    // Flatten the output before the dense layer
    const flatten = tf.layers.flatten().apply(conv3);

    // Output layer - one unit with sigmoid activation for binary classification
    const outputs = tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }).apply(flatten);

    // Create the model
    const model = tf.model({ inputs: inputs, outputs: outputs });

    return model;
}

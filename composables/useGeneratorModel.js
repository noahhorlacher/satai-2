import * as tf from '@tensorflow/tfjs';

export default function useGeneratorModel() {
    // Input layer
    const inputs = tf.input({ shape: [100] });

    // Dense layer
    const dense = tf.layers.dense({
        units: 8 * 8 * 256,
        activation: 'relu'
    }).apply(inputs);

    // Reshape layer to 3D tensor
    const reshape = tf.layers.reshape({
        targetShape: [8, 8, 256]
    }).apply(dense);

    // First transposed convolutional layer to upscale the image
    const conv2dTranspose1 = tf.layers.conv2dTranspose({
        filters: 128,
        kernelSize: 5,
        strides: 2,
        padding: 'same',
        activation: 'relu'
    }).apply(reshape);

    // Second transposed convolutional layer to upscale again
    const conv2dTranspose2 = tf.layers.conv2dTranspose({
        filters: 64,
        kernelSize: 5,
        strides: 2,
        padding: 'same',
        activation: 'relu'
    }).apply(conv2dTranspose1);

    // Output layer
    const outputs = tf.layers.conv2dTranspose({
        filters: 1,
        kernelSize: 5,
        strides: 2,
        padding: 'same',
        activation: 'sigmoid'
    }).apply(conv2dTranspose2);

    // Create the generator model
    const model = tf.model({ inputs: inputs, outputs: outputs });

    return model;
}

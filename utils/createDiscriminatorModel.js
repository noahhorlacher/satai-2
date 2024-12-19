import * as tf from '@tensorflow/tfjs'

export function createDiscriminatorModel(trainingDimensions, learningRate, clipValue) {
    const discriminator = tf.sequential()

    // First Convolutional layer - Assuming a larger kernel and stride to reduce dimension significantly
    discriminator.add(tf.layers.conv2d({
        inputShape: [trainingDimensions.y, trainingDimensions.x, 1],
        filters: 64,
        kernelSize: [3, 3],
        strides: [2, 2],
        padding: 'same',
        activation: 'relu'
    }));

    // Second Convolutional layer
    discriminator.add(tf.layers.conv2d({
        filters: 128,
        kernelSize: [3, 3],
        strides: [2, 2],  // Strides for downsampling
        padding: 'same',
        activation: 'relu',
        alpha: 0.2
    }));

    // Batch normalization to stabilize training
    discriminator.add(tf.layers.batchNormalization());

    // Third Convolutional layer
    discriminator.add(tf.layers.conv2d({
        filters: 256,
        kernelSize: [3, 3],
        strides: [2, 2],
        padding: 'same',
        activation: 'relu'
    }));

    // Fourth Convolutional layer
    // discriminator.add(tf.layers.conv2d({
    //     filters: 512,
    //     kernelSize: [3, 3],
    //     strides: [2, 2],
    //     padding: 'same',
    //     activation: 'relu'
    // }));

    // Flatten layer
    discriminator.add(tf.layers.flatten());

    // Dropout layer for regularization
    discriminator.add(tf.layers.dropout({
        rate: 0.4  // Dropout rate for regularization
    }));

    // Fully Connected Layer
    discriminator.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }))

    // Compile the discriminator
    discriminator.compile({
        loss: 'binaryCrossentropy',
        optimizer: tf.train.adam(learningRate, 0.5, 0.999),
        clipValue: clipValue
    })

    return discriminator
}
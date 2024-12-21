import * as tf from '@tensorflow/tfjs'

export function createDiscriminatorModel(trainingDimensions, learningRate) {
    const discriminator = tf.sequential()

    // First Convolutional Layer
    discriminator.add(tf.layers.conv2d({
        inputShape: [trainingDimensions.y, trainingDimensions.x, 1],
        filters: 64,
        kernelSize: [3, 3],
        strides: [2, 2],
        padding: 'same',
    }))
    discriminator.add(tf.layers.batchNormalization())
    discriminator.add(tf.layers.leakyReLU({ alpha: 0.2 }))

    // Second Convolutional Layer
    discriminator.add(tf.layers.conv2d({
        filters: 128,
        kernelSize: [3, 3],
        strides: [2, 2],  // Strides for downsampling
        padding: 'same'
    }))

    // Batch normalization to stabilize training
    discriminator.add(tf.layers.batchNormalization())
    discriminator.add(tf.layers.leakyReLU({ alpha: 0.2 }))

    // Third Convolutional Layer
    discriminator.add(tf.layers.conv2d({
        filters: 256,
        kernelSize: [3, 3],
        strides: [2, 2],
        padding: 'same'
    }))
    discriminator.add(tf.layers.batchNormalization())
    discriminator.add(tf.layers.leakyReLU({ alpha: 0.2 }))

    // Fourth Convolutional Layer
    discriminator.add(tf.layers.conv2d({
        filters: 512,
        kernelSize: [3, 3],
        strides: [2, 2],
        padding: 'same'
    }))
    discriminator.add(tf.layers.batchNormalization())
    discriminator.add(tf.layers.leakyReLU({ alpha: 0.2 }))

    // Flatten layer
    discriminator.add(tf.layers.flatten());

    // Dropout layer for regularization
    discriminator.add(tf.layers.dropout({
        rate: 0.3  // Dropout rate for regularization
    }))

    // Fully Connected Layer
    discriminator.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }))

    // Compile the discriminator
    discriminator.compile({
        loss: 'binaryCrossentropy',
        optimizer: tf.train.adam(learningRate, 0.5, 0.999),
    })

    return discriminator
}
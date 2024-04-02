import * as tf from '@tensorflow/tfjs'

export function createDiscriminatorModel(trainingDimensions, learningRate, clipValue) {
    const discriminator = tf.sequential()

    // First Convolutional layer - Assuming a larger kernel and stride to reduce dimension significantly
    discriminator.add(tf.layers.conv2d({
        inputShape: [trainingDimensions.y, trainingDimensions.x, 1],
        filters: trainingDimensions.y * 2,
        kernelSize: [5, 5],
        strides: [2, 2],
        padding: 'same',
        activation: 'relu'
    }));

    // Pooling Layer
    discriminator.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));

    // Second Convolutional layer
    discriminator.add(tf.layers.conv2d({
        filters: trainingDimensions.y,
        kernelSize: [3, 3],
        strides: [1, 1],
        padding: 'same',
        activation: 'relu'
    }));

    // Pooling Layer
    discriminator.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));

    // Third Convolutional layer
    discriminator.add(tf.layers.conv2d({
        filters: trainingDimensions.y / 2,
        kernelSize: [3, 3],
        strides: [1, 1],
        padding: 'same',
        activation: 'relu'
    }));

    // Pooling Layer
    discriminator.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));

    // Flatten layer
    discriminator.add(tf.layers.flatten());

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
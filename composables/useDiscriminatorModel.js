import * as tf from '@tensorflow/tfjs'

export default function useDiscriminatorModel() {
    // define the Adversarial model, binary classifier (is it real or fake)
    // takes an image and outputs a probability that the image is human made music
    const discriminatorModel = tf.sequential()

    // Input layer - adjust the inputShape to match your image size
    discriminatorModel.add(tf.layers.conv2d({
        inputShape: [64, 64, 1],  // Grayscale images
        filters: 8,
        kernelSize: 3,
        strides: 2,
        padding: 'same',
        activation: 'relu'
    }))

    // Downsampling
    discriminatorModel.add(tf.layers.conv2d({
        filters: 16,
        kernelSize: 3,
        strides: 2,
        padding: 'same',
        activation: 'relu'
    }))

    // Further downsampling
    discriminatorModel.add(tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        strides: 2,
        padding: 'same',
        activation: 'relu'
    }))

    // Flatten the output before the dense layer
    discriminatorModel.add(tf.layers.flatten())

    // Output layer - one unit with sigmoid activation for binary classification
    discriminatorModel.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }))

    // Compile the model
    discriminatorModel.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']  // It's useful to track accuracy for the discriminator
    })

    return discriminatorModel
}
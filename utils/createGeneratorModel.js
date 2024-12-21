import * as tf from '@tensorflow/tfjs'

export function createGeneratorModel(trainingDimensions, amountInputParameters) {
    const generator = tf.sequential()

    // Start with a dense layer taking the input noise vector (size amountInputParameters)
    generator.add(tf.layers.dense({
        inputShape: [amountInputParameters],
        units: 512, // Increased units for more capacity
        useBias: false
    }))
    generator.add(tf.layers.batchNormalization())
    generator.add(tf.layers.leakyReLU({ alpha: 0.2 }))

    // Adding another dense layer to expand
    generator.add(tf.layers.dense({
        units: 1 * (trainingDimensions.y / 8) * (trainingDimensions.x / 8), // Increased units for subsequent reshape
        useBias: false
    }))
    generator.add(tf.layers.batchNormalization())
    generator.add(tf.layers.leakyReLU({ alpha: 0.2 }))

    // Reshape to a 3D volume
    generator.add(tf.layers.reshape({
        targetShape: [trainingDimensions.y / 8, trainingDimensions.x / 8, 1] // Increased depth for richer feature representation
    }))

    // Use Conv2DTranspose to upscale the image dimensions while reducing depth
    // First Conv2DTranspose layer
    // stride is the factor by which the image is upsampled
    generator.add(tf.layers.conv2dTranspose({
        filters: 128, // Increased filter count
        kernelSize: 5,
        strides: 2,
        padding: 'same',
        useBias: false
    }))
    generator.add(tf.layers.batchNormalization())
    generator.add(tf.layers.leakyReLU({ alpha: 0.2 }))

    // Second Conv2DTranspose layer
    generator.add(tf.layers.conv2dTranspose({
        filters: 64, // Increased filter count
        kernelSize: 5,
        strides: 2,
        padding: 'same',
        useBias: false
    }))
    generator.add(tf.layers.batchNormalization())
    generator.add(tf.layers.leakyReLU({ alpha: 0.2 }))

    // Third Conv2DTranspose layer to reach the final size
    generator.add(tf.layers.conv2dTranspose({
        filters: 1, // Single channel for grayscale image
        kernelSize: 5,
        strides: 2,
        padding: 'same',
        activation: 'tanh'
    }));

    return generator
}
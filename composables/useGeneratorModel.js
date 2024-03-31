import * as tf from '@tensorflow/tfjs'

export default function useGeneratorModel() {
    // define the Generative model
    // takes random noise and generates an image that can be converted to midi
    // and should sound as if it was human made music
    const generatorModel = tf.sequential()

    // input layer
    generatorModel.add(tf.layers.dense({
        inputShape: [100], // 100 paramaters for the noise
        units: 8 * 8 * 256,
        activation: 'relu'
    }))

    // reshape to 3d tensor
    generatorModel.add(tf.layers.reshape({
        targetShape: [8, 8, 256]
    }))

    // transposed convolutional layers to upscale the image
    generatorModel.add(tf.layers.conv2dTranspose({
        filters: 128,
        kernelSize: 5,
        strides: 2,
        padding: 'same',
        activation: 'relu'
    }))

    // upscale again
    generatorModel.add(tf.layers.conv2dTranspose({
        filters: 64,
        kernelSize: 5,
        strides: 2,
        padding: 'same',
        activation: 'relu'
    }))

    // output layer
    generatorModel.add(tf.layers.conv2dTranspose({
        filters: 1,
        kernelSize: 5,
        strides: 2,
        padding: 'same',
        activation: 'sigmoid' // typical for GAN
    }))

    // compile the model
    const initialLearningRate = 0.001
    const optimizer = tf.train.adam(initialLearningRate)

    // generators aren't trained directly, they are trained as part of the GAN
    generatorModel.compile({
        optimizer: optimizer,
        loss: 'binaryCrossentropy'
    })

    console.log('Generator Model summary:')
    generatorModel.summary()

    return generatorModel
}
import * as tf from '@tensorflow/tfjs'

export default function useGeneratorModel() {
    // define the Generative model
    // takes random noise and generates an image that can be converted to midi
    // and should sound as if it was human made music
    const generatorModel = tf.sequential()

    // input layer
    generatorModel.add(tf.layers.dense({
        inputShape: [300],
        units: 256,
        activation: 'relu'
    }))

    // reshape to 3d tensor
    generatorModel.add(tf.layers.reshape({
        targetShape: [64, 64, 1]
    }))

    // transposed convolutional layers to upscale the image
    generatorModel.add(tf.layers.conv2dTranspose({
        filters: 32,
        kernelSize: 4,
        strides: 1,
        padding: 'same',
        activation: 'relu'
    }))

    // upscale again
    generatorModel.add(tf.layers.conv2dTranspose({
        filters: 16,
        kernelSize: 4,
        strides: 2,
        padding: 'same',
        activation: 'relu'
    }))

    // final layer (1 filter: grayscale)
    generatorModel.add(tf.layers.conv2dTranspose({
        filters: 1,
        kernelSize: 4,
        strides: 2,
        padding: 'same',
        activation: 'tanh' // typical for GAN
    }))

    // compile the model
    // generators aren't trained directly, they are trained as part of the GAN
    generatorModel.compile({
        optimizer: 'adam',	// adam is a good optimizer for GANs
        loss: 'binaryCrossentropy'	// binary crossentropy is a good loss function for GANs
    })

    console.log('Generator Model summary:')
    generatorModel.summary()

    return generatorModel
}
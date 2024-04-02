import * as tf from '@tensorflow/tfjs'

export function createGANModel(generator, discriminator, learningRate, clipValue) {
    const gan = tf.sequential()
    gan.add(generator)
    generator.trainable = false
    gan.add(discriminator)

    gan.compile({
        loss: 'binaryCrossentropy',
        optimizer: tf.train.adam(learningRate, 0.5, 0.999),
        clipValue: clipValue
    })

    return gan
}
import * as tf from '@tensorflow/tfjs';

export default function useDiscriminatorModel() {
    const discriminator = tf.sequential()

    discriminator.add(tf.layers.conv2d({
        inputShape: [64, 64, 1],
        filters: 32,
        kernelSize: 5,
        strides: 2,
        activation: 'relu',
        padding: 'same',
    }))

    // Add a Leaky ReLU layer.
    discriminator.add(tf.layers.leakyReLU());

    // Add another convolutional layer.
    discriminator.add(tf.layers.conv2d({ filters: 64, kernelSize: 5, strides: 2, padding: 'same' }));

    // Add another Leaky ReLU layer.
    discriminator.add(tf.layers.leakyReLU());

    // Add a flatten layer.
    discriminator.add(tf.layers.flatten());

    // Add a fully connected layer.
    discriminator.add(tf.layers.dense({ units: 1 }));

    // Add a sigmoid activation layer.
    discriminator.add(tf.layers.activation({ activation: 'sigmoid' }))

    return discriminator
}

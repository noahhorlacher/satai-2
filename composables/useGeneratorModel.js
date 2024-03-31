import * as tf from '@tensorflow/tfjs';

export default function useGeneratorModel() {
    const generator = tf.sequential();

    generator.add(tf.layers.dense({
        inputShape: [100],
        units: 64 * 64 * 1,
    }))

    generator.add(tf.layers.reshape({
        targetShape: [64, 64, 1],
    }))

    return generator
}

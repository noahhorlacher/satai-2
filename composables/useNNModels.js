export function useNNModels(settings = {
    outputDimensions: {
        x: 128,
        y: 64
    },
    discriminator: {
        learningRate: 0.0002,
    },
    generator: {
        amountInputParameters: 100
    },
    GAN: {
        learningRate: 0.0002,
        clipValue: 0.01
    }
}) {
    const discriminator = createDiscriminatorModel(settings.outputDimensions, settings.discriminator.learningRate)
    const generator = createGeneratorModel(settings.outputDimensions, settings.generator.amountInputParameters)
    const gan = createGANModel(generator, discriminator, settings.GAN.learningRate, settings.GAN.clipValue)

    return {
        discriminator,
        generator,
        gan
    }
}
export const useModelStore = defineStore({
    id: 'modelStore',
    state: () => ({
        discriminator: null,
        generator: null,
        gan: null
    }),
    actions: {
        initializeModels(settings = {
                outputDimensions: {
                    x: 128,
                    y: 64
                },
                discriminator: {
                    learningRate: 0.0002,
                    clipValue: 0.01
                },
                generator: {
                    amountInputParameters: 100
                },
                GAN: {
                    learningRate: 0.0002,
                    clipValue: 0.01
                }
            }) {
            this.discriminator = createDiscriminatorModel(settings.outputDimensions, settings.discriminator.learningRate, settings.discriminator.clipValue)
            this.generator = createGeneratorModel(settings.outputDimensions, settings.generator.amountInputParameters)
            this.gan = createGANModel(generator, discriminator, settings.GAN.learningRate, settings.GAN.clipValue)
        }
    },
    persist: {
        storage: localStorage,
    }
})
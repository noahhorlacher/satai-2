<script setup>
import * as tf from '@tensorflow/tfjs'
import JSZip from 'jszip'
import pako from 'pako'
import { Midi } from '@tonejs/midi'

/*
    architecture of discriminator and generator
    https://medium.com/ee-460j-final-project/generating-music-with-a-generative-adversarial-network-8d3f68a33096
*/

const midiVelocityThreshold = 0.1

const chartOptions = reactive({
    chart: {
        type: 'line'
    },
    xaxis: {
        labels: {
            show: false
        },
        categories: [],
    }
})

const chartSeries = reactive([
    {
        name: 'GAN Loss',
        data: []
    },
    {
        name: 'Discriminator Loss',
        data: []
    }
])

const fileInput = ref()
const selectedSamplesName = ref()

let trainingData = []

const { statusMessage } = toRefs(useStatusMessageStore())

const discriminatorLearningRate = 0.0001
const ganLearningRate = 0.0001

const discriminator = tf.sequential();

// First Convolutional layer - Assuming a larger kernel and stride to reduce dimension significantly
discriminator.add(tf.layers.conv2d({
    inputShape: [64, 64, 1],
    filters: 128,
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
    filters: 64,
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
    filters: 32,
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

const generator = tf.sequential();

// Start with a dense layer taking the input noise vector (size 100)
generator.add(tf.layers.dense({
    inputShape: [100],
    units: 512, // Increased units for more capacity
    useBias: false
}));
generator.add(tf.layers.batchNormalization());
generator.add(tf.layers.leakyReLU({ alpha: 0.2 }));

// Adding another dense layer to expand
generator.add(tf.layers.dense({
    units: 256 * 8 * 8, // Increased units for subsequent reshape
    useBias: false
}));
generator.add(tf.layers.batchNormalization());
generator.add(tf.layers.leakyReLU({ alpha: 0.2 }));

// Reshape to a 3D volume
generator.add(tf.layers.reshape({
    targetShape: [8, 8, 256] // Increased depth for richer feature representation
}));

// Use Conv2DTranspose to upscale the image dimensions while reducing depth
// First Conv2DTranspose layer
generator.add(tf.layers.conv2dTranspose({
    filters: 128, // Increased filter count
    kernelSize: 5,
    strides: 2,
    padding: 'same',
    useBias: false
}));
generator.add(tf.layers.batchNormalization());
generator.add(tf.layers.leakyReLU({ alpha: 0.2 }));

// Second Conv2DTranspose layer
generator.add(tf.layers.conv2dTranspose({
    filters: 64, // Increased filter count
    kernelSize: 5,
    strides: 2,
    padding: 'same',
    useBias: false
}));
generator.add(tf.layers.batchNormalization());
generator.add(tf.layers.leakyReLU({ alpha: 0.2 }));

// Third Conv2DTranspose layer to reach the final size
generator.add(tf.layers.conv2dTranspose({
    filters: 1, // Single channel for grayscale image
    kernelSize: 5,
    strides: 2,
    padding: 'same',
    activation: 'sigmoid'
}));

// Compile the discriminator
discriminator.compile({
    loss: 'binaryCrossentropy',
    optimizer: tf.train.adam(discriminatorLearningRate, 0.5, 0.999)
})

const gan = tf.sequential()
gan.add(generator)
generator.trainable = false
gan.add(discriminator)

gan.compile({
    loss: 'binaryCrossentropy',
    optimizer: tf.train.adam(ganLearningRate, 0.5, 0.999)
})

const epochs = 100
async function trainModel() {
    let backend = tf.getBackend()

    for (let i = 0; i < epochs; i++) {
        chartOptions.xaxis.categories.push(`Epoch ${i}`)

        let realImagesArray = getRandomSamples(10)

        // calculate to amount of numbers inside. It should be 10 * 64 * 64 
        while (realImagesArray.length < 10 || realImagesArray.some(image => image === undefined) || realImagesArray.some(image => image.length !== 64) || realImagesArray.some(image => image.some(row => row.length !== 64))) {
            realImagesArray = getRandomSamples(10)
        }

        // Convert each 2D image in the array to a 3D image by adding an extra dimension
        realImagesArray = realImagesArray.map(image => {
            return image.map(row => {
                return row.map(value => {
                    return [value]; // Adds an extra dimension
                });
            });
        });

        const realImages = tf.tensor4d(realImagesArray, [10, 64, 64, 1]);

        // Check if the conversion is correct
        // Generate a batch of fake images.
        const noise = tf.randomNormal([10, 100])
        const fakeImages = generator.predict(noise)

        // Create a batch of labels for the real and fake images.
        // With label smoothing
        const realLabels = tf.ones([10, 1]).mul(0.9)
        const fakeLabels = tf.zeros([10, 1]).mul(0.1)

        try {
            // Train the discriminator on real and fake images
            let dLossReal = await discriminator.trainOnBatch(realImages, realLabels)
            let dLossFake = await discriminator.trainOnBatch(fakeImages, fakeLabels)

            const dLoss = (dLossReal + dLossFake) / 2

            // Train the generator via the gan model
            const misleadingLabels = tf.ones([10, 1])
            const gLoss = await gan.trainOnBatch(noise, misleadingLabels)

            chartSeries[0].data.push(gLoss.toFixed(5))
            chartSeries[1].data.push(dLoss.toFixed(5))

            statusMessage.value = `Using [${backend}]\nTrained epoch ${i + 1} of ${epochs}.\nGAN loss: ${gLoss}\nDiscriminator loss: ${dLoss}`
        } catch (error) {
            console.error('Error training:', error)
            statusMessage.value = 'Error training. Check console'
        }

        await tf.nextFrame()
    }

    statusMessage.value = `Training completed. ${epochs} epochs trained. GAN loss: ${chartSeries[0].data.at(-1)}. Discriminator loss: ${chartSeries[1].data.at(-1)}`
}

function getRandomSamples(n) {
    // Ensure trainingData is not empty
    if (!trainingData || trainingData.length === 0) {
        console.error("No training data available")
        return []
    }

    // get a random batch from the trainingData
    let randomBatchIndex = Math.floor(Math.random() * trainingData.length)

    const randomBatchFile = trainingData[randomBatchIndex]

    // unzip the batch
    const unzippedData = pako.ungzip(randomBatchFile)
    const array = JSON.parse(new TextDecoder().decode(unzippedData))

    // get n random elements from array
    const result = []
    for (let i = 0; i < n; i++) {
        const randomIndex = Math.floor(Math.random() * array.length)
        result.push(array[randomIndex])
    }

    return result
}

async function handleFileImport(event) {
    const file = event.target.files[0]
    if (!file) return

    const reader = new FileReader()

    reader.onload = async (e) => {
        try {
            statusMessage.value = 'Importing training data...'
            trainingData = []

            selectedSamplesName.value = event.target.files[0].name

            // Decompress the zip
            let zip = new JSZip()
            let zipData = await zip.loadAsync(e.target.result)
            zip = null

            Object.keys(zipData.files).forEach(async (filename) => {
                const fileData = await zipData.files[filename].async('uint8array')
                trainingData.push(fileData)
            })

            zipData = null

            statusMessage.value = 'Training data imported successfully.'
        } catch (error) {
            statusMessage.value = `Error importing training data: ${error.message}`
        }
    }

    reader.readAsArrayBuffer(file)
}

function generate() {
    const noise = tf.randomNormal([1, 100])
    const generatedData = generator.predict(noise)
    const data = generatedData.arraySync()

    // Convert the data to a MIDI files
    const midi = new Midi()

    // Add a track
    const track = midi.addTrack()

    console.log('data', data)

    // Add notes
    data[0].forEach((row, rowIndex) => {
        row.forEach((value, columnIndex) => {
            // clamp 
            const velocity = Math.max(0, Math.min(1.0, value))

            if (velocity < midiVelocityThreshold) return

            track.addNote({
                midi: rowIndex + 36, // startOctave = 3
                time: columnIndex,
                duration: 100,
                velocity: velocity
            })
        })
    })

    // Convert the MIDI to a blob
    const blob = new Blob([midi.toArray()], { type: 'audio/midi' })

    // Download the blob
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'generated.mid'
    a.click()
    a.remove()
}
</script>

<template>
    <div>
        <div class="">
            <h3 class="text-sm mt-6 mb-2">Loss/Epoch</h3>
            <apexchart type="line" height="300px" :series="chartSeries" :options="chartOptions" />
        </div>
        <!-- Hidden file input -->
        <h3 class="text-sm mt-6 mb-2">Load Processed Training Samples (.zip)</h3>
        <div class="mb-4">
            <p v-if="selectedSamplesName" class="text-xs mb-4 text-gray-500 font-bold">Selected samples:<br><span
                    class="italic font-regular">{{
                selectedSamplesName }}</span></p>
            <el-button @click="fileInput.click()" size="large">
                <icon class="mr-2" name="material-symbols:attach-file" size="1.5em" />
                Choose File
            </el-button>
            <input class="hidden" type="file" ref="fileInput" @change="handleFileImport" accept=".zip" />
        </div>
    </div>

    <el-button @click="trainModel">Train Model</el-button>
    <el-button @click="generate">Generate</el-button>
</template>

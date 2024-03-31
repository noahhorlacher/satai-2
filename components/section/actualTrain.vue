<script setup>
import * as tf from '@tensorflow/tfjs'
import JSZip from 'jszip'
import pako from 'pako'
import { Midi } from '@tonejs/midi'

const chartData = reactive({
    labels: [],
    datasets: [
        {
            label: 'Generator Loss',
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1,
            data: []
        },
        {
            label: 'Discriminator Loss',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 1,
            data: []
        }
    ]
})

const chartOptions = reactive({
    responsive: true,
    maintainAspectRatio: false,
})

const fileInput = ref()
const selectedSamplesName = ref()

let trainingData = []

const { statusMessage } = toRefs(useStatusMessageStore())

const discriminatorLearningRate = 0.0002
const ganLearningRate = 0.0001

// initialize GAN model
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

const generator = tf.sequential();

generator.add(tf.layers.dense({
    inputShape: [100],
    units: 64 * 64 * 1,
}))

generator.add(tf.layers.reshape({
    targetShape: [64, 64, 1],
}))

// Compile the discriminator
discriminator.compile({
    loss: 'binaryCrossentropy',
    optimizer: tf.train.sgd(discriminatorLearningRate, 0.5, 0.999)
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
async function train() {
    for (let i = 0; i < epochs; i++) {
        console.log(`Training epoch ${i + 1} of ${epochs}...`)
        statusMessage.value = `Training epoch ${i + 1} of ${epochs}...`

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
        const realLabels = tf.ones([10, 1])
        const fakeLabels = tf.zeros([10, 1])

        try {
            // Train the discriminator on real and fake images.
            await discriminator.trainOnBatch(realImages, realLabels)
            const dLoss = await discriminator.trainOnBatch(fakeImages, fakeLabels)

            chartData.datasets[1].data.push(dLoss)

            // Train the generator via the gan model
            const misleadingLabels = tf.ones([10, 1])
            const gLoss = await gan.trainOnBatch(noise, misleadingLabels)

            chartData.datasets[0].data.push(gLoss)

            console.log('gLoss', gLoss, 'dLoss', dLoss)
            statusMessage.value = `gLoss: ${gLoss}, dLoss: ${dLoss}`
            // Optional: Visualize or evaluate the results as needed
            // ...
        } catch (error) {
            console.error('Error training:', error)
            statusMessage.value = 'Error training. Check console'
        }
    }

    console.log('Training complete')
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
    const scaledData = generatedData.mul(255)
    const data = scaledData.arraySync()

    // Convert the data to a MIDI files
    const midi = new Midi()

    // Add a track
    const track = midi.addTrack()

    // Add notes
    data[0].forEach((row, rowIndex) => {
        row.forEach((value, columnIndex) => {
            if (value > 127) {
                value = 127
            } else if (value < 0) {
                value = 0
            }

            track.addNote({
                midi: value,
                time: columnIndex,
                duration: 100,
                velocity: value
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
        <!-- <Line :data="chartData" :options="chartOptions" /> -->
        <!-- Hidden file input -->
        <h3 class="text-sm mt-6 mb-2">Load Processed Training Samples (.zip)</h3>
        <p v-if="selectedSamplesName" class="text-xs mb-4 text-gray-500 font-bold">Selected samples:<br><span
                class="italic font-regular">{{
            selectedSamplesName }}</span></p>
        <el-button @click="fileInput.click()" size="large">
            <icon class="mr-2" name="material-symbols:attach-file" size="1.5em" />
            Choose File
        </el-button>
        <input class="hidden" type="file" ref="fileInput" @change="handleFileImport" accept=".zip" />
    </div>

    <el-button @click="train">Train Model</el-button>
    <el-button @click="generate">Generate</el-button>
</template>

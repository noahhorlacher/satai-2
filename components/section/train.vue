<script setup>
import * as tf from '@tensorflow/tfjs'
import JSZip from 'jszip'
import pako from 'pako'
import { Midi } from '@tonejs/midi'

const { statusMessage } = toRefs(useStatusMessageStore())

statusMessage.value = 'Press an action button to begin...'

const busy = ref(false)

/*
    architecture of discriminator and generator
    https://medium.com/ee-460j-final-project/generating-music-with-a-generative-adversarial-network-8d3f68a33096
*/

// While training, reuse the same batch of samples to pick randomly from for a few epochs
// so the amount of ungzipping is reduced
const pickNewBatchEveryNEpochs = 50
const currentLoadedSampleBatch = {
    batchName: '',
    timesLoaded: pickNewBatchEveryNEpochs,
    data: []
}

const chartOptions = reactive({
    chart: {
        type: 'line'
    },
    xaxis: {
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

const discriminatorLearningRate = 0.00015
const ganLearningRate = 0.0001
const clipValue = 0.01

let epochs = 600
let batchSize = 32

// epochs that save the model and generate previews
const saveEpochs = [
    0, 1, 3, 5, 10, 15, 20, 30, 40, 50, 70, 90, 100, 120, 150, 180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 60000, 70000, 80000, 90000, 100000, 120000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 600000, 700000, 800000, 900000, 1000000
]

// for generating
const midiConfidenceThreshold = 0.1

const trainedForEpochs = ref(0)

const discriminator = tf.sequential()

const epochsSelection = ref(epochs)
const batchSizeSelection = ref(batchSize)

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

// on stability issues:
// use Wasserstein loss or mean squared error loss

// Compile the discriminator
discriminator.compile({
    loss: 'binaryCrossentropy',
    optimizer: tf.train.adam(discriminatorLearningRate, 0.5, 0.999),
    clipValue: clipValue
})

const gan = tf.sequential()
gan.add(generator)
generator.trainable = false
gan.add(discriminator)

gan.compile({
    loss: 'binaryCrossentropy',
    optimizer: tf.train.adam(ganLearningRate, 0.5, 0.999),
    clipValue: clipValue
})

let trainingStartDateTime
async function trainModel() {
    statusMessage.value = 'Starting training...'

    epochs = epochsSelection.value
    batchSize = batchSizeSelection.value

    busy.value = true
    let backend = tf.getBackend()
    trainingStartDateTime = new Date()

    for (let i = 0; i < epochs; i++) {
        let realImagesArray = await getRandomSamples(batchSize)

        // Convert each 2D image in the array to a 3D image by adding an extra dimension
        realImagesArray = realImagesArray.map(image => {
            return image.map(row => {
                return row.map(value => {
                    return [value] // Adds an extra dimension
                })
            })
        })

        const realImages = tf.tensor4d(realImagesArray, [batchSize, 64, 64, 1]);

        // Check if the conversion is correct
        // Generate a batch of fake images.
        const noise = tf.randomNormal([batchSize, 100])
        const fakeImages = generator.predict(noise)

        // Create a batch of labels for the real and fake images.
        // With label smoothing
        const realLabels = tf.ones([batchSize, 1]).mul(0.9)
        const fakeLabels = tf.zeros([batchSize, 1]).mul(0.1)

        try {
            // Train the discriminator on real and fake images
            let dLossReal = await discriminator.trainOnBatch(realImages, realLabels)
            let dLossFake = await discriminator.trainOnBatch(fakeImages, fakeLabels)

            const dLoss = (dLossReal + dLossFake) / 2

            // Train the generator via the gan model
            const misleadingLabels = tf.ones([batchSize, 1]).mul(0.9)
            const gLoss = await gan.trainOnBatch(noise, misleadingLabels)

            chartSeries[0].data.push(gLoss.toFixed(5))
            chartSeries[1].data.push(dLoss.toFixed(5))

            statusMessage.value = `💡 Using backend [${backend}] to train\n⏲ Started training on ${trainingStartDateTime.toLocaleString()}\n🥊 Trained epoch ${i + 1} of ${epochs}.\n🎨 GAN loss: ${gLoss}\n👓 Discriminator loss: ${dLoss}`
        } catch (error) {
            console.error('Error training:', error)
            statusMessage.value = 'Error training. Check console'
        }

        trainedForEpochs.value++

        if (trainedForEpochs.value % 100 == 0) {
            chartOptions.xaxis.categories.push(`Epoch ${i}`)
        }

        if (saveEpochs.includes(trainedForEpochs.value)) {
            previewCanvas()
        }

        await tf.nextFrame() // keep ui responsive
    }

    busy.value = false

    statusMessage.value = `Training completed. ${epochs} epochs trained. GAN loss: ${chartSeries[0].data.at(-1)}. Discriminator loss: ${chartSeries[1].data.at(-1)}`
}

async function getRandomSamples(n) {
    // Ensure trainingData is not empty
    if (!trainingData || trainingData.length === 0) {
        console.error("No training data available")
        return []
    }

    // pick a random batch if batch has been reused enough
    if (currentLoadedSampleBatch.timesLoaded >= pickNewBatchEveryNEpochs) {
        statusMessage.value = 'Picking a new batch of samples...'
        currentLoadedSampleBatch.batchName = trainingData[Math.floor(Math.random() * trainingData.length)]
        currentLoadedSampleBatch.timesLoaded = 0

        let fileData = trainingData[Math.floor(Math.random() * trainingData.length)]

        let unzippedData = await pako.ungzip(fileData)
        fileData = null


        let arrays = JSON.parse(new TextDecoder().decode(unzippedData))
        unzippedData = null

        currentLoadedSampleBatch.data = arrays
    }

    // get n random elements from trainingData
    const result = []
    for (let i = 0; i < n; i++) {
        const randomIndex = Math.floor(Math.random() * trainingData.length)
        result.push(currentLoadedSampleBatch.data[randomIndex])
    }

    return result
}

async function handleFileImport(event) {
    busy.value = true
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

            // load gzipped batches into trainingData array
            for (const filename of Object.keys(zipData.files)) {
                statusMessage.value = `Unzipping ${filename}...`
                let fileData = await zipData.files[filename].async('uint8array')
                trainingData.push(fileData)
            }

            zipData = null

            busy.value = false

            statusMessage.value = 'Training data imported successfully.'
        } catch (error) {
            statusMessage.value = `Error importing training data: ${error.message}`
        }
    }

    reader.readAsArrayBuffer(file)
}

function generateMIDI() {
    busy.value = true

    const noise = tf.randomNormal([1, 100])
    const generatedData = generator.predict(noise)
    let data = generatedData.arraySync()
    data = data[0]
    data = data.map(row => {
        return row.map(value => {
            return value[0] // remove unnecessary dimension
        })
    })

    // Convert the data to a MIDI files
    const midi = new Midi()

    // Add a track
    const track = midi.addTrack()

    for (let y = 0; y < data.length; y++) {
        for (let x = 0; x < data[y].length; x++) {
            // clamp 
            const velocity = Math.max(0, Math.min(1.0, data[y][x][0]))

            if (velocity < midiConfidenceThreshold) return

            track.addNote({
                midi: y + 36, // startOctave = 3
                time: x,
                duration: 100,
                velocity: velocity
            })
        }
    }

    // Convert the MIDI to a blob
    const blob = new Blob([midi.toArray()], { type: 'audio/midi' })

    // Download the blob
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'generated.mid'
    a.click()
    a.remove()

    busy.value = false
}

const canvasPreview = ref()
const previewImages = ref([])
function previewCanvas() {
    busy.value = true

    const noise = tf.randomNormal([1, 100])
    const generatedData = generator.predict(noise)
    let data = generatedData.arraySync()
    data = data[0]
    data = data.map(row => {
        return row.map(value => {
            return value[0] // remove unnecessary dimension
        })
    })

    const ctx = canvasPreview.value.getContext('2d')
    ctx.imageSmoothingEnabled = false

    ctx.clearRect(0, 0, 64, 64)

    const imageData = ctx.createImageData(64, 64)

    // loop over each pixel and set pixel brightness to the generated data$
    for (let y = 0; y < data.length; y++) {
        for (let x = 0; x < data[y].length; x++) {
            let value = data[y][x]
            const pixelIndex = (y * 64 + x) * 4

            // threshold
            if (value < midiConfidenceThreshold) {
                value = 0
            }

            // draw
            imageData.data[pixelIndex] = Math.round(value * 255)
            imageData.data[pixelIndex + 1] = Math.round(value * 255)
            imageData.data[pixelIndex + 2] = Math.round(value * 255)

            imageData.data[pixelIndex + 3] = 255
        }
    }

    // update canvas with new image data
    ctx.putImageData(imageData, 0, 0)
    canvasPreview.value.toBlob((blob) => {
        const url = URL.createObjectURL(blob)
        previewImages.value.push({
            description: `${trainedForEpochs.value} Epochs`,
            src: url
        })
    })

    busy.value = false
}

async function previewSample() {
    busy.value = true

    const sample = await getRandomSamples(1)

    const ctx = canvasPreview.value.getContext('2d')
    ctx.imageSmoothingEnabled = false

    ctx.clearRect(0, 0, 64, 64)

    const imageData = ctx.createImageData(64, 64)

    // loop over each pixel and set pixel brightness to the generated data$
    for (let y = 0; y < sample[0].length; y++) {
        for (let x = 0; x < sample[0][y].length; x++) {
            let value = sample[0][y][x]
            const pixelIndex = (y * 64 + x) * 4

            // threshold
            if (value < midiConfidenceThreshold) {
                value = 0
            }

            // draw
            imageData.data[pixelIndex] = Math.round(value * 255)
            imageData.data[pixelIndex + 1] = Math.round(value * 255)
            imageData.data[pixelIndex + 2] = Math.round(value * 255)

            imageData.data[pixelIndex + 3] = 255
        }
    }

    // update canvas with new image data
    ctx.putImageData(imageData, 0, 0)
    canvasPreview.value.toBlob((blob) => {
        const url = URL.createObjectURL(blob)
        previewImages.value.push({
            description: `Sample preview`,
            src: url
        })
    })

    busy.value = false
}

function nextSave() {
    const nextSaveEpoch = saveEpochs.find(epoch => epoch > trainedForEpochs.value)
    return nextSaveEpoch || 'None'
}
</script>

<template>
    <app-section title="Train">

        <div class="flex flex-row justify-between items-center mb-2">
            <h3 class="text-sm">Status</h3>
        </div>
        <div
            class="text-md rounded-md shadow-md bg-gray-900 text-green-400 py-2 px-4 mb-2 font-mono whitespace-pre-line">
            {{
                statusMessage }}
        </div>

        <div>
            <div v-if="previewImages.length > 0">
                <h3 class="text-sm mt-6 mb-2">
                    Latest Preview ({{ previewImages.at(-1).description }})
                    Next save at epoch {{ nextSave() }}
                </h3>
                <img :src="previewImages.at(-1).src" class="w-1/2 max-w-[350px] h-auto"
                    style="image-rendering: pixelated" />
            </div>
            <div>
                <h3 class="text-xs mt-6 mb-2">Trained for {{ trainedForEpochs }} epochs total</h3>
                <h3 class="text-sm mt-6 mb-2">Loss/Epoch</h3>
                <apexchart type="line" height="300px" :series="chartSeries" :options="chartOptions" />
            </div>

            <h3 class="text-sm mt-6 mb-2">Load Processed Training Samples (.zip)</h3>

            <div class="mb-4">
                <p v-if="selectedSamplesName" class="text-xs mb-4 text-gray-500 font-bold">
                    Selected samples:<br>
                    <span class="italic font-regular">
                        {{ selectedSamplesName }}
                    </span>
                </p>
                <el-button :disabled="busy" @click="fileInput.click()" size="large">
                    <icon class="mr-2" name="material-symbols:attach-file" size="1.5em" />
                    Choose File
                </el-button>
                <!-- Hidden file input -->
                <input class="hidden" type="file" ref="fileInput" @change="handleFileImport" accept=".zip" />
            </div>
        </div>

        <h3 class="text-sm mt-6 mb-2">Train</h3>
        <div class="flex flex-row gap-x-4 mb-4">
            <el-button @click="trainModel" :disabled="!trainingData || trainingData.length == 0 || busy">
                Train Model
            </el-button>
        </div>

        <h3 class="text-sm mt-6 mb-2">Training Settings</h3>
        <div class="flex flex-row gap-x-4 mb-4">
            <div>
                <p class="text-xs mb-1">Epochs</p>
                <el-input-number :disabled="!trainingData || trainingData.length == 0 || busy"
                    v-model="epochsSelection" />
            </div>

            <div>
                <p class="text-xs mb-1">Batch Size</p>
                <el-input-number :disabled="!trainingData || trainingData.length == 0 || busy"
                    v-model="batchSizeSelection" />
            </div>
        </div>

        <div>
            <h3 class="text-sm mt-6 mb-2">Test model</h3>
            <!-- <el-button @click="generate" :disabled="busy || trainedForEpochs == 0"> -->
            <el-button @click="generateMIDI" :disabled="busy">
                Generate MIDI
            </el-button>
            <el-button @click="previewCanvas" :disabled="busy">
                Preview Image
            </el-button>
            <el-button @click="previewSample" :disabled="busy">
                Preview Sample
            </el-button>
        </div>

        <div>
            <h3 class="text-sm mt-6 mb-2">Previews</h3>

            <div v-if="previewImages.length == 0" class="text-xs text-gray-500 mb-4">
                No previews available. Train or click "Preview Image" to generate previews.
            </div>

            <div class="flex flex-row-reverse flex-wrap mt-8 gap-4 justify-center">
                <figure v-for="(previewImage, index) of previewImages" class="grow shrink w-1/4 h-auto">
                    <figcaption class="text-xs mb-2">{{ previewImage.description }}</figcaption>
                    <img class="w-full" style="image-rendering: pixelated" :src="previewImage.src"
                        :key="`previewImage-${index}`" />
                </figure>
            </div>

            <canvas class="hidden" width="64" height="64" ref="canvasPreview" />
        </div>
    </app-section>
</template>
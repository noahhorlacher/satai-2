<script setup>
import * as tf from '@tensorflow/tfjs'
import JSZip from 'jszip'
import pako from 'pako'

const loading = ref(false)

const { trainingData } = toRefs(useTrainingDataStore())
const { statusMessage } = toRefs(useStatusMessageStore())

statusMessage.value = 'Press an action button to begin...'

const trainingDataUrl = '/data/midi/midi_all.zip'
const dataPreprocessorBatchSize = 200

// load midi data
// my midi files were already pre-processed to only have 1 track (track 0)
let midiFiles = []
async function loadData() {
    loading.value = true

    midiFiles = []

    try {
        statusMessage.value = 'Downloading midi files ZIP...'
        const response = await fetch(trainingDataUrl)
        const zipData = await response.blob()
        const jszip = new JSZip()

        // load zip content
        const zip = await jszip.loadAsync(zipData)

        let idx = 1
        let filesAmount = Object.keys(zip.files).length

        // Unzip each file
        for (const [filename, file] of Object.entries(zip.files)) {
            statusMessage.value = `Unzipping midi file ${idx++}/${filesAmount}\n(${filename})`

            if (filename.endsWith('.midi') || filename.endsWith('.mid')) {
                const midiFile = await file.async("arraybuffer")
                midiFiles.push(midiFile)
            }
        }

        statusMessage.value = `Loaded ${midiFiles.length} MIDI files.`
    } catch (err) {
        console.error('Error loading midi files:', err)
        statusMessage.value = 'Error loading midi files. Check console'
    }
    loading.value = false

}

// preprocess midi data for training
async function preprocessData() {
    loading.value = true

    let batchSizes = dataPreprocessorBatchSize > 0 ? dataPreprocessorBatchSize : midiFiles.length
    const amountBatches = Math.ceil(midiFiles.length / batchSizes)

    const zipFile = new JSZip()

    for (let i = 0; i < midiFiles.length; i += batchSizes) {
        const batch = midiFiles.slice(i, i + batchSizes)

        let currentBatch = i / batchSizes + 1
        const _preprocessedData = await MIDIPreprocessor.preprocess(batch, undefined, `${currentBatch} of ${amountBatches}`)

        // download
        const data = JSON.stringify(_preprocessedData)
        const gzipData = pako.gzip(data)

        zipFile.file(`batch-${currentBatch}_of_${amountBatches}.json.gz`, gzipData)
    }

    // Compress the entire array of preprocessed data
    statusMessage.value = `Compressing all data`
    const zipData = await zipFile.generateAsync({ type: 'uint8array' })

    exportTrainingData(zipData, 'SatAi-Training-Data')

    statusMessage.value = 'Preprocessing complete.'
    loading.value = false
}

async function createTrainingData() {
    await loadData()
    await preprocessData()
}

function exportTrainingData(dataToExport, fileName) {
    statusMessage.value = `Downloading training data...`

    const blob = new Blob([dataToExport], { type: 'application/zip' });
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${fileName}.zip`
    a.click()
    URL.revokeObjectURL(url)
    a.remove()
}

// initialize GAN model
const generatorModel = ref()
const discriminatorModel = ref()

function initializeModel() {
    loading.value = true
    generatorModel.value = useGeneratorModel()
    discriminatorModel.value = useDiscriminatorModel()
    loading.value = false

    console.log('Generator Model summary:')
    generatorModel.value.summary()

    console.log('Discriminator Model summary:')
    discriminatorModel.value.summary()

    statusMessage.value = 'Models initialized.'
}
function exponentialDecay(initialRate, decayRate, decaySteps, step) {
    return initialRate * Math.pow(decayRate, Math.floor(step / decaySteps));
}

function trainModel() {
    const epochs = 10
    const epochsToSave = [1, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000]
    const batchSize = 128
    const numBatches = 200

    const initialRate = 0.001
    const decayRate = 0.96
    const decaySteps = 1000
    let step = 0

    for (let epoch = 0; epoch < epochs; epoch++) {
        let totalDLoss, totalGLoss
        let numBatchesProcessed = 0

        // batch processing
        for (let batch = 0; batch < numBatches; batch++) {
            // update leaning rate
            const currentLearningRate = exponentialDecay(
                initialRate,
                decayRate,
                decaySteps,
                step)

            // Recreate the optimizer with the new learning rate
            const newOptimizer = tf.train.adam(currentLearningRate)
            generatorModel.optimizer = newOptimizer
            discriminatorModel.optimizer = newOptimizer

            // ----- train discriminator -----
            // generate fake data using generator
            const noise = tf.randomNormal([batchSize / 2, 100])

            console.log('model', generatorModel.value)
            const fakeData = generatorModel.value.predict(noise)
            console.log('fakeData', fakeData)

            // get real data (randomly sampled from training data)
            const realData = pickRandomElements(trainingData.value, batchSize / 2)

            // combine and shuffle fake and real data
            const combinedData = tf.concat([realData, fakeData])
            // combine and shuffle labels (real: 1, fake: 0)
            const realLabels = tf.ones([batchSize / 2, 1]); // Labels for real data
            const fakeLabels = tf.zeros([batchSize / 2, 1]); // Labels for fake data
            const labels = tf.concat([realLabels, fakeLabels])

            // train discriminator
            const dLoss = discriminatorModel.value.trainOnBatch(combinedData, labels)
            totalDLoss += dLoss

            // ----- Train Generator -----
            // generate fake data (new noise)
            const noiseForGen = tf.randomNormal([batchSize, 100])
            // Create labels - all marked as real (to fool the discriminator)
            const misleadingLabels = tf.ones([batchSize, 1])

            // Freeze discriminator weights during generator training
            discriminatorModel.value.trainable = false

            // train generator via discriminator
            const gLoss = generatorModel.value.trainOnBatch(noiseForGen, misleadingLabels)
            totalGLoss += gLoss

            // unfreeze discriminator weights
            discriminatorModel.value.trainable = true

            // Log losses, update UI
            statusMessage.value = `Epoch ${epoch + 1}/${epochs}, Batch ${batch + 1}/${numBatches}, D Loss: ${dLoss}, G Loss: ${gLoss}`

            numBatchesProcessed++
            step++
        }

        // Calculate the average loss over all batches
        const avgDLoss = totalDLoss / numBatchesProcessed;
        const avgGLoss = totalGLoss / numBatchesProcessed;

        // Log the average loss for the epoch
        statusMessage.value = `Epoch ${epoch + 1}/${epochs}, Avg D Loss: ${avgDLoss}, Avg G Loss: ${avgGLoss}`

        // save model if saveable epoch
        if (epochsToSave.includes(epoch + 1)) {
            generatorModel.value.save(`localstorage://${epoch + 1}-generator`)
            discriminatorModel.value.save(`localstorage://${epoch + 1}-discriminator`)
        }
    }
}

function generateMidiMatrix() {
    const generatedData = generatorModel.predict(someInputNoise)
    const scaledData = generatedData.mul(255)
}

function pickRandomElements(arr, n) {
    let result = arr.slice()
    for (let i = result.length - 1; i > 0; i--) {
        // Generate random index
        let j = Math.floor(Math.random() * (i + 1))
        // Swap elements at indices i and j
        [result[i], result[j]] = [result[j], result[i]]
    }
    // Return the first n elements
    return result.slice(0, n)
}
</script>

<template>
    <app-section title="Train">
        <div class="flex flex-row justify-between items-center mb-2">
            <h3 class="text-sm">Status</h3>
        </div>
        <div class="text-md rounded-md bg-gray-900 text-green-400 py-2 px-4 mb-2 font-mono whitespace-pre-line">{{
            statusMessage }}
        </div>

        <h3 class="text-sm mt-6 mb-2">Actions</h3>
        <el-button @click="createTrainingData" :disabled="loading">Create training data</el-button>
        <el-button @click="initializeModel" :disabled="loading">Initialize
            Model</el-button>
        <el-button :disabled="loading" @click="trainModel">Train Model</el-button>
        <el-button :disabled="loading">Save Model</el-button>
    </app-section>
</template>
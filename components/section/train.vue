<script setup>
import * as tf from '@tensorflow/tfjs'
import JSZip from 'jszip'
import pako from 'pako'

const loading = ref(false)

let trainingData = []
const fileInput = ref()
const { statusMessage } = toRefs(useStatusMessageStore())

const selectedSamplesName = ref()

statusMessage.value = 'Press an action button to begin...'

const dataPreprocessorBatchSize = 200

const trainingZips = reactive([
    { label: `midi-all.zip (49.7 MB) (14'718 files)`, value: 'midi-all.zip' },
    { label: `midi-test-large.zip (4.54 MB) (1'881 files)`, value: 'midi-test-large.zip' },
    { label: `midi-test-small.zip (80.6 KB) (45 files)`, value: 'midi-test-small.zip' },
])
const selectedTrainingDataUrl = ref(trainingZips[0].value)

// load midi data
// my midi files were already pre-processed to only have 1 track (track 0)
let midiFiles = []
async function loadData() {
    loading.value = true

    midiFiles = []

    try {
        statusMessage.value = `Downloading ${selectedTrainingDataUrl.value}...`
        const response = await fetch(`/data/midi/${selectedTrainingDataUrl.value}`)
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

        zipFile.file(`batch-${currentBatch}-of-${amountBatches}_size-${batch.length}.json.gz`, gzipData)
    }

    // Compress the entire array of preprocessed data
    statusMessage.value = `Compressing all data`
    const zipData = await zipFile.generateAsync({ type: 'uint8array' })

    exportTrainingData(zipData, `SatAi-Training-Data_${selectedTrainingDataUrl.value.split('/').pop().replace('.zip', '')}_batch-size-${dataPreprocessorBatchSize}_${amountBatches}-batches`)

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
let generatorModel
let discriminatorModel
function initializeModel() {
    loading.value = true
    generatorModel = useGeneratorModel()
    discriminatorModel = useDiscriminatorModel()
    loading.value = false

    console.log('Generator Model summary:')
    generatorModel.summary()

    console.log('Discriminator Model summary:')
    discriminatorModel.summary()

    statusMessage.value = 'Models initialized.'
}
function exponentialDecay(initialRate, decayRate, decaySteps, step) {
    return initialRate * Math.pow(decayRate, Math.floor(step / decaySteps));
}

// Function to set trainable status for all layers of a model
function setTrainableStatus(model, isTrainable) {
    model.layers.forEach(layer => {
        layer.trainable = isTrainable
    })
}

async function trainModel() {
    // Training parameters
    const epochs = 10
    const batchSize = 128
    const zDim = 100 // Dimensionality of the generator input
    const initialRate = 0.001
    const numBatches = 200

    // Compile both models
    const optimizer = tf.train.adam(initialRate)
    generatorModel.compile({ optimizer: optimizer, loss: 'binaryCrossentropy' })
    discriminatorModel.compile({ optimizer: optimizer, loss: 'binaryCrossentropy', metrics: ['accuracy'] })

    // Training loop
    for (let epoch = 0; epoch < epochs; epoch++) {
        for (let batch = 0; batch < numBatches / batchSize; batch++) {
            console.log('batch', batch + 1, 'of', numBatches, 'in epoch', epoch + 1, 'of', epochs)

            // Generate noise for generator input
            const z = tf.randomNormal([batchSize / 2, zDim])

            // Generate fake MIDI data
            const fakeMIDI = generatorModel.predict(z)

            // Get real MIDI data batch
            const realMIDIArray = getRandomSamples(batchSize / 2)
            const realMIDITensor = tf.tensor4d(realMIDIArray, [batchSize / 2, fakeMIDI.shape[1], fakeMIDI.shape[2], fakeMIDI.shape[3]])

            // Concatenate real and fake data
            console.log('mixedMIDI', realMIDITensor, fakeMIDI)
            const mixedMIDI = tf.concat([realMIDITensor, fakeMIDI], 0);

            console.log('mixedLabels')
            const mixedLabels = tf.concat([tf.ones([batchSize / 2, 1]), tf.zeros([batchSize / 2, 1])], 0);

            // Train discriminator on both real and fake data
            setTrainableStatus(discriminatorModel, true);
            const dLoss = await discriminatorModel.trainOnBatch(mixedMIDI, mixedLabels);

            // Generate new noise for generator training
            const zNew = tf.randomNormal([batchSize, zDim]);
            const misleadingLabels = tf.ones([batchSize, 1]);

            // Train generator (via discriminator's error)
            setTrainableStatus(discriminatorModel, false);
            const gLoss = await generatorModel.trainOnBatch(zNew, misleadingLabels);

            // Update UI
            statusMessage.value = `Epoch ${epoch + 1}/${epochs}, Batch ${batch + 1}, D Loss: ${dLoss}, G Loss: ${gLoss}`;
        }
    }

    statusMessage.value = 'Training complete.'
}

function generateMidiMatrix() {
    const generatedData = generatorModel.predict(someInputNoise)
    const scaledData = generatedData.mul(255)
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
</script>

<template>
    <app-section title="Train">

        <div class="flex flex-row justify-between items-center mb-2">
            <h3 class="text-sm">Status</h3>
        </div>
        <div class="text-md rounded-md bg-gray-900 text-green-400 py-2 px-4 mb-2 font-mono whitespace-pre-line">{{
            statusMessage }}
        </div>

        <div>
            <h3 class="text-sm mt-6 mb-2">Actions</h3>
            <el-button @click="createTrainingData" :disabled="loading">Create training data</el-button>
            <el-button @click="initializeModel" :disabled="loading">Initialize
                Model</el-button>
            <el-button :disabled="loading" @click="trainModel">Train Model</el-button>
            <el-button :disabled="loading">Save Model</el-button>
        </div>

        <div>
            <h3 class="text-sm mt-6 mb-2">Training Data to Process</h3>
            <el-select v-model="selectedTrainingDataUrl" placeholder="Select Training Data ZIP" size="large">
                <el-option v-for="item in trainingZips" :key="item.value" :label="item.label" :value="item.value" />
            </el-select>
        </div>

        <div>
            <!-- Hidden file input -->
            <h3 class="text-sm mt-6 mb-2">Load Processed Training Samples (.zip)</h3>
            <p v-if="selectedSamplesName" class="text-xs mb-4 text-gray-500 font-bold">Selected samples:<br><span
                    class="italic font-regular">{{
            selectedSamplesName }}</span></p>
            <el-button @click="fileInput.click()" :disabled="loading" size="large">
                <icon class="mr-2" name="material-symbols:attach-file" size="1.5em" />
                Choose File
            </el-button>
            <input class="hidden" type="file" ref="fileInput" @change="handleFileImport" accept=".zip" />
        </div>

    </app-section>
</template>
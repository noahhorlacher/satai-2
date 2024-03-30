<script setup>
import JSZip from 'jszip'
import pako from 'pako'

const loading = ref(false)

const cancelAction = ref(false)

const { trainingData } = toRefs(useTrainingDataStore())
const { statusMessage } = toRefs(useStatusMessageStore())

statusMessage.value = 'Press a button to begin...'

const dataPreprocessorBatchSize = 1000

// load midi data
// my midi files were already pre-processed to only have 1 track (track 0)
let midiFiles = []
const dataLoaded = ref(false)
async function loadData() {
    loading.value = true
    cancelAction.value = false

    midiFiles = []

    try {
        statusMessage.value = 'Downloading midi files ZIP...'
        const response = await fetch('/data/midi/midi_all.zip')
        const zipData = await response.blob()
        const jszip = new JSZip()

        // load zip content
        const zip = await jszip.loadAsync(zipData)

        let idx = 1
        let filesAmount = Object.keys(zip.files).length

        // Unzip each file
        for (const [filename, file] of Object.entries(zip.files)) {
            if (cancelAction.value) {
                break
            }

            statusMessage.value = `Unzipping midi file ${idx++}/${filesAmount}\n(${filename})`

            if (filename.endsWith('.midi') || filename.endsWith('.mid')) {
                const midiFile = await file.async("arraybuffer")
                midiFiles.push(midiFile)
            }
        }

        statusMessage.value = cancelAction.value ? `Loading Canceled` : `Loaded ${midiFiles.length} MIDI files.`
        dataLoaded.value = !cancelAction.value
    } catch (err) {
        console.error('Error loading midi files:', err)
        statusMessage.value = 'Error loading midi files. Check console'
    }
    loading.value = false

}

// preprocess midi data for training
const preprocessedData = ref()
const dataPreprocessed = ref(false)
async function preprocessData() {
    loading.value = true

    let batchNumber = 1
    for(let i = 0; i < midiFiles.length; i+=dataPreprocessorBatchSize) {
        const batch = midiFiles.slice(i, i+dataPreprocessorBatchSize)
        const _preprocessedData = await MIDIPreprocessor.preprocess(batch, undefined, `${batchNumber} of ${Math.ceil(midiFiles.length/dataPreprocessorBatchSize)}`)

        exportTrainingData(_preprocessedData, `${batchNumber}-of-${Math.ceil(midiFiles.length/dataPreprocessorBatchSize)}`)

        batchNumber++
    }

    statusMessage.value = 'Preprocessing complete.'

    loading.value = false
}

function exportTrainingData(dataToExport, batchProgress) {
    statusMessage.value = `downloading training data part ${batchProgress}...`

    if(!dataToExport || dataToExport.length === 0) {
        return
    }

    const data = JSON.stringify(dataToExport)
    const compressedData = pako.deflate(data)

    const blob = new Blob([compressedData], { type: 'application/gzip' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `training-data-part-${batchProgress}.json.gz`
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
}
</script>

<template>
    <app-section title="Train">
        <div class="flex flex-row justify-between items-center mb-2">
            <h3 class="text-sm">Status</h3>
            <el-button link bg size="small" @click="cancelAction = true" :disabled="!loading">Cancel Action</el-button>
        </div>
            <div class="text-md rounded-md bg-gray-200 py-2 px-4 mb-2 font-mono whitespace-pre-line">{{ statusMessage }}</div>

        <h3 class="text-sm mt-6 mb-2">Actions</h3>
        <el-button @click="loadData" :disabled="loading">Load Data</el-button>
        <el-button @click="preprocessData" :disabled="loading">Process Data</el-button>
        <el-button @click="initializeModel" :disabled="loading || !dataLoaded || !dataPreprocessed">Initialize Model</el-button>
        <el-button :disabled="loading">Train Model</el-button>
        <el-button :disabled="loading">Save Model</el-button>
    </app-section>
</template>
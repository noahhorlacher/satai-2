<script setup>
import JSZip from 'jszip'

const loading = ref(false)

const status = ref('Press a button to begin...')
const cancelAction = ref(false)

// load midi data
// my midi files were already pre-processed to only have 1 track (track 0)
const midiFiles = ref([])
const dataLoaded = ref(false)
async function loadData() {
    loading.value = true
    cancelAction.value = false

    midiFiles.value = []

    try {
        status.value = 'Downloading midi files ZIP...'
        const response = await fetch('/data/midi/midi.zip')
        const zipData = await response.blob()
        const jszip = new JSZip()

        // load zip content
        const zip = await jszip.loadAsync(zipData)

        let idx = 1
        let filesAmount = Object.keys(zip.files).length

        // Process each file
        for (const [filename, file] of Object.entries(zip.files)) {
            if (cancelAction.value) {
                break
            }

            status.value = `Unzipping midi file ${idx++}/${filesAmount}\n(${filename})`

            if (filename.endsWith('.midi') || filename.endsWith('.mid')) {
                const midiFile = await file.async("arraybuffer")
                midiFiles.value.push(midiFile)
            }
        }

        preprocessedData.value = midiFiles
        status.value = cancelAction.value ? `Loading Canceled` : `Loaded ${midiFiles.value.length} MIDI files.`
        dataLoaded.value = !cancelAction.value
    } catch (err) {
        console.error('Error loading midi files:', err)
        status.value = 'Error loading midi files. Check console'
    }
    loading.value = false

}

// preprocess midi data for training
const preprocessedData = ref()
const dataPreprocessed = ref(false)
async function preprocessData() {
    loading.value = true
    preprocessedData.value = MIDIPreprocessor.preprocess([...midiFiles.value])
    loading.value = false
    dataPreprocessed.value = true
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
            <el-button size="small" @click="cancelAction = true" :disabled="!loading">Cancel Action</el-button>
        </div>
            <div class="text-md bg-gray-200 py-2 px-4 mb-2 font-mono whitespace-pre-line">{{ status }}</div>

        <h3 class="text-sm mt-6 mb-2">Actions</h3>
        <el-button @click="loadData" :disabled="loading">Load Data</el-button>
        <el-button @click="preprocessData" :disabled="loading || !dataLoaded">Process Data</el-button>
        <el-button @click="initializeModel" :disabled="loading || !dataLoaded || !dataPreprocessed">Initialize Model</el-button>
        <el-button :disabled="loading">Train Model</el-button>
        <el-button :disabled="loading">Save Model</el-button>
    </app-section>
</template>
<script setup>
import JSZip from 'jszip'
import pako from 'pako'

const loading = ref(false)

const { statusMessage } = toRefs(useStatusMessageStore())

const selectedUnprocessed = ref()

const fileInput = ref()

statusMessage.value = 'Press an action button to begin...'

const dataPreprocessorBatchSize = 400

let loadedUnprocessedName
let loadedUnprocessedData

// load midi data
// my midi files were already pre-processed to only have 1 track (track 0)
let midiFiles = []
async function loadData() {
    loading.value = true

    midiFiles = []

    try {
        statusMessage.value = `Loading ${loadedUnprocessedName}...`

        const zipData = loadedUnprocessedData
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

    exportTrainingData(zipData, `SatAi-Training-Data_${loadedUnprocessedName}_batch-size-${dataPreprocessorBatchSize}_${amountBatches}-batches`)

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

function handleFileImport(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.readAsArrayBuffer(file);

    reader.onload = async (e) => {
        try {
            statusMessage.value = 'Importing data to process...';
            loadedUnprocessedName = file.name;
            loadedUnprocessedData = e.target.result;

            statusMessage.value = 'Training data imported successfully.';
        } catch (error) {
            statusMessage.value = `Error importing training data: ${error.message}`;
        }
    };

    reader.onerror = (error) => {
        statusMessage.value = `FileReader error: ${error.message}`;
    };
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
            <el-button :disabled="loading">Save Model</el-button>
        </div>

        <div>
            <h3 class="text-sm mt-6 mb-2">Training Data to Process</h3>
            <p v-if="selectedUnprocessed" class="text-xs mb-4 text-gray-500 font-bold">Selected MIDI files:<br><span
                    class="italic font-regular">{{
            selectedUnprocessed }}</span></p>
            <el-button @click="fileInput.click()" size="large">
                <icon class="mr-2" name="material-symbols:attach-file" size="1.5em" />
                Choose File
            </el-button>
            <input class="hidden" type="file" ref="fileInput" @change="handleFileImport" accept=".zip" />
        </div>

        <section-actual-train />
    </app-section>
</template>
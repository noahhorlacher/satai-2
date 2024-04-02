<script setup>
import JSZip from 'jszip'
import pako from 'pako'

const busy = ref(false)
const { statusMessage } = toRefs(useStatusMessageStore())

const fileInput = ref()

statusMessage.value = 'Press an action button to begin...'

const dataPreprocessorBatchSize = 100 // how many midi files (not samples) per batch

const loadedUnprocessedName = ref()
const loadedUnprocessedData = ref()

// load midi data
let midiFiles = []

// preprocess midi data for training
async function preprocessData() {
    busy.value = true

    let batchSizes = dataPreprocessorBatchSize > 0 ? dataPreprocessorBatchSize : midiFiles.length
    const amountBatches = Math.ceil(midiFiles.length / batchSizes)

    const zipFile = new JSZip()

    let amountSamples = 0

    for (let i = 0; i < midiFiles.length; i += batchSizes) {
        const batch = midiFiles.slice(i, i + batchSizes)

        let currentBatch = i / batchSizes + 1
        const preprocessedBatch = await MIDIPreprocessor.preprocess(batch, undefined, `${currentBatch} of ${amountBatches}`)

        amountSamples += preprocessedBatch.length
        const data = JSON.stringify(preprocessedBatch)
        const gzipData = pako.gzip(data)

        // add the gzip compressed batch to the zip file
        zipFile.file(`batch-${currentBatch}-of-${amountBatches}_size-${batch.length}.json.gz`, gzipData)
    }

    // Compress the entire array of preprocessed data
    statusMessage.value = `Compressing all data`
    const zipData = await zipFile.generateAsync({ type: 'uint8array' })

    let filenameWithoutExtension = loadedUnprocessedName.value.split('.').slice(0, -1).join('.')

    await downloadData(
        zipData,
        `SatAi_2_Training_Samples_${filenameWithoutExtension}_batch-size-${dataPreprocessorBatchSize}_samples-${amountSamples}.zip`,
        'application/zip'
    )

    statusMessage.value = `Preprocessing complete. Final amount of samples: ${amountSamples}`
    busy.value = false
}

// handle file import
async function newMIDIZipChosen(event) {
    busy.value = true

    const { fileName, trainingSamples } = await handleFileImport(event, true, 'unprocessed MIDI ZIP file')

    loadedUnprocessedName.value = fileName
    midiFiles = trainingSamples

    loadedUnprocessedData.value = true
    busy.value = false
}
</script>

<template>
    <app-section title="Preprocess">
        <div class="flex flex-row justify-between items-center mb-2">
            <h3 class="text-sm">Status</h3>
        </div>
        <div
            class="text-md rounded-md bg-gray-900 text-green-400 py-2 px-4 mb-2 font-mono whitespace-pre-line shadow-md">
            {{ statusMessage }}
        </div>

        <div>
            <h3 class="text-sm mt-6 mb-2">Training Data to Process</h3>
            <p v-if="loadedUnprocessedName" class="text-xs mb-4 text-gray-500 font-bold">Selected MIDI files:<br><span
                    class="italic font-regular">{{
                loadedUnprocessedName }}</span></p>
            <el-button @click="fileInput.click()" :disabled="busy" size="large">
                <icon class="mr-2" name="material-symbols:attach-file" size="1.5em" />
                Choose File
            </el-button>
            <input class="hidden" type="file" ref="fileInput" @change="newMIDIZipChosen" accept=".zip" />
        </div>

        <div>
            <h3 class="text-sm mt-6 mb-2">Actions</h3>
            <el-button @click="preprocessData" :disabled="!loadedUnprocessedData || busy">Preprocess training
                data</el-button>
        </div>
    </app-section>
</template>
<script setup>
import * as tf from '@tensorflow/tfjs'
import pako from 'pako'
import { Midi } from '@tonejs/midi'

const { statusMessage } = toRefs(useStatusMessageStore())

statusMessage.value = 'Press an action button to begin...'

const busy = ref(false)

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
        labels: {
            show: false
        }
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

const trainedForEpochs = ref(0)

let trainingData = []

let epochs = 1000
let batchSize = 32

const epochsSelection = ref(epochs)
const batchSizeSelection = ref(batchSize)

// for generating
// Threshold for value to be considered a note (below is 0)
const midiConfidenceThreshold = ref(0.3)

// Threshold difference in velocity to last x unit (same pitch) to interrupt last and start a new note
const velocityDifferenceThreshold = ref(0.1)

// Assuming dimensions and startOctave are available from your preprocessing settings
const startOctave = 3;

const trainingDimensions = {
    x: 192,
    y: 64
}

const discriminatorLearningRate = 0.001
const ganLearningRate = 0.0015
const clipValue = 0.01

const generatorParamsAmount = 100

/*
    For stability issues:
    Try Wasserstein loss or mean squared error loss

    Architecture based on:
    https://medium.com/ee-460j-final-project/generating-music-with-a-generative-adversarial-network-8d3f68a33096
*/
const discriminator = createDiscriminatorModel(trainingDimensions, discriminatorLearningRate, clipValue)
const generator = createGeneratorModel(trainingDimensions, generatorParamsAmount)
const gan = createGANModel(generator, discriminator, ganLearningRate, clipValue)

function shouldSaveEpoch() {
    const i = trainedForEpochs.value

    if (i < 100 && i % 10 === 0) {
        return true
    } else if (i < 500 && i % 20 === 0) {
        return true
    } else if (i < 1000 && i % 50 === 0) {
        return true
    } else if (i < 10000 && i % 100 === 0) {
        return true
    } else if (i < 100000 && i % 1000 === 0) {
        return true
    } else if (i % 10000 === 0) {
        return true
    }

    return false
}

let trainingStartDateTime
let trainingPreviewNoise = tf.randomNormal([3, generatorParamsAmount])
async function trainModel() {
    busy.value = true
    statusMessage.value = 'Starting training...'

    epochs = epochsSelection.value
    batchSize = batchSizeSelection.value

    let backend = tf.getBackend()
    trainingStartDateTime = new Date()

    let realImagesArray

    for (let i = 0; i < epochs; i++) {
        try {
            realImagesArray = await getRandomSamples(batchSize)

            // Convert each 2D image in the array to a 3D image by adding an extra dimension
            realImagesArray = realImagesArray.map(image => {
                return image.map(row => {
                    return row.map(value => {
                        return [value] // Adds an extra dimension
                    })
                })
            })

            // Convert the array to a tensor
            let realImages = tf.tensor4d(realImagesArray, [batchSize, trainingDimensions.y, trainingDimensions.x, 1]);

            // Check if the conversion is correct
            // Generate a batch of fake images.
            const noise = tf.randomNormal([batchSize, generatorParamsAmount])
            const fakeImages = generator.predict(noise)

            // Create a batch of labels for the real and fake images.
            // With label smoothing
            const realLabels = tf.ones([batchSize, 1]).mul(0.9)
            const fakeLabels = tf.zeros([batchSize, 1]).mul(0.1)

            // Train the discriminator on real and fake images
            let dLossReal = await discriminator.trainOnBatch(realImages, realLabels)
            let dLossFake = await discriminator.trainOnBatch(fakeImages, fakeLabels)

            realImages.dispose()
            fakeImages.dispose()
            realLabels.dispose()
            fakeLabels.dispose()

            const dLoss = (dLossReal + dLossFake) / 2

            // Train the generator via the gan model
            const misleadingLabels = tf.ones([batchSize, 1]).mul(0.9)
            const gLoss = await gan.trainOnBatch(noise, misleadingLabels)

            noise.dispose()
            misleadingLabels.dispose()

            statusMessage.value = `💡 Using backend [${backend}] to train\n⏲ Started training on ${trainingStartDateTime.toLocaleString()}\n🥊 Trained epoch ${i + 1} of ${epochs}.\n🎨 GAN loss: ${gLoss}\n👓 Discriminator loss: ${dLoss}`

            chartSeries[0].data.push(gLoss.toFixed(5))
            chartSeries[1].data.push(dLoss.toFixed(5))

            if (shouldSaveEpoch()) {
                previewImage(true)
                generateMIDI(true)
            }

            trainedForEpochs.value++
        } catch (error) {
            console.error('Error training:', error)
            console.log('the samples in question', realImagesArray)
            statusMessage.value = 'Error training. Check console'
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

function generateMIDI(training = false) {
    busy.value = true;

    let noise = training ? trainingPreviewNoise : tf.randomNormal([1, generatorParamsAmount]);
    const generatedData = generator.predict(noise);
    let data = generatedData.arraySync();

    // Flatten the data and remove the unnecessary dimension
    data = data.map(sample => sample.map(row => row.map(value => value[0])));

    // Convert the data to a MIDI file
    const midi = new Midi();

    // Add a track
    const track = midi.addTrack();

    // Create notes from the generated data
    for (let sample of data) {
        for (let y = 0; y < sample.length; y++) {
            let noteStartX = null;
            let lastVelocity = 0;

            for (let x = 0; x < sample[y].length; x++) {
                const velocity = Math.max(0, Math.min(1, sample[y][x]));

                if (velocity > midiConfidenceThreshold) {
                    // Check if the difference in velocity is large enough to start a new note
                    if (noteStartX === null || Math.abs(velocity - lastVelocity) > velocityDifferenceThreshold) {
                        noteStartX = x; // Note onset
                        lastVelocity = velocity;
                    }

                    // If this is the last pixel in the row or next pixel is below the threshold, end the note
                    if (x === sample[y].length - 1 || (Math.abs(sample[y][x + 1] - velocity) > velocityDifferenceThreshold)) {
                        const midiNumber = y + (startOctave * 12);
                        const startTime = 2 * 8 * (noteStartX / trainingDimensions.x);
                        const endTime = 2 * 8 * ((x + 1) / trainingDimensions.x);
                        const duration = endTime - startTime;

                        track.addNote({
                            midi: midiNumber,
                            time: startTime,
                            duration: duration,
                            velocity: velocity // Velocity already normalized
                        });

                        noteStartX = null; // Reset for next note
                    }
                } else {
                    noteStartX = null; // Reset if current pixel is below the threshold
                    lastVelocity = 0;
                }
            }
        }
    }

    // Convert the MIDI to a blob and download
    const blob = new Blob([midi.toArray()], { type: 'audio/midi' });
    downloadData(blob, `SatAi2-sample_epoch-${trainedForEpochs.value}.mid`, 'audio/midi')

    busy.value = false;
}

const canvasPreview = ref()
const previewImages = ref([])
function previewImage(training = false) {
    busy.value = true

    const noise = training ? trainingPreviewNoise : tf.randomNormal([1, generatorParamsAmount])

    let generatedData = generator.predict(noise)

    let data = generatedData.arraySync()


    data = data.map(sample => {
        return sample.map(row => {
            return row.map(value => {
                return value[0] // remove unnecessary dimension
            })
        })
    })

    const ctx = canvasPreview.value.getContext('2d')
    ctx.imageSmoothingEnabled = false

    ctx.clearRect(0, 0, trainingDimensions.x, trainingDimensions.y)

    const imageData = ctx.createImageData(trainingDimensions.x, trainingDimensions.y)
    console.log('data', data)


    // loop over each pixel and set pixel brightness to the generated data
    for (let sample of data) {
        for (let y = 0; y < trainingDimensions.y; y++) {
            for (let x = 0; x < trainingDimensions.x; x++) {
                let value = sample[y][x]
                const pixelIndex = (y * trainingDimensions.x + x) * 4

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
            previewImages.value.unshift({
                description: `Epoch ${trainedForEpochs.value}`,
                src: url
            })
        })
    }



    busy.value = false
}

async function previewSample() {
    if (!trainingData || trainingData.length == 0) {
        console.error('No training data available')
        statusMessage.value = 'No training data available.'
        return
    }

    busy.value = true

    currentLoadedSampleBatch.batchName = trainingData[Math.floor(Math.random() * trainingData.length)]
    currentLoadedSampleBatch.timesLoaded = 0

    let fileData = trainingData[Math.floor(Math.random() * trainingData.length)]

    let unzippedData = await pako.ungzip(fileData)
    fileData = null

    let arrays = JSON.parse(new TextDecoder().decode(unzippedData))
    unzippedData = null

    // get random element from trainingData
    const randomIndex = Math.floor(Math.random() * arrays.length)
    const randomSample = arrays[randomIndex]

    const ctx = canvasPreview.value.getContext('2d')
    ctx.imageSmoothingEnabled = false

    ctx.clearRect(0, 0, trainingDimensions.x, trainingDimensions.y)

    const imageData = ctx.createImageData(trainingDimensions.x, trainingDimensions.y)

    // loop over each pixel and set pixel brightness to the generated data$
    for (let y = 0; y < randomSample.length; y++) {
        for (let x = 0; x < randomSample[y].length; x++) {
            let value = randomSample[y][x]
            const pixelIndex = (y * trainingDimensions.x + x) * 4

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

async function newSampleFileChosen(event) {
    busy.value = true
    const { fileName, trainingSamples } = await handleFileImport(event, 'training samples')

    selectedSamplesName.value = fileName
    trainingData = trainingSamples

    busy.value = false
}

function nextSave() {
    const i = trainedForEpochs.value

    if (i < 100) {
        return i - (i % 10) + 10
    } else if (i < 500 && i % 20 === 0) {
        return i - (i % 20) + 20
    } else if (i < 1000 && i % 50 === 0) {
        return i - (i % 50) + 50
    } else if (i < 10000 && i % 100 === 0) {
        return i - (i % 100) + 100
    } else if (i < 100000 && i % 1000 === 0) {
        return i - (i % 1000) + 1000
    } else {
        return i - (i % 10000) + 10000
    }
}

// download both models
async function saveModel() {
    const discriminatorModelSaveResult = await discriminator.save(`downloads://discriminator-model-${trainedForEpochs.value}-epochs`)
    const generatorModelSaveResult = await generator.save(`downloads://generator-model-${trainedForEpochs.value}-epochs`)
    const ganModelSaveResult = await gan.save(`downloads://gan-model-${trainedForEpochs.value}-epochs`)
}

async function loadModel() {
    // implement loading from localstorage or get request
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
                    Latest Preview ({{ previewImages[0].description }})
                    Next save at epoch {{ nextSave() }}
                </h3>
                <nuxt-img :src="previewImages[0].src" class="w-full md:w-1/2 max-w-[350px] h-auto"
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
                <input class="hidden" type="file" ref="fileInput" @change="newSampleFileChosen" accept=".zip" />
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
            <h3 class="text-sm mt-6 mb-2">MIDI Generator Settings</h3>

            <div>
                <p class="text-xs mb-1">MIDI Confidence Threshold</p>
                <el-input-number :step="0.05" v-model="midiConfidenceThreshold" />
            </div>
            <div>
                <p class="text-xs mb-1">Velocity Difference Threshold</p>
                <el-input-number :step="0.05" v-model="velocityDifferenceThreshold" />
            </div>
        </div>

        <div>
            <h3 class="text-sm mt-6 mb-2">Test Model</h3>
            <el-button @click="generateMIDI" :disabled="trainingData.length <= 0 || busy">
                Generate MIDI
            </el-button>
            <el-button @click="previewImage" :disabled="trainingData.length <= 0 || busy">
                Preview Image
            </el-button>
            <el-button @click="previewSample" :disabled="trainingData.length <= 0 || busy">
                Preview Sample
            </el-button>
            <el-button @click="saveModel" :disabled="trainedForEpochs <= 0 || busy">
                Save Model
            </el-button>
        </div>

        <div>
            <h3 class="text-sm mt-6 mb-2">Previews</h3>

            <div v-if="previewImages.length == 0" class="text-xs text-gray-500 mb-4">
                No previews available. Train or click "Preview Image" to generate previews.
            </div>

            <div class="flex flex-row flex-wrap mt-8 gap-4 justify-center">
                <figure v-for="(previewImage, index) of previewImages" class="grow shrink w-1/3 h-auto">
                    <figcaption class="text-xs mb-2">{{ previewImage.description }}</figcaption>
                    <nuxt-img class="w-full" style="image-rendering: pixelated" :src="previewImage.src"
                        :key="`previewImage-${index}`" />
                </figure>
            </div>

            <canvas class="hidden" :width="trainingDimensions.x" :height="trainingDimensions.y" ref="canvasPreview" />
        </div>
    </app-section>
</template>
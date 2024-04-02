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

const discriminatorLearningRate = 0.001
const ganLearningRate = 0.001
const clipValue = 0.01

let epochs = 1000
let batchSize = 32

// for generating
const midiConfidenceThreshold = 0.3

// Assuming dimensions and startOctave are available from your preprocessing settings
const startOctave = 3;


const trainedForEpochs = ref(0)

const discriminator = tf.sequential()

const epochsSelection = ref(epochs)
const batchSizeSelection = ref(batchSize)

const trainingDimensions = {
    x: 192,
    y: 64
}
const generatorParamsAmount = 100

// First Convolutional layer - Assuming a larger kernel and stride to reduce dimension significantly
discriminator.add(tf.layers.conv2d({
    inputShape: [trainingDimensions.y, trainingDimensions.x, 1],
    filters: trainingDimensions.y * 2,
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
    filters: trainingDimensions.y,
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
    filters: trainingDimensions.y / 2,
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

// Start with a dense layer taking the input noise vector (size generatorParamsAmount)
generator.add(tf.layers.dense({
    inputShape: [generatorParamsAmount],
    units: 512, // Increased units for more capacity
    useBias: false
}));
generator.add(tf.layers.batchNormalization());
generator.add(tf.layers.leakyReLU({ alpha: 0.2 }));

// Adding another dense layer to expand
generator.add(tf.layers.dense({
    units: 1 * (trainingDimensions.y / 8) * (trainingDimensions.x / 8), // Increased units for subsequent reshape
    useBias: false
}));
generator.add(tf.layers.batchNormalization());
generator.add(tf.layers.leakyReLU({ alpha: 0.2 }));

// Reshape to a 3D volume
generator.add(tf.layers.reshape({
    targetShape: [trainingDimensions.y / 8, trainingDimensions.x / 8, 1] // Increased depth for richer feature representation
}));

// Use Conv2DTranspose to upscale the image dimensions while reducing depth
// First Conv2DTranspose layer
// stride is the factor by which the image is upsampled
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
let trainingPreviewNoise = tf.randomNormal([3, generatorParamsAmount])

async function trainModel() {
    statusMessage.value = 'Starting training...'

    epochs = epochsSelection.value
    batchSize = batchSizeSelection.value

    busy.value = true
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
            realImages = tf.tensor4d(realImagesArray, [batchSize, trainingDimensions.y, trainingDimensions.x, 1]);

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

            chartSeries[0].data.push(gLoss.toFixed(5))
            chartSeries[1].data.push(dLoss.toFixed(5))

            statusMessage.value = `💡 Using backend [${backend}] to train\n⏲ Started training on ${trainingStartDateTime.toLocaleString()}\n🥊 Trained epoch ${i + 1} of ${epochs}.\n🎨 GAN loss: ${gLoss}\n👓 Discriminator loss: ${dLoss}`
        } catch (error) {
            console.error('Error training:', error)
            console.log('the samples in question', realImagesArray)
            statusMessage.value = 'Error training. Check console'
        }

        trainedForEpochs.value++

        if (trainedForEpochs.value % 10 == 0) {
            previewCanvas(true)
            generateMIDI(true)
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

function generateMIDI(training = false) {
    if (!trainingData || trainingData.length == 0) {
        console.error('No training data available');
        statusMessage.value = 'No training data available.';
        return;
    }

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

    // Assuming midiConfidenceThreshold, startOctave, trainingDimensions are defined
    for (let sample of data) {
        for (let y = 0; y < sample.length; y++) {
            for (let x = 0; x < sample[y].length; x++) {
                const velocity = Math.round(Math.max(0, Math.min(127, sample[y][x] * 127)));

                if (velocity > midiConfidenceThreshold) {
                    const midiNumber = y + (startOctave * 12); // Calculate MIDI note number
                    const time = 2 * 8 * (x / trainingDimensions.x); // Time in beats
                    const duration = 2 * 7 * (1 / trainingDimensions.x); // Duration in beats

                    track.addNote({
                        midi: midiNumber,
                        time: time,
                        duration: duration,
                        velocity: velocity / 127 // Normalize velocity to 0-1 range
                    });
                }
            }
        }
    }

    // Convert the MIDI to a blob and download
    const blob = new Blob([midi.toArray()], { type: 'audio/midi' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'generated.mid';
    a.click();
    a.remove();

    busy.value = false;
}


const canvasPreview = ref()
const previewImages = ref([])
function previewCanvas(training = false) {
    if (!trainingData || trainingData.length == 0) {
        console.error('No training data available')
        statusMessage.value = 'No training data available.'
        return
    }


    busy.value = true

    const noise = training ? trainingPreviewNoise : tf.randomNormal([1, generatorParamsAmount])

    let generatedData = generator.predict(noise)

    console.log('generatedData', generatedData)

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

function nextSave() {
    const nextSaveEpoch = trainedForEpochs.value - (trainedForEpochs.value % 10) + 10
    return nextSaveEpoch || 'None'
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
                <nuxt-img :src="previewImages[0].src" class="w-1/2 max-w-[350px] h-auto"
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
            <el-button @click="generateMIDI" :disabled="busy">
                Generate MIDI
            </el-button>
            <el-button @click="previewCanvas" :disabled="busy">
                Preview Image
            </el-button>
            <el-button @click="previewSample" :disabled="trainingData.length <= 0 || busy">
                Preview Sample
            </el-button>
            <el-button @click="saveModel" :disabled="busy">
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
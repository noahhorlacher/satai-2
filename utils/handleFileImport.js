import JSZip from 'jszip'

export async function handleFileImport(event, importName = 'training data') {
    const { statusMessage } = toRefs(useStatusMessageStore())
    const file = event.target.files[0]

    if (!file) return Promise.resolve({ fileName: '', trainingSamples: [] })

    return new Promise((resolve, reject) => {
        const reader = new FileReader()

        reader.onload = async (e) => {
            try {
                statusMessage.value = `Importing ${importName} (${file.name})...`
                const trainingSamples = []

                // Decompress the zip
                let zip = new JSZip()
                let zipData = await zip.loadAsync(e.target.result)
                zip = null

                // Load gzipped batches into trainingSamples array
                for (const filename of Object.keys(zipData.files)) {
                    statusMessage.value = `Unzipping ${filename}...`
                    const fileData = await zipData.files[filename].async('uint8array')
                    trainingSamples.push(fileData)
                }

                zipData = null
                statusMessage.value = 'Training data imported successfully.'

                resolve({ fileName: file.name, trainingSamples })
            } catch (error) {
                statusMessage.value = `Error importing training data: ${error.message}`
                reject(error)
            }
        };

        reader.onerror = (error) => {
            statusMessage.value = `Error reading file: ${error.message}`
            reject(error)
        }

        reader.readAsArrayBuffer(file)
    })
}
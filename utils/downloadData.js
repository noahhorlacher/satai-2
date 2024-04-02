export async function downloadData(data, fileName, dataType) {
    const { statusMessage } = toRefs(useStatusMessageStore())
    statusMessage.value = `Downloading ${fileName}...`

    const blob = new Blob([data], { type: dataType });
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${fileName}`
    a.click()
    URL.revokeObjectURL(url)
    a.remove()
}
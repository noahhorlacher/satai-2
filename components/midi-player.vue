<script setup>
import PicoAudio from 'picoaudio'

const props = defineProps({
    src: String,
    description: String,
});

const picoAudio = ref(null)

onMounted(() => {
    picoAudio.value = new PicoAudio()
    picoAudio.value.init()
    picoAudio.value.addEventListener('songEnd', () => {
        stop()
    })

    // load file
    const parsedData = picoAudio.value.parseSMF(base64ToArrayBuffer(props.src))
    picoAudio.value.setData(parsedData)
    picoAudio.value.setMasterVolume(0.7)
    picoAudio.value.setReverb(true)
    picoAudio.value.setReverbVolume(0.5)
    picoAudio.value.setChorus(true)
    picoAudio.value.setChorusVolume(0.5)
})

const isPlaying = ref(false);
const isPaused = ref(false);

function play() {
    isPlaying.value = true
    picoAudio.value.play()
}

function pause() {
    picoAudio.value.pause()
    isPaused.value = true;
    isPlaying.value = false;
}


function stop() {
    picoAudio.value.initStatus()
    isPaused.value = false;
    isPlaying.value = false;
}

function resume() {
    picoAudio.value.play()

    isPaused.value = false;
    isPlaying.value = true;
}

function togglePlay() {
    if (isPaused.value) {
        resume();
        return;
    } else if (isPlaying.value) {
        pause();
    } else {
        play();
    }
}

function base64ToArrayBuffer(base64) {
    var base64String = base64.split(',')[1];
    var binaryString = atob(base64String);
    var bytes = new Uint8Array(binaryString.length);
    for (var i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}

function download() {
    const objectURL = URL.createObjectURL(new Blob([base64ToArrayBuffer(props.src)], { type: 'audio/midi' }));

    const link = document.createElement('a');
    link.href = objectURL;
    link.download = props.description.replaceAll(' ', '_').toLowerCase() + '.mid'
    link.click();
}
</script>

<template>
    <p class="text-xs mb-2">{{ description }}</p>
    <div class="flex flex-row text-gray-900 w-fit shadow-md rounded-md overflow-hidden">
        <div class="bg-gray-50 cursor-pointer hover:bg-gray-400 flex items-center gap-x-2" @click="togglePlay">
            <Icon :name="isPlaying ? 'mdi:pause' : 'mdi:play'" class="mx-auto h-8 w-8 p-2" />
        </div>
        <div class="bg-gray-50 cursor-pointer hover:bg-gray-400 flex items-center gap-x-2" @click="stop">
            <Icon name="mdi:stop" class="mx-auto h-8 w-8 p-2" />
        </div>
        <div @click="download" class="bg-gray-50 cursor-pointer hover:bg-gray-400 flex items-center gap-x-2">
            <Icon name="mdi:download" class="mx-auto h-8 w-8 p-2" />
        </div>
    </div>
</template>
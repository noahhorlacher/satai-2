<script setup>
let links = [
    { label: 'Home', url: '/' },
    { label: 'Preprocess', url: '/preprocess' },
    { label: 'Train', url: '/train' },
    { label: 'Generate', url: '/generate' },
]
</script>

<template>
    <app-section title="SatAi 2.0">
        <p class="mb-6">
            SatAi 2.0 is the second iteration of a MIDI generator Neural Network, specifically a Deep Convolutional
            Generative Adversarial Network (DCGAN).
        </p>

        <h3 class="mb-2">Navigation</h3>
        <div class="flex flex-col">
            <nuxt-link v-for="link of links" :key="link.url" :to="link.url">
                <el-button link>{{ link.label }}</el-button>
            </nuxt-link>
        </div>
    </app-section>
</template>
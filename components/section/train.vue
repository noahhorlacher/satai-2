<script setup>
const { statusMessage } = toRefs(useStatusMessageStore())

statusMessage.value = 'Press an action button to begin...'
</script>

<template>
    <app-section title="Train">

        <div class="flex flex-row justify-between items-center mb-2">
            <h3 class="text-sm">Status</h3>
        </div>
        <div class="text-md rounded-md bg-gray-900 text-green-400 py-2 px-4 mb-2 font-mono whitespace-pre-line">{{
            statusMessage }}
        </div>

        <section-actual-train />
    </app-section>
</template>
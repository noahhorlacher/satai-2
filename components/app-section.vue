<script setup>
const props = defineProps({
    title: String,

})
</script>

<template>
    <section class="mx-auto w-full">
        <h2 v-if="title" v-once class="text-2xl mb-6">{{ title }}</h2>
        <slot />
    </section>
</template>
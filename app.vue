<script setup>
useHead({
  title: 'SatAi',
  bodyAttrs: {
    class: 'bg-gray-200'
  },
  meta: [
    { name: 'description', content: 'A midi generation tool based on SatAi GAN' },
    { name: 'viewport', content: 'width=device-width, initial-scale=1' }
  ]
})
</script>

<template>
  <NuxtLayout>
    <NuxtPage />
  </NuxtLayout>
</template>

<style>
html {
  font-family: 'Poppins', sans-serif;
}
</style>
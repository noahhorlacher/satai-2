// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  devtools: { enabled: true },
  ssr: false,
  modules: [
    "@pinia/nuxt",
    "@nuxtjs/tailwindcss",
    "@element-plus/nuxt",
    "@nuxtjs/google-fonts",
    "@nuxtjs/fontaine",
    "@pinia-plugin-persistedstate/nuxt",
    "nuxt-icon"
  ],
  googleFonts: {
    families: {
      Poppins: true
    },
    download: true,
    prefetch: true,
    overwriting: true
  }
})

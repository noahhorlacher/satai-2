import { defineStore } from 'pinia'

export const useStatusMessageStore = defineStore({
    id: 'statusMessageStore',
    state: () => ({
        statusMessage: '',
    }),
})
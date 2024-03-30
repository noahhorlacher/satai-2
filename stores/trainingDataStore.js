import { defineStore } from 'pinia'

export const useTrainingDataStore = defineStore({
    id: 'trainingDataStore',
    state: () => ({
        trainingData: [],
    }),
})
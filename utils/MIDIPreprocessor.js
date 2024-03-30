import { Midi } from '@tonejs/midi'

const pitches = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

const chromaticInstruments = []
for (let i = 1; i < 48; i++) chromaticInstruments.push(i)
for (let i = 49; i < 97; i++) chromaticInstruments.push(i)
for (let i = 105; i < 113; i++) chromaticInstruments.push(i)

export default class MIDIPreprocessor {
    // outputs matrices.
    // x-axis is time, y-axis is pitch.
    // velocity 0 is black, velocity 127 is white.
    static async preprocess(midiFiles = [], options = {
        dimensions: 64,
        startOctave: 3,
        horizontalResolution: 1 / 8,
        stepSizeX: 1,
        transpositions: [+7, -7],
    }) {
        const { statusMessage } = toRefs(useStatusMessageStore())

        let allProcessedMidiMatrices = []

        for (let index = 0; index < midiFiles.length; index++) {
            statusMessage.value = `Processing MIDI file ${index + 1}/${midiFiles.length}...`
            await new Promise(resolve => setTimeout(resolve, 0))

            const midiArrayBuffer = midiFiles[index]
            const processedMidiMatrices = await MIDIPreprocessor.processSingleMidi(midiArrayBuffer, options)
            if (processedMidiMatrices) {
                allProcessedMidiMatrices = allProcessedMidiMatrices.concat(processedMidiMatrices)
            }
        }

        // delete all matrices that are just 0
        let finalMidiMatrices = []

        statusMessage.value = `Filtering out empty matrices...`

        for (let [index, matrix] of allProcessedMidiMatrices.entries()) {
            statusMessage.value = `Filtering out empty matrices ${index + 1}/${allProcessedMidiMatrices.length}...`
            await new Promise(resolve => setTimeout(resolve, 0))
            if (matrix.some(row => row.some(val => val !== 0))) {
                finalMidiMatrices.push(matrix)
            }
        }

        return finalMidiMatrices
    }

    static async processSingleMidi(midiArrayBuffer, options) {
        const dimensions = options.dimensions;
        const quantization = options.horizontalResolution;
        const startOctave = options.startOctave;
        const stepSizeX = options.stepSizeX;
        const transpositions = options.transpositions;

        let processedMidiMatrices = [];

        try {
            const midiBlobUri = URL.createObjectURL(new Blob([midiArrayBuffer], { type: 'audio/midi' }));
            const midiData = await Midi.fromUrl(midiBlobUri);

            // filter out non-chromatic instruments
            midiData.tracks = midiData.tracks.filter(track => chromaticInstruments.includes(track.instrument.number))

            // select the track with the most notes
            let maxNotes = 0;
            let maxNotesTrack = null;

            for (let i = 0; i < midiData.tracks.length; i++) {
                if (midiData.tracks[i].notes.length > maxNotes) {
                    maxNotes = midiData.tracks[i].notes.length;
                    maxNotesTrack = midiData.tracks[i];
                }
            }

            midiData.tracks = [maxNotesTrack];

            // quantize midi file
            const quantizedNotes = MIDIPreprocessor.quantizeNotes(midiData, quantization)
            if (!quantizedNotes) return false
            midiData.tracks[0].notes = quantizedNotes

            const PPQ = midiData.header.ppq;

            const timeSignatures = midiData.header.timeSignatures || [[4, 4]];
            const timeSignature = timeSignatures.length > 0 ? timeSignatures[0] : [4, 4]; // assuming first time signature is used throughout
            const barLengthTicks = (timeSignature[0] / timeSignature[1]) * PPQ * 4; // Length of a bar in ticks for 4/4 time

            // for each step on x axis
            for (let currentTick = 0; currentTick < midiData.durationTicks; currentTick += barLengthTicks * stepSizeX) {
                let barEndTick = currentTick + barLengthTicks * stepSizeX;

                const midiSegmentNotes = MIDIPreprocessor.sliceMidi(midiData, currentTick, barEndTick);

                // create image with transposition 0
                const normalMidiMatrix = await MIDIPreprocessor.createMidiMatrix(midiSegmentNotes, dimensions, startOctave, PPQ);
                processedMidiMatrices.push(normalMidiMatrix);

                for (let transposition of transpositions) {
                    const transposedSegmentNotes = MIDIPreprocessor.transposeNotes(midiSegmentNotes, transposition);
                    const midiMatrix = await MIDIPreprocessor.createMidiMatrix(transposedSegmentNotes, dimensions, startOctave, PPQ);
                    processedMidiMatrices.push(midiMatrix);
                }
            }
        } catch (e) {
            console.error(e)
        }

        return processedMidiMatrices;
    }


    static quantizeNotes(midiData, quantization) {
        // Adjust midi note start times and durations to the nearest quantization step
        const PPQ = midiData.header.ppq
        const quantizationStep = PPQ / ((1 / quantization) / 4)
        let newNotes = []

        if (!midiData.tracks[0]) return false

        for (const note of midiData.tracks[0].notes) {
            // convert duration to ticks
            const durationInTicks = note.duration * PPQ

            // quantize
            const quantizedStartTime = Math.floor(note.ticks / quantizationStep) * quantizationStep
            const quantizedEndTime = Math.floor((note.ticks + durationInTicks) / quantizationStep) * quantizationStep

            // calculate new duration and convert back to seconds
            const quantizedDuration = (quantizedEndTime - quantizedStartTime) / PPQ

            newNotes.push({
                ...note,
                time: quantizedStartTime / PPQ,
                ticks: quantizedStartTime,
                duration: quantizedDuration
            })
        }

        return newNotes
    }

    static async createMidiMatrix(midiDataNotes, dimensions, startOctave, ppq) {
        // Create the matrix array
        const midiMatrix = Array.from({ length: dimensions }, () => new Array(dimensions).fill(0))

        // Loop through each note and update image matrix
        midiDataNotes.forEach(note => {
            const x = Math.floor(note.time / ppq * dimensions) // Calculate x coordinate
            const y = (note.midi - (startOctave * 12)) % dimensions // Calculate y coordinate

            if (y >= 0 && y <= dimensions - 1) {
                const velocity = Math.floor((note.velocity || 1.0) * 255) // Convert velocity to grayscale value

                // Handle the case where multiple notes map to the same pixel
                // Here, we're just setting it to the brightness of the note, but you might want to average or take max/min
                midiMatrix[y][x] = velocity
            }
        })

        return midiMatrix
    }

    static transposeNotes(midiDataNotes, semitoneShift) {
        // Transpose notes by the given number of semitones
        let newNotes = []

        for (const note of midiDataNotes) {
            let newMidi = note.midi + semitoneShift
            let newOctave = Math.floor((newMidi - 21 + 9) / 12)
            let newPitch = pitches[(newMidi - 21) % 12]
            let newName = newPitch + newOctave

            newNotes.push({
                ...note,
                midi: newMidi,
                octave: newOctave,
                pitch: newPitch,
                name: newName
            })
        }

        return newNotes
    }

    static sliceMidi(midiData, startTick, endTick) {
        let newNotes = []
        newNotes = midiData.tracks[0].notes.filter(note => note.ticks >= startTick && note.ticks < endTick)
        return newNotes
    }
}
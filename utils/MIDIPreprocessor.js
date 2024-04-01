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
        horizontalResolution: 1 / 16,
        stepSizeX: 1,
        transpositions: [],
        minimumNotes: 6
    }, batchProgress) {
        const { statusMessage } = toRefs(useStatusMessageStore())

        let allProcessedMidiMatrices = []

        for (let index = 0; index < midiFiles.length; index++) {
            statusMessage.value = `Processing batch ${batchProgress}\nFile ${index + 1}/${midiFiles.length}...`

            const midiArrayBuffer = midiFiles[index]

            const processedMidiMatrices = await MIDIPreprocessor.processSingleMidi(midiArrayBuffer, options)
            if (processedMidiMatrices) {
                allProcessedMidiMatrices = allProcessedMidiMatrices.concat(processedMidiMatrices)
            }
        }

        // delete all matrices that are just zeros
        let finalMidiMatrices = []

        for (let [index, matrix] of allProcessedMidiMatrices.entries()) {
            if (matrix.some(row => row.some(val => val !== 0))) {
                finalMidiMatrices.push(matrix)
            }
        }

        return finalMidiMatrices
    }

    static sampleHasCorrectAmountRows(sample) {
        let result = sample.length == 64

        if (!result) {
            // console.log('Sample does not have correct amount of rows')
            // console.log(sample, sample.length)
        }

        return result
    }

    static sampleHasCorrectAmountColumns(sample) {
        let result = sample.every(row => row.length == 64)

        if (!result) {
            // console.log('Sample does not have correct amount of columns.')
            // console.log(sample)
        }

        return result
    }

    static sampleHasMinimumNotes(sample, minimumNotes) {
        let amountNotes = sample.reduce((acc, row) => acc + row.reduce((acc2, note) => acc2 + (note > 0 ? 1 : 0), 0), 0)
        let result = amountNotes > minimumNotes

        if (!result) {
            // shut the hell up
            // console.log(`Sample has ${amountNotes} notes but needs at least ${minimumNotes}.`)
        }

        return result
    }

    static async processSingleMidi(midiArrayBuffer, options) {
        const dimensions = options.dimensions;
        let processedMidiMatrices = [];

        try {
            const midiBlobUri = URL.createObjectURL(new Blob([midiArrayBuffer], { type: 'audio/midi' }));
            const midiData = await Midi.fromUrl(midiBlobUri);

            const PPQ = midiData.header.ppq;
            const timeSignatures = midiData.header.timeSignatures || [[4, 4]];
            const timeSignature = timeSignatures[0] || [4, 4]
            const ticksPerMeasure = (timeSignature[0] / timeSignature[1]) * PPQ * 4;

            for (let track of midiData.tracks) {
                if (chromaticInstruments.includes(track.instrument.number)) {
                    track.notes.forEach(note => {
                        const startMeasure = Math.floor(note.ticks / ticksPerMeasure);
                        const endMeasure = Math.floor((note.ticks + note.durationTicks) / ticksPerMeasure);

                        for (let measure = startMeasure; measure <= endMeasure; measure++) {
                            if (!processedMidiMatrices[measure]) {
                                processedMidiMatrices[measure] = MIDIPreprocessor.createEmptyMatrix(dimensions);
                            }
                            const startTick = note.ticks - (measure * ticksPerMeasure);
                            const endTick = (note.ticks + note.durationTicks) - (measure * ticksPerMeasure);

                            MIDIPreprocessor.fillMatrix(processedMidiMatrices[measure], startTick, endTick, note.midi, note.velocity, ticksPerMeasure, dimensions);
                        }
                    });
                }
            }
        } catch (e) {
            console.error(e);
        }

        return processedMidiMatrices.filter(matrix => matrix.some(row => row.some(val => val !== 0)));
    }

    static createEmptyMatrix(dimensions) {
        return new Array(dimensions).fill().map(() => new Array(dimensions).fill(0));
    }

    static fillMatrix(matrix, startTick, endTick, midiNote, velocity, ticksPerMeasure, dimensions) {
        const pitchIndex = midiNote % dimensions;
        const startIdx = Math.floor(startTick / ticksPerMeasure * matrix[0].length);
        const endIdx = Math.ceil(endTick / ticksPerMeasure * matrix[0].length);

        for (let x = startIdx; x < endIdx && x < matrix[0].length; x++) {
            matrix[pitchIndex][x] = Math.round(velocity * 127); // Adjust as per required scaling
        }
    }

    // Improved quantization logic
    static quantizeNotes(midiData, quantization) {
        const PPQ = midiData.header.ppq;
        const quantizationStep = PPQ / ((1 / quantization) / 4);
        let newNotes = [];

        if (!midiData.tracks[0]) return false;

        for (const note of midiData.tracks[0].notes) {
            const quantizedStartTime = Math.round(note.ticks / quantizationStep) * quantizationStep;
            const quantizedEndTime = Math.round((note.ticks + note.durationTicks) / quantizationStep) * quantizationStep;
            const quantizedDuration = (quantizedEndTime - quantizedStartTime) / PPQ;

            newNotes.push({
                ...note,
                time: quantizedStartTime / PPQ,
                ticks: quantizedStartTime,
                duration: quantizedDuration
            });
        }

        return newNotes;
    }

    // Enhanced matrix generation logic
    static async createMidiMatrix(midiDataNotes, dimensions, startOctave, ppq) {
        const midiMatrix = Array.from({ length: dimensions }, () => new Array(dimensions).fill(0));

        midiDataNotes.forEach(note => {
            const xStart = Math.floor(note.time * ppq * dimensions);
            const xEnd = Math.floor((note.time + note.duration) * ppq * dimensions);
            const y = (note.midi - (startOctave * 12)) % dimensions;

            if (y >= 0 && y < dimensions) {
                for (let x = xStart; x < xEnd && x < dimensions; x++) {
                    midiMatrix[y][x] = Math.max(midiMatrix[y][x], note.velocity);
                }
            }
        });

        return midiMatrix;
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
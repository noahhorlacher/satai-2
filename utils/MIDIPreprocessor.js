import MidiParser from 'midi-parser-js'

export default class MIDIPreprocessor {
    // outputs grayscale images.
    // x-axis is time, y-axis is pitch.
    // velocity 0 is black, velocity 127 is white.
    static async preprocess(midiFiles = [], options = {
        dimensions: 64,
        horizontalResolution: 1 / 16,
        stepSizeX: 16,
        transpositions: [+7, -7],
    }) {
        // final array of images
        const midiImages = []

        // create 64px*64px grayscale images
        const dimensions = options.dimensions

        // horizontal Resolution notes (1/16 = 16th notes)
        // notes get quantized to that resolution
        const horizontalResolution = options.horizontalResolution
        const quantization = horizontalResolution

        // What variants of the image should be created
        const stepSizeX = options.stepSizeX // step-size on x axis
        const transpositions = options.transpositions // for each x-step, create variations with these transpositions in semitones (aside from 0)

        // for each midi file
        for (let midiFileBuffer of midiFiles) {
            let midiData
            try {
                console.log('parsing midi file', midiFileBuffer)
                midiData = await MIDIPreprocessor.parseMidiPromise(midiFileBuffer).then(data => console.log(data)).catch(err => {
                    console.log('Failed to parse MIDI file', err)
                })
                console.log('got midi data', midiData)

                const midiNotes = midiData.tracks[0]

                // quantize notes
                // const quantizedNotes = MIDIPreprocessor.quantizeNotes([...midiNotes], quantization)
                console.log('notes', midiNotes)
                // for each step on x axis
                // create image with transposition 0
                // for each transposition
                // create image with transposition
            } catch (e) {
                console.error(e)
            }
        }


        return midiImages
    }

    static parseMidiPromise(midiFileBuffer) {
        return new Promise((resolve, reject) => {
            try {
                const midiFileUint8 = new Uint8Array(midiFileBuffer)
                MidiParser.parse(midiFileUint8, midiData => {
                    if (midiData) {
                        resolve(midiData)
                    } else {
                        reject(new Error("Failed to parse MIDI file"))
                    }
                })
            } catch (e) {
                reject(e)
            }
        })
    }

    static quantizeNotes(notes, quantization) {
        // Adjust midi note start times and durations to the nearest quantization step
        const quantizedNotes = []

        for (const note of notes) {
            // quantize note start time
            // quantize note duration
            // add quantized note to quantizedNotes
            note.start = Math.round(note.start / quantization) * quantization
        }

    }

    static createImage(notes, dimensions) {
        // Create a 64x64 grayscale image
    }

    static transposeNotes(notes, semitoneShift) {
        // Transpose notes by the given number of semitones
    }
}
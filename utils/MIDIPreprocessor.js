import { Midi } from '@tonejs/midi';

const chromaticInstruments = [...Array(113).keys()].filter(i => !(i === 48 || (i > 97 && i < 105)));

export default class MIDIPreprocessor {
    // Constants
    static DEFAULT_DIMENSIONS = { x: 256, y: 64 };
    static DEFAULT_START_OCTAVE = 3;
    static DEFAULT_HORIZONTAL_RESOLUTION = 1 / 8;
    static DEFAULT_STEP_SIZE_X = 1;
    static DEFAULT_TRANSPOSITIONS = [-7, 7];
    static DEFAULT_MINIMUM_NOTES = 16;
    static DEFAULT_MINIMUM_DIFFERENT_PITCHES = 5;

    // Process multiple MIDI files
    static async preprocess(midiFiles = [], options = {}, batchProgress) {
        const finalOptions = { ...this.defaultOptions(), ...options };
        const statusMessage = toRefs(useStatusMessageStore()).statusMessage;

        let allProcessedMidiMatrices = [];

        for (let index = 0; index < midiFiles.length; index++) {
            statusMessage.value = `Processing batch ${batchProgress}\nFile ${index + 1}/${midiFiles.length}...`;
            const midiArrayBuffer = midiFiles[index];
            const processedMidiMatrices = await this.processSingleMidi(midiArrayBuffer, finalOptions);
            allProcessedMidiMatrices = [...allProcessedMidiMatrices, ...processedMidiMatrices];
        }

        return this.filterEmptyMatrices(allProcessedMidiMatrices);
    }

    // Default options
    static defaultOptions() {
        return {
            dimensions: this.DEFAULT_DIMENSIONS,
            startOctave: this.DEFAULT_START_OCTAVE,
            horizontalResolution: this.DEFAULT_HORIZONTAL_RESOLUTION,
            stepSizeX: this.DEFAULT_STEP_SIZE_X,
            transpositions: this.DEFAULT_TRANSPOSITIONS,
            minimumNotes: this.DEFAULT_MINIMUM_NOTES,
            minimumDifferentPitches: this.DEFAULT_MINIMUM_DIFFERENT_PITCHES
        };
    }

    // Process a single MIDI file
    static async processSingleMidi(midiArrayBuffer, options) {
        try {
            const midiData = await this.getMidiData(midiArrayBuffer);
            const validTrackIndex = this.getValidTrackIndex(midiData);

            if (validTrackIndex === -1) {
                throw new Error('No valid track found');
            }

            const track = midiData.tracks[validTrackIndex];
            const processedTrack = this.processTrack(track, options, midiData.header.ppq, midiData.header.timeSignatures);
            return processedTrack;
        } catch (error) {
            console.error(error);
            return [];
        }
    }

    // Get MIDI data from ArrayBuffer
    static async getMidiData(arrayBuffer) {
        const blob = new Blob([arrayBuffer], { type: 'audio/midi' });
        const url = URL.createObjectURL(blob);
        return Midi.fromUrl(url);
    }

    // Find the valid track with most notes
    static getValidTrackIndex(midiData) {
        let maxNotes = 0, validTrackIndex = -1;
        midiData.tracks.forEach((track, index) => {
            if (chromaticInstruments.includes(track.instrument.number)) {
                if (track.notes.length > maxNotes) {
                    maxNotes = track.notes.length;
                    validTrackIndex = index;
                }
            }
        });

        if (validTrackIndex === -1) {
            throw new Error('No valid track found');
        }
        return validTrackIndex;
    }

    // Process the selected MIDI track
    static processTrack(track, options, PPQ, timeSignatures) {
        const processedMidiMatrices = [];

        const ticksPerMeasure = this.calculateTicksPerMeasure(PPQ, timeSignatures);

        track.notes.forEach(note => {
            const { startMeasure, endMeasure } = this.getNoteMeasureRange(note, ticksPerMeasure);
            for (let measure = startMeasure; measure <= endMeasure; measure++) {
                if (!processedMidiMatrices[measure]) {
                    processedMidiMatrices[measure] = this.createEmptyMatrix(options.dimensions);
                }
                const matrix = processedMidiMatrices[measure];
                this.fillMatrix(matrix, note, measure, ticksPerMeasure, options);
            }
        });

        return processedMidiMatrices.filter(matrix => this.isMatrixValid(matrix, options));
    }

    // Calculate ticks per measure
    static calculateTicksPerMeasure(PPQ, timeSignatures) {
        const timeSignature = timeSignatures[0] || [4, 4];
        return (timeSignature[0] / timeSignature[1]) * PPQ * 4;
    }

    // Get note measure range
    static getNoteMeasureRange(note, ticksPerMeasure) {
        const startMeasure = Math.floor(note.ticks / ticksPerMeasure);
        const endMeasure = Math.floor((note.ticks + note.durationTicks) / ticksPerMeasure);
        return { startMeasure, endMeasure };
    }

    // Create an empty matrix based on dimensions
    static createEmptyMatrix(dimensions) {
        return Array.from({ length: dimensions.y }, () => new Array(dimensions.x).fill(0));
    }

    // Fill matrix with note data
    static fillMatrix(matrix, note, measure, ticksPerMeasure, options) {
        const { dimensions, startOctave } = options;
        const pitchIndex = note.midi - (startOctave * 12);

        if (pitchIndex < 0 || pitchIndex >= dimensions.y) return;

        const startTick = note.ticks - (measure * ticksPerMeasure);
        const endTick = (note.ticks + note.durationTicks) - (measure * ticksPerMeasure);
        const startIdx = Math.floor(startTick / ticksPerMeasure * matrix[0].length);
        const endIdx = Math.ceil(endTick / ticksPerMeasure * matrix[0].length);

        for (let x = startIdx; x < endIdx && x < matrix[0].length; x++) {
            matrix[pitchIndex][x] = note.velocity;
        }
    }

    // Filter out matrices that do not meet the criteria
    static filterEmptyMatrices(matrices) {
        return matrices.filter(matrix => matrix.some(row => row.some(val => val !== 0)));
    }

    // Validate matrix based on minimum notes and pitch variety criteria
    static isMatrixValid(matrix, options) {
        const { minimumNotes, minimumDifferentPitches } = options;
        const nonZeroRows = matrix.filter(row => row.some(val => val !== 0));
        return nonZeroRows.length >= minimumDifferentPitches && nonZeroRows.reduce((acc, row) => acc + row.reduce((acc, val) => acc + (val !== 0 ? 1 : 0), 0), 0) >= minimumNotes;
    }
}
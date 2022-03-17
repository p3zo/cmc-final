"""Process polyphonic midi to make it more well-suited for chord extraction"""

import argparse
import os
import warnings
from functools import partial

import numpy as np
import pandas as pd
import pretty_midi
from music21 import converter, stream

try:
    THIS_DIR = os.path.dirname(os.path.realpath(__file__))
except:
    THIS_DIR = os.getcwd()

DATA_DIR = os.path.join(THIS_DIR, "../data")
OUTPUT_DIR = os.path.join(THIS_DIR, "../output")
INPUT_PATH = os.path.join(DATA_DIR, "Sakamoto_MerryChristmasMr_Lawrence.mid")

if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# note values, e.g. 1/4 = quarter note, 1/64 = 64th note
SHORTEST_NOTE = 1 / 64
SMOOTH_BEAT = 1
QUANTIZE_BEAT = 1 / 2


def parse_notes(pmid):
    """Parses a pretty_midi object into a 2D array with (start, end, note) columns."""
    points = []

    for instrument in pmid.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            points.append((note.start, note.end, note.pitch))

    dtype = np.dtype([("start", float), ("end", float), ("note", int)])
    notes = np.array(points, dtype=dtype)

    return np.sort(notes, order="start")


def load_midi_file(filepath):
    # warnings can be verbose when midi has no metadata e.g. tempo, key, time signature
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            midi = pretty_midi.PrettyMIDI(filepath)
        except Exception as e:
            print(f"Failed loading file {filepath}: {e}")
            return

    return midi


def round_to_target(target, x):
    """Round `x` to the closest `target`

    e.g.
        round_to_target(1.12, .5) == 1.00
        round_to_target(1.12, .25) == 1.00
        round_to_target(1.13, .25) == 1.25
    """
    y = 1 / target
    return round(x * y) / y


def quantize_times(times, beat):
    """Quantize an array of times to the closest beat.

    e.g. quantize_times([1.1, 1.2, 1.3, 1.4], 0.5) == [1., 1., 1.5, 1.5]
    """
    func = partial(round_to_target, beat)

    return list(map(func, times))


def drop_short_notes(notes, shortest_duration):
    """Removes notes that are shorter than shortest_duration"""

    short_note_ixs = []

    for ix, n in enumerate(notes):
        if n["end"] - n["start"] < shortest_duration:
            short_note_ixs.append(ix)

    return np.delete(notes, short_note_ixs)


def smooth_notes(notes, step_size):
    """Joins adjacent, repeated notes into single, longer notes"""

    df = pd.DataFrame(notes)

    steps = np.arange(df.start.min(), df.end.max() + step_size, step_size)

    for ix, s in enumerate(steps):
        if ix == 0:
            continue

        seg = df[(df["start"] >= steps[ix - 1]) & (df["end"] < s)]

        vcs = seg.note.value_counts()
        vcs1 = vcs[vcs > 1]

        for k, v in vcs1.items():
            svc = seg[seg.note == k]

            df.loc[svc.index[0], "end"] = seg.loc[svc.index[-1]]["end"]

            df.drop(index=svc.index[1:], inplace=True)

    tups = list(df.itertuples(index=False, name=None))

    dtype = np.dtype([("start", float), ("end", float), ("note", int)])

    return np.array(tups, dtype=dtype)


def mk_midi_from_notes(notes):
    midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    piano = pretty_midi.Instrument(program=piano_program)

    for e in notes:
        note = pretty_midi.Note(
            velocity=100, pitch=e["note"], start=e["start"], end=e["end"]
        )
        piano.notes.append(note)

    midi.instruments.append(piano)

    return midi


def prepare(
    filepath,
    shortest_note,
    smooth_beat,
    quantize_beat,
):
    """Pre-process midi to improve effectiveness of chord extraction algorithm"""

    # Load the midi file
    midi = load_midi_file(filepath)

    notes = parse_notes(midi)

    tempos = midi.get_tempo_changes()

    if len(tempos[0]) == 0:
        raise ValueError("No tempos")
    if len(tempos[0]) > 1:
        raise ValueError("Multiple tempos. TODO: handle tempo changes")

    bpm = tempos[1][0]
    whole_note_dur = 60 / bpm * 4  # seconds

    # TODO: calculate note durations based on time signature

    # convert durations from beats to seconds
    shortest_dur = whole_note_dur * shortest_note
    smooth_dur = whole_note_dur * smooth_beat
    quantize_dur = whole_note_dur * quantize_beat

    cleaned = drop_short_notes(notes, shortest_dur)
    smoothed = smooth_notes(cleaned, smooth_dur)
    cleaned_again = drop_short_notes(smoothed, shortest_dur * 4)

    # TODO: lengthen notes to fill in rests

    quantized_onsets = quantize_times(cleaned_again["start"], quantize_dur)
    quantized_offsets = quantize_times(cleaned_again["end"], quantize_dur)

    cleaned_again["start"] = quantized_onsets
    cleaned_again["end"] = quantized_offsets

    # If start == end, the notes last until the next onset of that pitch
    # so we move the end of the note back
    for cn in cleaned_again:
        if cn["start"] == cn["end"]:
            cn["end"] += quantize_dur

    return mk_midi_from_notes(cleaned_again)


def extract(filepath):
    score = converter.parse(filepath)

    # sChords = score.chordify()
    # # for thisChord in sChords.recurse().getElementsByClass("Chord"):
    # #     print(thisChord.measureNumber, thisChord.beatStr, thisChord)

    # sFlat = sChords.flatten()

    # sOnlyChords = sFlat.getElementsByClass("Chord")

    # displayPart = stream.Part(id="displayPart")

    # for i in range(0, len(sOnlyChords) - 1):
    #     thisChord = sOnlyChords[i]
    #     nextChord = sOnlyChords[i + 1]

    #     # if thisChord.isTriad() is True or thisChord.isSeventh() is True:
    #     closePositionThisChord = thisChord.closedPosition(forceOctave=4)
    #     closePositionNextChord = nextChord.closedPosition(forceOctave=4)

    #     m = stream.Measure()
    #     m.append(closePositionThisChord)
    #     m.append(closePositionNextChord)
    #     displayPart.append(m)

    # return displayPart
    return score.chordify()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        type=str,
        default=INPUT_PATH,
        help="Path to a midi file",
    )
    args = parser.parse_args()

    filepath = args.filepath
    songname = os.path.splitext(os.path.basename(filepath))[0]

    intermediate_path = os.path.join(OUTPUT_DIR, f"prepped-{songname}.mid")
    output_path = os.path.join(OUTPUT_DIR, f"chords-{songname}.mid")

    print("Prepping midi...")

    # Process the midi
    prepared = prepare(filepath, SHORTEST_NOTE, SMOOTH_BEAT, QUANTIZE_BEAT)
    prepared.write(intermediate_path)

    # Extract chords from the prepped midi
    extracted = extract(intermediate_path)
    midi_out = extracted.write("midi", fp=output_path)
    print(f"Wrote chords to {output_path}")

    # Prep one more time...
    prepared = prepare(output_path, SHORTEST_NOTE, SMOOTH_BEAT, QUANTIZE_BEAT)

    # Write the final file
    final_path = os.path.join(OUTPUT_DIR, f"double-prepped-{songname}.mid")
    prepared.write(final_path)

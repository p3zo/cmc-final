import random
import string

import miditoolkit
import numpy as np
from miditoolkit.midi.containers import Instrument, Marker, Note, TempoChange

BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4


def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    if np.isnan(probs).any():
        return None
    else:
        return probs


def nucleus(probs, p):
    probs /= sum(probs) + 1e-5
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word


def sampling(logit, p=None, t=1.0, is_training=False):

    logit = logit.squeeze().cpu().numpy()
    probs = softmax_with_temperature(logits=logit, temperature=t)

    if probs is None:
        return None

    if p is not None:
        cur_word = nucleus(probs, p=p)

    else:
        cur_word = weighted_sampling(probs)
    return cur_word


def write_midi(words, path_outfile, word2event):

    class_keys = word2event.keys()
    # words = np.load(path_infile)
    midi_obj = miditoolkit.midi.parser.MidiFile()

    bar_cnt = 0
    cur_pos = 0

    all_notes = []

    cnt_error = 0
    for i in range(len(words)):
        vals = []
        for kidx, key in enumerate(class_keys):
            vals.append(word2event[key][words[i][kidx]])
        # print(vals)

        if vals[3] == "Metrical":
            if vals[2] == "Bar":
                bar_cnt += 1
            elif "Beat" in vals[2]:
                beat_pos = int(vals[2].split("_")[1])
                cur_pos = bar_cnt * BAR_RESOL + beat_pos * TICK_RESOL

                # chord
                if vals[1] != "CONTI" and vals[1] != 0:
                    midi_obj.markers.append(Marker(text=str(vals[1]), time=cur_pos))

                if vals[0] != "CONTI" and vals[0] != 0:
                    tempo = int(vals[0].split("_")[-1])
                    midi_obj.tempo_changes.append(
                        TempoChange(tempo=tempo, time=cur_pos)
                    )
            else:
                pass
        elif vals[3] == "Note":

            try:
                pitch = vals[4].split("_")[-1]
                duration = vals[5].split("_")[-1]
                velocity = vals[6].split("_")[-1]

                if int(duration) == 0:
                    duration = 60
                end = cur_pos + int(duration)

                all_notes.append(
                    Note(
                        pitch=int(pitch), start=cur_pos, end=end, velocity=int(velocity)
                    )
                )
            except:
                continue
        else:
            pass

    # save midi
    piano_track = Instrument(0, is_drum=False, name="piano")
    piano_track.notes = all_notes
    midi_obj.instruments = [piano_track]
    midi_obj.dump(path_outfile)


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str

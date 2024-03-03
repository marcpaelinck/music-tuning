import itertools
import json
import math
from pprint import pprint

import numpy as np
import pandas as pd
from scipy.io import wavfile

from tuning.common.classes import AggregatedPartialDict, Instrument
from tuning.common.constants import Folder, InstrumentGroupName
from tuning.common.utils import get_path, parse_json_file, read_group_from_jsonfile


def get_notelist_summary():
    note_list = parse_json_file(
        Instrument,
        get_path(InstrumentGroupName.GONG_KEBYAR, Folder.ANALYSES, "partials_per_note.json"),
    )

    summary = [
        (
            note_info.note.value,
            sorted([partial.reduced_ratio for partial in note_info.partials.root]),
        )
        for note_info in note_list.notes
    ]
    pprint(summary)


def analyze_limits():
    with open(get_path(InstrumentGroupName.TEST, Folder.SOUND, "limits.json"), "r") as infile:
        limits = json.load(infile)

    blocks = [
        [limits[i][1], limits[i + 1][1]]
        for i in range(len(limits) - 1)
        if limits[i][0] == "S" and limits[i + 1][0] == "E"
    ]

    print(len(blocks))
    threshold = 441
    idx = 0
    while idx < len(blocks) - 1:
        block1 = blocks[idx]
        block2 = blocks[idx + 1]
        if block2[0] - block1[1] < threshold:
            block1.remove(block1[1])
            block1.append(block2[0])
            blocks.remove(block2)
        else:
            idx += 1
    print(len(blocks))
    threshold = 44100
    blocks = [block for block in blocks if block[1] - block[0] > threshold]
    print(len(blocks))


def summarize_spectrum_files(groupname: InstrumentGroupName):
    orchestra = get_spectrum_summary(group=groupname)
    SEQ_LENGTH = 5
    results = []
    for instrument in orchestra.instruments:
        for note in instrument.notes:
            amplitudes = pd.read_csv(
                get_path(groupname, Folder.SPECTRUM, note.spectrumfilepath), sep="\t"
            )["amplitude"]
            percentiles = [0.8, 0.95, 1]
            quantiles = [amplitudes.quantile(q) for q in percentiles]
            THRESHOLD = amplitudes.quantile(0.95)
            mid_seq = math.floor(SEQ_LENGTH / 2)
            sequences = list(zip(*[amplitudes[i:] for i in range(SEQ_LENGTH)]))
            peak_sequences = [
                sequence
                for sequence in sequences
                if all(sequence[i] < sequence[i + 1] for i in range(mid_seq))
                and all(sequence[i] > sequence[i + 1] for i in range(mid_seq, SEQ_LENGTH - 1))
                and sequence[mid_seq] > THRESHOLD
            ]
            results.append([instrument.name, note.name, THRESHOLD, len(peak_sequences)])
    pprint(results)
    print(
        min([res[-1] for res in results]),
        max([res[-1] for res in results]),
    )


def inspect_wav_file(groupname: InstrumentGroupName, file: str, start: float, stop: float):
    sample_rate, data = wavfile.read(get_path(groupname, Folder.SOUND, file))
    amplitudes = pd.DataFrame(data)
    amplitudes["time"] = (amplitudes.index / sample_rate).round(2)
    return amplitudes[(amplitudes.time >= start) & (amplitudes.time <= stop)]


def dissonance_s_array(f: np.array, g: np.array, ampl_f: float, ampl_g: float) -> float:
    """
    Returns the dissonance value between two frequencies, each with given amplitude.
    The frequency of f must be smaller or equal to that of g.
    [Sethares]
    """
    b1 = 3.5
    b2 = 5.75
    d = abs(g - f)
    s = 0.24 / (np.minimum(f, g) * 0.021 + 19)
    # l12 = min(ampl_to_loudness(f.amplitude), ampl_to_loudness(g.amplitude))
    l12 = min(ampl_f, ampl_g)
    result = l12 * (np.exp(-b1 * s * d) - np.exp(-b2 * s * d))
    return result


if __name__ == "__main__":
    f_partials = [100, 150, 246]
    g_partials = [100, 200, 300, 400]
    fg_combinations = np.array(np.meshgrid(f_partials, g_partials)).T.reshape(-1, 2)
    gg_combinations = np.array(np.meshgrid(g_partials, g_partials)).T.reshape(-1, 2)
    gg_combinations = gg_combinations[gg_combinations[:, 0] < gg_combinations[:, 1]]
    ff_combinations = np.array(np.meshgrid(f_partials, f_partials)).T.reshape(-1, 2)
    ff_combinations = ff_combinations[ff_combinations[:, 0] < ff_combinations[:, 1]]

    STEPS = 10
    step = 0.1
    increments = (np.array(list(range(STEPS))) * step)[:, np.newaxis]
    frequencies = np.sort(-(increments.flatten() + 100))

    f_values = np.append(
        np.tile(fg_combinations[:, 0], (STEPS, 1)), increments + gg_combinations[:, 0], axis=1
    )
    g_values = increments + np.append(fg_combinations[:, 1], gg_combinations[:, 1])

    # Add internal g combinations

    print(f_values)
    print(g_values)
    print(f"{f_values.shape=}, {g_values.shape=}")
    print(increments.flatten())
    print(frequencies)

import json
import math
from pprint import pprint

import pandas as pd
from scipy.io import wavfile

from tuning.common.classes import Instrument
from tuning.common.constants import FileType, InstrumentGroupName
from tuning.common.utils import get_path, parse_json_file, read_group_from_jsonfile


def get_notelist_summary():
    note_list = parse_json_file(
        Instrument,
        get_path(InstrumentGroupName.GONG_KEBYAR, FileType.ANALYSES, "partials_per_note.json"),
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
    with open(get_path(InstrumentGroupName.TEST, FileType.SOUND, "limits.json"), "r") as infile:
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
                get_path(groupname, FileType.SPECTRUM, note.spectrumfilepath), sep="\t"
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
    sample_rate, data = wavfile.read(get_path(groupname, FileType.SOUND, file))
    amplitudes = pd.DataFrame(data)
    amplitudes["time"] = (amplitudes.index / sample_rate).round(2)
    return amplitudes[(amplitudes.time >= start) & (amplitudes.time <= stop)]


if __name__ == "__main__":
    groupname = InstrumentGroupName.SEMAR_PAGULINGAN
    orchestra = read_group_from_jsonfile(groupname, read_sounddata=False, read_spectrumdata=False)
    instrumenttypes = {instr.instrumenttype for instr in orchestra.instruments}
    print(instrumenttypes)

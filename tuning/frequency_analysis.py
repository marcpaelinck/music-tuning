"""
Analyzes the spectrum of each file in the input data folder and saves all the frequencies 
having an amplitude higher than the given threshold.
Then aggregates all 'similar' harmonic ratios using a DISTINCTIVENESS value.
"""
import json
import math
import os
from enum import Enum, auto
from typing import List

import numpy as np
import pandas as pd
from pydantic import BaseModel, RootModel

GONG_KEBYAR = "gongkebyar"
SEMAR_PAGULINGAN = "semarpagulingan"
ANGKLUNG = "angklung"

DATA_FOLDER = ".\\tuning\\data\\" + GONG_KEBYAR
THRESHOLD = -48  # dB
# During the aggregation step, frequencies whose mutual ratio is less
# than the distinctiveness will be considered equal.
DISTINCTIVENESS = pow(2, 75 / 1200)  # 3/4 of a semitone.
NOTE_SEQ = ["ding", "dong", "deng", "deung", "dung", "dang", "daing"]


class HarmonicType(Enum):
    BASE_NOTE = auto()
    HARMONIC = auto()


class Harmonic(BaseModel):
    freq: float
    ampl: float


HarmonicList = RootModel[List[Harmonic]]


class TypedHarmonic(BaseModel):
    harmonic: Harmonic
    ratio: float
    reduced_ratio: float
    type: HarmonicType


TypedHarmonicList = RootModel[List[TypedHarmonic]]


class AggregatedHarmonic(BaseModel):
    ratio: float
    ampl: float
    typed_harmonics: TypedHarmonicList


AggregatedHarmonicList = RootModel[List[AggregatedHarmonic]]


class Note(BaseModel):
    group: str
    instrument: str
    tuning: str
    name: str
    octave: str
    freq: float
    index: int
    typed_harmonics: TypedHarmonicList


NoteList = RootModel[List[Note]]

Frequency = float
Ratio = float
Amplitude = float
Interval = float
# Harmonic = dict[str, float]
# Note = dict[str, str | list[Harmonic]]


def reduce_to_octave(ratio: Ratio) -> Ratio:
    # Returns the ratio reduced to an ratio within an octave (1 <= reduced ratio < 2)
    return pow(2, math.log(ratio, 2) % 1)


def get_harmonics(folder: str, file: str, octave_range: list[Frequency]) -> list[Note]:
    """
    Determines frequency peaks by identifying frequencies that exceed the the given threshold,
    where the preceding 2 have a lower amplitude and are increasing, while following 2 frequencies
    have a lower amplitude and are decreasing.
    Then determines the frequency with maximum amplitude using quadratic interpolation.
    Args:
        folder (str): relative or absolute folder path
        file (str): file name
        octave (str): list containing lowest and highest freq of the octave

    Returns:
        list[dict[str, str | dict[str, float]]]: a list of peak frequencies, each described as a dict
    """
    spectrum_df = pd.read_csv(os.path.join(folder, file), sep="\t")
    spectrum_dict = spectrum_df.rename(
        columns={"Frequency (Hz)": "Hz", "Level (dB)": "dB"}
    ).to_dict(orient="list")

    # Create list of tuples (freq, ampl)
    spectrum_tuples = list(zip(spectrum_dict["Hz"], spectrum_dict["dB"]))
    # Generate all sequences of five successive tuples
    sequences = list(zip(*[spectrum_tuples[i:] for i in range(5)]))
    # Select sequences containing a peak above the given threshold, then split each sequence in two lists: freq and ampl
    peak_seq = [
        seq
        for seq in sequences
        if seq[0][1] < seq[1][1] < seq[2][1] > seq[3][1] > seq[4][1]
        and seq[2][1] > THRESHOLD
    ]
    peak_seq = [([p[0] for p in seq], [p[1] for p in seq]) for seq in peak_seq]
    # For each sequence, perform polynomial interpolation (ax^2 + bx + c) and calculate max frequency = -b/2a
    poly = [np.polyfit(p[0], p[1], 2) for p in peak_seq]
    peaks = [
        (x := -p[1] / (2 * p[0]), p[0] * pow(x, 2) + p[1] * x + p[2]) for p in poly
    ]
    typed_harmonics = [
        Harmonic(
            freq=int(p[0]),
            ampl=round(p[1], 1),
        )
        for p in peaks
    ]

    # Determine the base note: this is the harmonic withtin the octave range with the highest amplitude.
    harmonics_within_octave = [
        harmonic
        for harmonic in typed_harmonics
        if octave_range[0] * 0.9 < harmonic.freq < octave_range[1] * 1.1
    ]
    basenote = next(
        harmonic
        for harmonic in harmonics_within_octave
        if harmonic.ampl == max(h.ampl for h in harmonics_within_octave)
    )
    basenoteindex = typed_harmonics.index(basenote)

    # Create typed harmonics
    typed_harmonics = [
        TypedHarmonic(
            harmonic=harmonic,
            ratio=round((ratio := round(harmonic.freq / basenote.freq, 2)), 5),
            reduced_ratio=round(reduce_to_octave(ratio), 5),
            type=HarmonicType.BASE_NOTE
            if harmonic is basenote
            else HarmonicType.HARMONIC,
        )
        for harmonic in typed_harmonics
    ]

    group = folder.split("\\")[-1]
    categories = file.split(".")[0].split("-")
    return Note(
        group=group,
        instrument=categories[1],
        tuning=categories[2],
        name=categories[3],
        octave=categories[4],
        freq=basenote.freq,
        index=basenoteindex,
        typed_harmonics=typed_harmonics,
        comment="freq in Hz, ampl in dB.\n"
        + "Interval is the ratio with the first peak, which is considered to be the base note.\n"
        + "Reduced_ratio is related to the octave of the base note.",
    )


def get_octave_range(
    file: str, octave_ranges: dict[str, list[Frequency]]
) -> tuple[Frequency]:
    pos = file.find("octave") + 6
    octave = file[pos]
    return octave_ranges[octave]


def aggregate_next_pair(harmonics=list[list[TypedHarmonic]]) -> bool:
    def avg_harmonic(harmonics: list[TypedHarmonic]):
        return np.average([harmonic.reduced_ratio for harmonic in harmonics])

    i = 0
    j = i + 1
    aggregated = False
    while j < len(harmonics) and not aggregated:
        avg1 = avg_harmonic(harmonics[i])
        avg2 = avg_harmonic(harmonics[j])
        if avg2 / avg1 < DISTINCTIVENESS:
            if avg1 != 1 or avg2 == 1:
                # discard in case of base note
                harmonics[i].extend(harmonics[j])
            harmonics.remove(harmonics[j])
            aggregated = True
        i += 1
        j += 1
    return aggregated


def aggregate_harmonics(note_list: NoteList) -> AggregatedHarmonicList:
    all_harmonics = sorted(
        [
            [typed_harmonic]
            for note in note_list.root
            for typed_harmonic in note.typed_harmonics.root
        ],
        key=lambda lst: lst[0].reduced_ratio,
    )

    while aggregate_next_pair(all_harmonics):
        pass
    return AggregatedHarmonicList(
        AggregatedHarmonic(
            ratio=round(
                np.average([harmonic.reduced_ratio for harmonic in harmonics]), 5
            ),
            ampl=round(
                np.average([harmonic.harmonic.ampl for harmonic in harmonics]), 5
            ),
            typed_harmonics=harmonics,
        )
        for harmonics in all_harmonics
    )


def determine_harmonics_all_files(folder: str):
    """
    Determines frequency peaks for all files in the given folder
    Args:
        folder_in (str): input folder containing the spectra of individual notes
        folder_out (str): output folder for results file
    """
    with open(os.path.join(folder, "octaves_ranges.json"), "r") as octfile:
        octave_ranges = json.load(octfile)

    file_names = [
        f
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.startswith("spectrum")
    ]
    note_list = NoteList(
        sorted(
            [
                get_harmonics(
                    folder, filename, get_octave_range(filename, octave_ranges)
                )
                for filename in file_names
            ],
            key=lambda note: NOTE_SEQ.index(note.name),
        )
    )
    with open(os.path.join(folder, "harmonics_per_note.json"), "w") as outfile:
        outfile.write(note_list.model_dump_json(indent=4))

    aggregated_harmonics = aggregate_harmonics(note_list)
    with open(os.path.join(folder, "aggregated_harmonics.json"), "w") as outfile:
        outfile.write(aggregated_harmonics.model_dump_json(indent=4))


if __name__ == "__main__":
    determine_harmonics_all_files(DATA_FOLDER)

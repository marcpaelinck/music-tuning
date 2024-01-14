"""
Analyzes the spectrum of each file in the input data folder and saves all the frequencies 
having an amplitude higher than the given threshold.
Then aggregates all 'similar' harmonic ratios using a DISTINCTIVENESS value.
"""
import json
import math
import os
import re
from typing import List

import numpy as np
import pandas as pd
from classes import (
    AggregatedHarmonic,
    AggregatedHarmonicList,
    Frequency,
    Harmonic,
    HarmonicType,
    Note,
    NoteList,
    Octave,
    Ratio,
    SpectrumInfo,
    Tone,
)

GONG_KEBYAR = "gongkebyar"
SEMAR_PAGULINGAN = "semarpagulingan"
ANGKLUNG = "angklung"

DATA_FOLDER = ".\\tuning\\data\\" + GONG_KEBYAR
THRESHOLD = -48  # dB
# During the aggregation step, frequencies whose mutual ratio is less
# than the distinctiveness will be considered equal.
DISTINCTIVENESS = pow(2, 75 / 1200)  # 3/4 of a semitone.
NOTE_SEQ = ["ding", "dong", "deng", "deung", "dung", "dang", "daing"]


def reduce_to_octave(ratio: Ratio) -> Ratio:
    # Returns the ratio reduced to an ratio within an octave (1 <= reduced ratio < 2)
    return pow(2, math.log(ratio, 2) % 1)


def get_harmonics(spectrum_info: SpectrumInfo, tone_list: list[Tone]) -> List[Note]:
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
        List[dict[str, str | dict[str, float]]]: a list of peak frequencies, each described as a dict
    """
    # Generate all sequences of five successive tuples
    sequences = list(zip(*[tone_list[i:] for i in range(5)]))
    # Select sequences containing a peak above the given threshold
    peak_sequences = [
        sequence
        for sequence in sequences
        if sequence[0].amplitude
        < sequence[1].amplitude
        < sequence[2].amplitude
        > sequence[3].amplitude
        > sequence[4].amplitude
        and sequence[2].amplitude > THRESHOLD
    ]
    # For each sequence, perform polynomial interpolation (ax^2 + bx + c)
    # and find frequency with max amplitude (frequency = -b/2a)
    poly = [
        np.polyfit([s.frequency for s in sequence], [s.amplitude for s in sequence], 2)
        for sequence in peak_sequences
    ]
    tones = [
        Tone(
            frequency=int(x := -p[1] / (2 * p[0])),
            amplitude=round(p[0] * pow(x, 2) + p[1] * x + p[2], 1),
        )
        for p in poly
    ]

    # Determine the base note: this is the harmonic within the octave range with the highest amplitude.
    tones_within_octave = [
        tone
        for tone in tones
        if spectrum_info.octave.start_freq * 0.9
        < tone.frequency
        < spectrum_info.octave.end_freq * 1.1
    ]
    basenote = next(
        tone
        for tone in tones_within_octave
        if tone.amplitude == max(h.amplitude for h in tones_within_octave)
    )
    basenoteindex = tones.index(basenote)

    # Create harmonics
    harmonics = [
        Harmonic(
            tone=harmonic,
            ratio=(round(ratio := harmonic.frequency / basenote.frequency, 5)),
            reduced_ratio=round(reduce_to_octave(ratio), 5),
            type=HarmonicType.BASE_NOTE
            if harmonic is basenote
            else HarmonicType.HARMONIC,
        )
        for harmonic in tones
    ]

    return Note(
        spectrum=spectrum_info,
        freq=basenote.frequency,
        index=basenoteindex,
        harmonics=harmonics,
    )


def create_note_list_from_folder(folder: str) -> NoteList:
    """Creates a NoteList object from the spectrum files in the given folder.
    For each file in the folder, a Note object is created containing a
    a list of harmonics belonging to that note.

    Args:
        folder (str): full path to the folder containing spectrum files.

    Returns:
        List[Note]: The list as decribed above.
    """

    def get_spectrum_info(
        file: str, octave_ranges: dict[str, List[Frequency]]
    ) -> SpectrumInfo:
        # Parses the file name into a SpectrumInfo object.
        items = file.split(".")[0].split("-")
        return SpectrumInfo(
            instrument=items[1],
            tuning=items[2],
            note=items[3],
            octave=Octave(
                sequence=int(octave := items[5]),
                start_freq=octave_ranges[octave][0],
                end_freq=octave_ranges[octave][1],
            ),
        )

    def get_tone_list(file) -> list[Tone]:
        # converts the contents of a spectrum file into a list of Tone objects.
        spectrum_df = pd.read_csv(os.path.join(folder, file), sep="\t")
        spectrum_list = spectrum_df.rename(
            columns={"Frequency (Hz)": "frequency", "Level (dB)": "amplitude"}
        ).to_dict(orient="records")
        return [Tone(**record) for record in spectrum_list]

    with open(os.path.join(folder, "octaves_ranges.json"), "r") as octfile:
        octave_ranges = json.load(octfile)

    file_names = [
        f
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and re.match("^spectrum-[a-z\\d\\-]+.txt$", f)
    ]

    return NoteList(
        group=os.path.basename(folder),
        comment="freq in Hz, ampl in dB.\n"
        + "Interval is the ratio with the first peak, which is considered to be the base note.\n"
        + "Reduced_ratio is related to the octave of the base note.",
        notes=sorted(
            [
                get_harmonics(
                    spectrum_info=get_spectrum_info(filename, octave_ranges),
                    tone_list=get_tone_list(filename),
                )
                for filename in file_names
            ],
            key=lambda note: NOTE_SEQ.index(note.spectrum.note),
        ),
    )


def group_harmonics(harmonics=List[List[Harmonic]]):
    """Groups similar harmonics together in the same sub-list.
       DISTINCTIVENESS is used to determine similarity.
       Base notes are kept in a separate group. No 'similar' harmonics are added to this group.

    Args:
        harmonics (List[List[TypedHarmonic]]): Initial list.
        Initially, each (sub-)list contains exactly one harmonic.

    Returns:
        None
    """

    def avg_harmonic(harmonics: List[Harmonic]):
        return np.average([harmonic.reduced_ratio for harmonic in harmonics])

    # Scan the list from beginning to end. If two consecutive (lists of) tones are 'similar'
    # then join them into one list and start again from the beginning of the list.
    while True:
        i = 0
        aggregated = False
        while i + 1 < len(harmonics) and not aggregated:
            avg1 = avg_harmonic(harmonics[i])
            avg2 = avg_harmonic(harmonics[i + 1])
            if (avg2 / avg1 < DISTINCTIVENESS) and not (avg1 == 1 < avg2):
                # merge both lists but keep base notes (ratio==1) in a separate list.
                harmonics[i].extend(harmonics[i + 1])
                harmonics.remove(harmonics[i + 1])
                aggregated = True
            i += 1
        if not aggregated:
            break


def aggregate_harmonics(note_list: NoteList) -> AggregatedHarmonicList:
    """Generates a list of aggregated harmonics over all notes.

    Args:
        note_list (NoteList): List of notes with harmonics.

    Returns:
        AggregatedHarmonicList: List of aggregated harmonics.
    """
    all_harmonics = sorted(
        [[harmonic] for note in note_list.notes for harmonic in note.harmonics.root],
        key=lambda lst: lst[0].reduced_ratio,
    )

    group_harmonics(all_harmonics)

    return AggregatedHarmonicList(
        AggregatedHarmonic(
            ratio=round(
                np.average([harmonic.reduced_ratio for harmonic in harmonics]), 5
            ),
            ampl=round(
                np.average([harmonic.tone.amplitude for harmonic in harmonics]), 5
            ),
            harmonics=harmonics,
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
    note_list = create_note_list_from_folder((folder))

    with open(os.path.join(folder, "harmonics_per_note.json"), "w") as outfile:
        outfile.write(note_list.model_dump_json(indent=4))

    aggregated_harmonics = aggregate_harmonics(note_list)
    with open(os.path.join(folder, "aggregated_harmonics.json"), "w") as outfile:
        outfile.write(aggregated_harmonics.model_dump_json(indent=4))


if __name__ == "__main__":
    determine_harmonics_all_files(DATA_FOLDER)

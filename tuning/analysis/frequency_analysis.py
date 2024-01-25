"""
Analyzes the spectrum of each file in the input data folder and saves all the frequencies 
having an amplitude higher than the given threshold.
Then aggregates all 'similar' partial ratios using a DISTINCTIVENESS value.
"""
import json
import math
import os
import re
from typing import List

import numpy as np
import pandas as pd

from tuning.analysis.classes import (
    AggregatedPartial,
    AggregatedPartialList,
    Frequency,
    NoteInfo,
    NoteInfoList,
    Octave,
    Partial,
    PartialType,
    Ratio,
    SpectrumInfo,
    Tone,
)
from tuning.analysis.constants import FileType, InstrumentGroup
from tuning.analysis.utils import get_path

THRESHOLD = -48  # dB
# During the aggregation step, frequencies whose mutual ratio is less
# than the distinctiveness will be considered equal.
DISTINCTIVENESS = pow(2, 75 / 1200)  # 3/4 of a semitone.


def reduce_to_octave(ratio: Ratio) -> Ratio:
    # Returns the ratio reduced to an ratio within an octave (1 <= reduced ratio < 2)
    return pow(2, math.log(ratio, 2) % 1)


def get_partials(spectrum_info: SpectrumInfo, tone_list: list[Tone]) -> List[NoteInfo]:
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

    # Determine the fundamental frequency: this is the partial within the octave range with the highest amplitude.
    tones_within_octave = [
        tone
        for tone in tones
        if spectrum_info.octave.start_freq * 0.9
        < tone.frequency
        < spectrum_info.octave.end_freq * 1.1
    ]
    fundamental = next(
        tone
        for tone in tones_within_octave
        if tone.amplitude == max(h.amplitude for h in tones_within_octave)
    )
    fundamentalindex = tones.index(fundamental)

    # Create partials
    partials = [
        Partial(
            tone=partial,
            ratio=(round(ratio := partial.frequency / fundamental.frequency, 5)),
            reduced_ratio=round(reduce_to_octave(ratio), 5),
            type=PartialType.FUNDAMENTAL
            if partial is fundamental
            else PartialType.PARTIAL,
        )
        for partial in tones
    ]

    return NoteInfo(
        note=spectrum_info.note,
        instrument=spectrum_info.instrument,
        ombaktype=spectrum_info.ombaktype,
        octave=spectrum_info.octave,
        spectrum=spectrum_info,
        freq=fundamental.frequency,
        index=fundamentalindex,
        partials=partials,
    )


def create_note_list_from_folder(folder: str) -> NoteInfoList:
    """Creates a NoteList object from the spectrum files in the given folder.
    For each file in the folder, a Note object is created containing a
    a list of partials belonging to that note.

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
            ombaktype=items[2],
            note=items[3],
            octave=Octave(
                sequence=int(octave := items[4]),
                start_freq=octave_ranges[octave][0],
                end_freq=octave_ranges[octave][1],
            ),
        )

    def create_tone_list(file) -> list[Tone]:
        # converts the contents of a spectrum file into a list of Tone objects.
        spectrum_df = pd.read_csv(os.path.join(folder, file), sep="\t")
        # spectrum_list = spectrum_df.rename(
        #     columns={"Frequency (Hz)": "frequency", "Level (dB)": "amplitude"}
        # ).to_dict(orient="records")
        spectrum_list = spectrum_df.set_axis(
            ["frequency", "amplitude"], axis=1
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

    return NoteInfoList(
        group=os.path.basename(folder),
        comment="freq in Hz, ampl in dB.\n"
        + "Interval is the ratio with the first peak, which is considered to be the fundamental.\n"
        + "Reduced_ratio is related to the octave of the fundamental.",
        notes=sorted(
            [
                get_partials(
                    spectrum_info=get_spectrum_info(filename, octave_ranges),
                    tone_list=create_tone_list(filename),
                )
                for filename in file_names
            ],
            key=lambda noteinfo: noteinfo.note,
        ),
    )


def group_partials(partials=List[List[Partial]]):
    """Groups similar partials together in the same sub-list.
       DISTINCTIVENESS is used to determine similarity.
       Fundamentals are kept in a separate group. No 'similar' partials are added to this group.

    Args:
        partials (List[List[TypedPartial]]): Initial list.
        Initially, each (sub-)list contains exactly one partial.

    Returns:
        None
    """

    def avg_partial(partials: List[Partial]):
        return np.average([partial.reduced_ratio for partial in partials])

    # Scan the list from beginning to end. If two consecutive (lists of) tones are 'similar'
    # then join them into one list and start again from the beginning of the list.
    while True:
        i = 0
        aggregated = False
        while i + 1 < len(partials) and not aggregated:
            avg1 = avg_partial(partials[i])
            avg2 = avg_partial(partials[i + 1])
            if (avg2 / avg1 < DISTINCTIVENESS) and not (avg1 == 1 < avg2):
                # merge both lists but keep fundamentals (ratio==1) in a separate list.
                partials[i].extend(partials[i + 1])
                partials.remove(partials[i + 1])
                aggregated = True
            i += 1
        if not aggregated:
            break


def aggregate_partials(note_list: NoteInfoList) -> AggregatedPartialList:
    """Generates a list of aggregated partials over all notes.

    Args:
        note_list (NoteList): List of notes with partials.

    Returns:
        AggregatedPartialList: List of aggregated partials.
    """
    all_partials = sorted(
        [[partial] for note in note_list.notes for partial in note.partials.root],
        key=lambda lst: lst[0].reduced_ratio,
    )

    group_partials(all_partials)

    return AggregatedPartialList(
        AggregatedPartial(
            ratio=round(np.average([partial.reduced_ratio for partial in partials]), 5),
            ampl=round(np.average([partial.tone.amplitude for partial in partials]), 5),
            partials=partials,
        )
        for partials in all_partials
    )


def determine_partials_all_files(group: InstrumentGroup):
    """
    Determines frequency peaks for all files in the given folder
    Args:
        instrument (str): subfolder containing the data
        folder_out (str): output folder for results file
    """
    note_list = create_note_list_from_folder(get_path(group, FileType.SPECTRUM))

    with open(
        get_path(group, FileType.OUTPUT, "partials_per_note.json"), "w"
    ) as outfile:
        outfile.write(note_list.model_dump_json(indent=4))

    aggregated_partials = aggregate_partials(note_list)
    with open(
        get_path(group, FileType.OUTPUT, "aggregated_partials.json"), "w"
    ) as outfile:
        outfile.write(aggregated_partials.model_dump_json(indent=4))


if __name__ == "__main__":
    determine_partials_all_files(InstrumentGroup.GONG_KEBYAR)
    import json

    # with open(
    #     get_path(
    #         InstrumentGroup.GONG_KEBYAR, FileType.OUTPUT, "partials_per_note.json"
    #     ),
    #     "r",
    # ) as infile:
    #     json_repr = json.load(infile)
    # note_list = NoteList.model_validate_json(json_data=json.dumps(json_repr))

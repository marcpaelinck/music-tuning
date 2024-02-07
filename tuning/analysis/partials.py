"""
Analyzes the spectrum of each file in the input data folder and saves all the frequencies 
having an amplitude higher than the given threshold.
Then aggregates all 'similar' partial ratios using a DISTINCTIVENESS value.
"""

import logging
import math

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from tuning.common.classes import InstrumentGroup, Note, Partial, Ratio, Tone
from tuning.common.constants import (
    DEFAULT_PARTIAL_COUNT,
    DISTINCTIVENESS,
    INSTRUMENT_FIELDS,
    NOTE_FIELDS,
    OCTAVE_FIELDS,
    SPECTRUM_FILE_PATTERN,
    FileType,
    InstrumentGroupName,
)
from tuning.common.utils import read_group_from_jsonfile

logging.basicConfig(
    format="%(asctime)s - %(name)-12s %(levelname)-7s: %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def reduce_to_octave(ratio: Ratio) -> Ratio:
    # Returns the ratio reduced to an ratio within an octave (1 <= reduced ratio < 2)
    return pow(2, math.log(ratio, 2) % 1)


def interpolate_max(x_values, y_values) -> Tone:
    f = CubicSpline(x_values, y_values)
    x_roots = f.derivative().roots()
    y_roots = f(x_roots)
    index_max = np.argmax(y_roots)
    return x_roots[index_max], y_roots[index_max]


def select_loudest_partial_regions(
    tone_list: list[Tone],
    count: int = DEFAULT_PARTIAL_COUNT,
    distinct: float = DISTINCTIVENESS,
) -> list[list[Tone]]:
    ...
    SEQ_LENGTH = 5
    mid_seq = math.floor(SEQ_LENGTH / 2)
    # Create sequences of five consecutive tones
    sequences = list(zip(*[tone_list[i:] for i in range(SEQ_LENGTH)]))
    threshold = pd.Series([t.amplitude for t in tone_list]).quantile(0.95)

    # Select sequences containing a peak above the given threshold
    sequences = [
        sequence
        for sequence in sequences
        if sequence[mid_seq].amplitude > threshold
        and all(sequence[i].amplitude < sequence[mid_seq].amplitude for i in range(mid_seq))
        and all(
            sequence[mid_seq].amplitude > sequence[i + 1].amplitude
            for i in range(mid_seq, SEQ_LENGTH - 1)
        )
    ]
    logger.info(f"--- Found {len(sequences)} peaks.")

    # Select the top x sequences, ignore sequences that are too close to already selected sequences
    logger.info(f"--- Selecting {count} loudest partials.")
    sequences = sorted(sequences, key=lambda seq: seq[mid_seq].amplitude, reverse=True)
    top_sequences = []
    for new_seq in sequences:
        if all(
            max(new_seq[mid_seq].frequency, seq[mid_seq].frequency)
            / min(new_seq[mid_seq].frequency, seq[mid_seq].frequency)
            > distinct
            for seq in top_sequences
        ):
            top_sequences.append(new_seq)
        if len(top_sequences) > count:
            break
    return top_sequences


def get_partials(note: Note, count: int = DEFAULT_PARTIAL_COUNT) -> list[Partial]:
    """
    Determines frequency peaks by identifying frequencies that exceed the given threshold,
    where the preceding 2 values have a lower amplitude and are increasing, while the following 2
    values have a lower amplitude and are decreasing.
    Then determines the frequency with maximum amplitude using quadratic interpolation.
    Args:
        folder (str): relative or absolute folder path
        file (str): file name
        octave (str): list containing lowest and highest freq of the octave

    Returns:
        List[dict[str, str | dict[str, float]]]: a list of peak frequencies, each described as a dict
    """
    # Generate all sequences of five successive tuples
    peak_sequences = select_loudest_partial_regions(note.spectrum.tones, count=count)

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
    tones = sorted(tones, key=lambda t: t.amplitude, reverse=True)

    # Determine the fundamental frequency: this is the partial within the octave range with the highest amplitude.
    tones_within_octave = [
        tone
        for tone in tones
        if note.octave.start_freq * 0.9 < tone.frequency < note.octave.end_freq * 1.1
    ]
    fundamental = next(
        (
            tone
            for tone in tones_within_octave
            if tone.amplitude == max(h.amplitude for h in tones_within_octave)
        ),
        None,
    )

    # Determine partials to keep
    tones_to_keep = sorted(tones, key=lambda t: t.amplitude, reverse=True)[:count]
    if not fundamental in tones_to_keep:
        x = 1

    # Create partials
    partials = [
        Partial(
            tone=tone,
            ratio=(round(tone.frequency / fundamental.frequency, 5)),
            isfundamental=(tone is fundamental),
        )
        for tone in tones_to_keep
    ]
    return partials
    # return NoteInfo(
    #     note=spectrum_info.note,
    #     instrument=spectrum_info.instrument,
    #     ombaktype=spectrum_info.ombaktype,
    #     octave=spectrum_info.octave,
    #     spectrum=spectrum_info,
    #     freq=fundamental.frequency,
    #     partial_index=fundamentalindex,
    #     partials=partials,
    # )


def create_partials(orchestra: InstrumentGroup, count: int = DEFAULT_PARTIAL_COUNT) -> None:
    """
    Determines frequency peaks for all files in the given folder
    Args:
        orchestra (InstrumentGroup): set of instruments
        keep (int): maximum number of partials to keep
    """

    if not orchestra.has_spectra:
        logger.warning("No spectra available: call create_spectra first.")
        return

    for instrument in orchestra.instruments:
        for note in instrument.notes:
            logger.info(
                f"Creating partials for {instrument.name} {instrument.code} {note.name} {note.order_in_soundfile}"
            )
            note.partials = get_partials(
                note=note,
                count=count,
            )

    return orchestra


# def group_partials(partials=List[List[Partial]]):
#     """Groups similar partials together in the same sub-list.
#        DISTINCTIVENESS is used to determine similarity.
#        Fundamentals are kept in a separate group. No 'similar' partials are added to this group.

#     Args:
#         partials (List[List[TypedPartial]]): Initial list.
#         Initially, each (sub-)list contains exactly one partial.

#     Returns:
#         None
#     """

#     def avg_partial(partials: List[Partial]):
#         return np.average([partial.reduced_ratio for partial in partials])

#     # Scan the list from beginning to end. If two consecutive (lists of) tones are 'similar'
#     # then join them into one list and start again from the beginning of the list.
#     while True:
#         i = 0
#         aggregated = False
#         while i + 1 < len(partials) and not aggregated:
#             avg1 = avg_partial(partials[i])
#             avg2 = avg_partial(partials[i + 1])
#             if (avg2 / avg1 < DISTINCTIVENESS) and not (avg1 == 1 < avg2):
#                 # merge both lists but keep fundamentals (ratio==1) in a separate list.
#                 partials[i].extend(partials[i + 1])
#                 partials.remove(partials[i + 1])
#                 aggregated = True
#             i += 1
#         if not aggregated:
#             break


# def aggregate_partials(note_list: NoteInfoList) -> AggregatedPartialList:
#     """Generates a list of aggregated partials over all notes.

#     Args:
#         note_list (NoteList): List of notes with partials.

#     Returns:
#         AggregatedPartialList: List of aggregated partials.
#     """
#     all_partials = sorted(
#         [[partial] for note in note_list.notes for partial in note.partials.root],
#         key=lambda lst: lst[0].ratio,
#     )

#     group_partials(all_partials)

#     return AggregatedPartialList(
#         AggregatedPartial(
#             ratio=round(np.average([partial.ratio for partial in partials]), 5),
#             ampl=round(np.average([partial.tone.amplitude for partial in partials]), 5),
#             partials=partials,
#         )
#         for partials in all_partials
#     )


if __name__ == "__main__":
    orchestra = read_group_from_jsonfile(InstrumentGroupName.SEMAR_PAGULINGAN)
    create_partials(orchestra)

    # with open(
    #     get_path(
    #         InstrumentGroup.GONG_KEBYAR, FileType.OUTPUT, "partials_per_note.json"
    #     ),
    #     "r",
    # ) as infile:
    #     json_repr = json.load(infile)
    # note_list = NoteList.model_validate_json(json_data=json.dumps(json_repr))

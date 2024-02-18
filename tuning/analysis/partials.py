"""
Analyzes the spectrum of each file in the input data folder and saves all the frequencies 
having an amplitude higher than the given threshold.
Then aggregates all 'similar' partial ratios using a DISTINCTIVENESS value.
"""

import math

from scipy.signal import find_peaks

from tuning.common.classes import (
    FreqUnit,
    InstrumentGroup,
    Note,
    NoteName,
    Partial,
    Ratio,
    Tone,
)
from tuning.common.constants import (
    DEFAULT_NR_OF_PARTIALS,
    DISTINCTIVENESS_CENT,
    MAX_OCTAVES,
    MIN_PROMINENCE,
    PEAK_WINDOW,
    InstrumentGroupName,
)
from tuning.common.utils import (
    convert_freq,
    convert_spectrum_freq,
    get_logger,
    get_path,
    read_group_from_jsonfile,
    save_group_to_jsonfile,
)

logger = get_logger(__name__)


def reduce_to_octave(ratio: Ratio) -> Ratio:
    # Returns the ratio reduced to an ratio within an octave (1 <= reduced ratio < 2)
    return pow(2, math.log(ratio, 2) % 1)


def get_partials(
    note: Note,
    count: int = DEFAULT_NR_OF_PARTIALS,
    distinct: float = DISTINCTIVENESS_CENT,
    prominence: float = MIN_PROMINENCE,
    peak_window: int = PEAK_WINDOW,
    max_octaves: int = MAX_OCTAVES,
) -> list[Partial]:
    STEP = 1
    spectrum_cent = convert_spectrum_freq(spectrum=note.spectrum, to_unit=FreqUnit.CENT, step=STEP)
    base_octave_cent = (
        convert_freq(note.octave.start_freq, from_unit=FreqUnit.HERZ, to_unit=FreqUnit.CENT),
        convert_freq(note.octave.end_freq, from_unit=FreqUnit.HERZ, to_unit=FreqUnit.CENT),
    )
    minfreq_cent = base_octave_cent[0]
    maxfreq_cent = minfreq_cent + max_octaves * 1200
    spectrum_cent.tones = [
        tone for tone in spectrum_cent.tones if minfreq_cent <= tone.frequency <= maxfreq_cent
    ]
    yvals = [tone.amplitude for tone in spectrum_cent.tones]
    minheight = -100
    mid_freq = (note.octave.end_freq - note.octave.start_freq) / 2
    window_length = math.ceil(
        convert_freq(mid_freq + peak_window // 2, from_unit=FreqUnit.HERZ, to_unit=FreqUnit.CENT)
        - convert_freq(mid_freq - peak_window // 2, from_unit=FreqUnit.HERZ, to_unit=FreqUnit.CENT)
    )

    indices, properties = find_peaks(
        yvals,
        height=minheight,
        threshold=None,
        distance=int(distinct / STEP),
        prominence=prominence,
        width=None,
        wlen=window_length,
        rel_height=0.5,
        plateau_size=None,
    )

    indices = list(indices)
    peaks_cent = [
        (spectrum_cent.tones[i], properties["prominences"][indices.index(i)]) for i in indices
    ]
    peaks_best_prominence_cent = sorted(peaks_cent, key=lambda p: p[1], reverse=True)[: 2 * count]
    best_peaks_cent = sorted(
        peaks_best_prominence_cent, key=lambda p: p[0].amplitude, reverse=True
    )[:count]

    peaks_herz = [
        (
            Tone(
                frequency=round(
                    convert_freq(tone.frequency, from_unit=FreqUnit.CENT, to_unit=FreqUnit.HERZ),
                    5,
                ),
                amplitude=round(tone.amplitude, 5),
            ),
            prominence,
        )
        for tone, prominence in best_peaks_cent
    ]

    # Determine the fundamental: the partial within the octave range with the highest amplitude.
    tones_within_octave = [
        (tone, prominence)
        for tone, prominence in peaks_herz
        if note.octave.start_freq < tone.frequency < note.octave.end_freq
    ]
    fundamental = next(
        iter(sorted([t[0] for t in tones_within_octave], key=lambda t: -t.amplitude)), None
    )

    partials = [
        Partial(
            tone=tone,
            ratio=(round(tone.frequency / fundamental.frequency, 5)),
            prominence=round(prominence, 5),
            isfundamental=(tone is fundamental),
        )
        for tone, prominence in peaks_herz
    ]

    return partials, properties


def create_partials(orchestra: InstrumentGroup, count: int = DEFAULT_NR_OF_PARTIALS) -> None:
    """
    Determines frequency peaks for all files in the given folder
    Args:
        orchestra (InstrumentGroup): set of instruments
        keep (int): maximum number of partials to keep
    """

    if not orchestra.has_spectra:
        logger.warning("No spectra available: call create_spectra first.")
        return

    properties = dict()
    for instrument in orchestra.instruments:
        if DEBUG and instrument.code != INSTR_NOTE[0]:
            continue
        for note in instrument.notes:
            if DEBUG and note.name is not INSTR_NOTE[1]:
                continue
            logger.info(
                f"Creating partials for {instrument.instrumenttype} {instrument.code} {note.name.value} {note.order_in_soundfile}"
            )
            idx = f"{instrument.code}-{note.name.value}"  ###########
            note.partials, properties[idx] = get_partials(
                note=note,
                count=count,
            )

    return orchestra


DEBUG = False
INSTR_NOTE = ("JEG1", NoteName.DING)

if __name__ == "__main__":
    GROUPNAME = InstrumentGroupName.SEMAR_PAGULINGAN
    orchestra = read_group_from_jsonfile(groupname=GROUPNAME, read_sounddata=False)
    create_partials(orchestra)
    print("saving results")
    save_group_to_jsonfile(orchestra)

    # with open(
    #     get_path(
    #         InstrumentGroup.GONG_KEBYAR, FileType.OUTPUT, "partials_per_note.json"
    #     ),
    #     "r",
    # ) as infile:
    #     json_repr = json.load(infile)
    # note_list = NoteList.model_validate_json(json_data=json.dumps(json_repr))

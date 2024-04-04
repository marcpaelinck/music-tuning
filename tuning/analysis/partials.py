"""
Analyzes the spectrum of each file in the input data folder and saves all the frequencies 
having an amplitude higher than the given threshold.
Then aggregates all 'similar' partial ratios using a DISTINCTIVENESS value.
"""

import math
import sys
from itertools import chain

from scipy.signal import find_peaks

from tuning.analysis.aggregation import aggregated_partials_list
from tuning.analysis.spectrum import truncate_spectrum
from tuning.common.classes import (
    FreqUnit,
    InstrumentGroup,
    Note,
    NoteName,
    Octave,
    Partial,
    Ratio,
    Spectrum,
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
    step_read_group_from_jsonfile,
    step_save_group_to_jsonfile,
)

logger = get_logger(__name__)


def reduce_to_octave(ratio: Ratio) -> Ratio:
    # Returns the ratio reduced to an ratio within an octave (1 <= reduced ratio < 2)
    return pow(2, math.log(ratio, 2) % 1)


def get_partials(
    spectrum: Spectrum,
    octave: Octave,
    count: int = DEFAULT_NR_OF_PARTIALS,
    distinct: float = DISTINCTIVENESS_CENT,
    prominence: float = MIN_PROMINENCE,
    peak_window: int = PEAK_WINDOW,
    max_octaves: int = MAX_OCTAVES,
) -> list[Partial]:
    """
    Retrieves a list of "strongest" partials from the frequency spectrum of the given note.
    The partials are selected on highest amplitude and highest peak prominence. The latter
    is a measure of the height of the peak compared to its surrounding. See this link:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html#scipy.signal.peak_prominences

    Args:
        note (Note):
        count (int, optional): Number of partials to retrieve. Defaults to DEFAULT_NR_OF_PARTIALS.
        distinct (float, optional): Minimum separation of the partials in Cents. Defaults to DISTINCTIVENESS_CENT.
        prominence (float, optional): Minimum prominence of the partials. See link above for more information.
                                      Defaults to MIN_PROMINENCE.
        peak_window (int, optional): Used to determine the peak prominence. Defaults to PEAK_WINDOW.
        max_octaves (int, optional): Maximum number of octaves to consider, starting with the octave
                                     containing the fundamental frequency. Defaults to MAX_OCTAVES.

    Returns:
        list[Partial]: The partials with highest amplitude and prominence.
    """
    STEP = 1
    spectrum_cent = convert_spectrum_freq(spectrum=spectrum, to_unit=FreqUnit.CENT, step=STEP)
    base_octave_cent = (
        convert_freq(octave.start_frequency, from_unit=FreqUnit.HERTZ, to_unit=FreqUnit.CENT),
        convert_freq(octave.end_frequency, from_unit=FreqUnit.HERTZ, to_unit=FreqUnit.CENT),
    )
    minfreq_cent = base_octave_cent[0]
    maxfreq_cent = minfreq_cent + max_octaves * 1200
    spectrum_cent = truncate_spectrum(spectrum_cent, min_freq=minfreq_cent, max_freq=maxfreq_cent)
    minheight = -100
    mid_freq = (octave.end_frequency - octave.start_frequency) / 2
    window_length = math.ceil(
        convert_freq(mid_freq + 2 * peak_window // 2, from_unit=FreqUnit.HERTZ, to_unit=FreqUnit.CENT)
        - convert_freq(mid_freq - 2 * peak_window // 2, from_unit=FreqUnit.HERTZ, to_unit=FreqUnit.CENT)
    )

    indices = set()
    while (prominence > 1) and (len(indices) < count):
        indices_, properties = find_peaks(
            spectrum_cent.amplitudes,
            height=minheight,
            threshold=None,
            distance=int(distinct / STEP),
            prominence=prominence,
            width=None,
            wlen=window_length,
            rel_height=0.5,
            plateau_size=None,
        )
        indices = indices | set(indices_)
        total_found = len(indices)
        if total_found < count:
            prominence -= 1

    indices = list(indices)
    peaks_cent = [
        (
            spectrum_cent.frequencies[i],
            spectrum_cent.amplitudes[i],
            properties["prominences"][indices.index(i)],
        )
        for i in indices
    ]
    octave_bounds_cent = (
        convert_freq(octave.start_frequency, from_unit=FreqUnit.HERTZ, to_unit=FreqUnit.CENT),
        convert_freq(octave.end_frequency, from_unit=FreqUnit.HERTZ, to_unit=FreqUnit.CENT),
    )
    peaks_cent_within_octave = {
        peak for peak in peaks_cent if octave_bounds_cent[0] <= peak[0] <= octave_bounds_cent[1]
    }
    # Find the [2*count] most prominent peaks
    peaks_best_prominence_cent = sorted(peaks_cent, key=lambda p: p[-1], reverse=True)[: 2 * count]

    # Add all peaks within the octave. These might have a low prominence if they are
    # wider than the prominence window length.
    peaks_best_prominence_cent = list(set(peaks_best_prominence_cent) | peaks_cent_within_octave)
    # Select the [count] peaks with highest amplitude
    best_peaks_cent = sorted(peaks_best_prominence_cent, key=lambda p: p[1], reverse=True)[:count]

    peaks_herz = [
        (
            Tone(
                frequency=round(
                    convert_freq(frequency, from_unit=FreqUnit.CENT, to_unit=FreqUnit.HERTZ),
                    5,
                ),
                amplitude=round(amplitude, 5),
            ),
            prominence,
        )
        for frequency, amplitude, prominence in best_peaks_cent
    ]

    # Determine the fundamental: the partial within the octave range with the highest amplitude.
    tones_within_octave = [
        (tone, prominence)
        for tone, prominence in peaks_herz
        if octave.start_frequency < tone.frequency < octave.end_frequency
    ]
    fundamental = next(iter(sorted([t[0] for t in tones_within_octave], key=lambda t: -t.amplitude)), None)

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


def step_create_partials(group: InstrumentGroup, count: int = DEFAULT_NR_OF_PARTIALS) -> InstrumentGroup:
    """
    Determines frequency peaks for all files in the given folder
    Args:
        orchestra (InstrumentGroup): set of instruments
        keep (int): maximum number of partials to keep
    """

    if not group.has_spectra:
        logger.warning("No spectra available: call create_spectra first.")
        return

    properties = dict()
    for instrument in group.instruments:
        for note in instrument.notes:
            logger.info(
                f"Determining partials for {instrument.instrumenttype} {instrument.code} {note.name.value} {note.order_in_soundfile}"
            )
            idx = f"{instrument.code}-{note.name.value}"  ###########
            note.partials = list()
            for spectrum in note.spectra:
                partials, properties[idx] = get_partials(
                    spectrum=spectrum,
                    octave=note.octave,
                    count=count,
                )
                note.partials.append(partials)
            note.aggr_partials = aggregated_partials_list(
                list(chain.from_iterable(note.partials)), dist_cent=50, keep_nr=sys.maxsize
            )
            fundamental_aggr = next((p for p in note.aggr_partials if p.isfundamental), None)
            if fundamental_aggr:
                note.fundamental_index = note.aggr_partials.index(fundamental_aggr)
                note.frequency = fundamental_aggr.tone.frequency
            # # Check that all aggr_partials are equal to the partials of the first sample
            # if not all(
            #     aggr.tone.frequency == part.tone.frequency for aggr, part in zip(note.aggr_partials, note.partials[0])
            # ):
            #     logger.warning("Not all aggregated partials are equal to the original partials")

    return group


if __name__ == "__main__":
    # Set this value before running
    GROUPNAME = InstrumentGroupName.SEMAR_PAGULINGAN

    orchestra = step_read_group_from_jsonfile(groupname=GROUPNAME, read_spectrumdata=True, save_spectrumdata=False)
    step_create_partials(orchestra)
    print("saving results")
    step_save_group_to_jsonfile(orchestra, save_spectrumdata=False)

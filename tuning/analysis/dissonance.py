"""
References:
Jensen: https://jjensen.org/DissonanceCurve.html
Völk 2015: Florian Völk 1 2015, Updated analytical expressions for critical bandwidth and critical-band rate

"""

from itertools import product

import numpy as np

from tuning.analysis.aggregation import dict_key_retrieve_instrgroup
from tuning.analysis.utils import get_minima
from tuning.common.classes import (
    AggregatedPartialDict,
    Frequency,
    Instrument,
    InstrumentType,
    Note,
    NoteName,
    OmbakType,
    Partial,
    Ratio,
    Tone,
)
from tuning.common.constants import (
    AGGREGATED_PARTIALS_FILE,
    Folder,
    InstrumentGroupName,
)
from tuning.common.utils import (
    ampl_to_loudness,
    db_to_ampl,
    get_logger,
    get_path,
    read_object_from_jsonfile,
)
from tuning.visualization.utils import PlotType, plot_graphs

logger = get_logger(__name__)


def dissonance_s(
    f_freq: float | np.ndarray[float],
    f_ampl: float | np.ndarray[float],
    g_freq: float | np.ndarray[float],
    g_ampl: float | np.ndarray[float],
) -> float | np.ndarray[float]:
    """
    Returns the dissonance value between two frequences, each with given amplitude.
    The frequency of f must be smaller or equal to that of g.
    [Sethares]
    """
    b1 = 3.5
    b2 = 5.75
    d = abs(g_freq - f_freq)
    s = 0.24 / (np.minimum(f_freq, g_freq) * 0.021 + 19)
    l12 = np.minimum(f_ampl, g_ampl)
    result = l12 * (np.exp(-b1 * s * d) - np.exp(-b2 * s * d))
    return result


def dissonance_j(
    f_freq: float | np.ndarray[float],
    f_ampl: float | np.ndarray[float],
    g_freq: float | np.ndarray[float],
    g_ampl: float | np.ndarray[float],
) -> float | np.ndarray[float]:
    """
    Returns the dissonance value between two frequences, each with given amplitude.
    [Jensen]
    """
    q = abs(g_freq - f_freq) / (0.021 * np.minimum(f_freq, g_freq) + 19)
    return f_ampl * g_ampl * (np.exp(-0.84 * q) - np.exp(-1.38 * q))


def dissonance_profile(
    f_partials: list[Partial],
    g_partials: list[Partial],
    fixed_ampl=None,
    is_reverse=False,
) -> list[list[Frequency], list[float]]:

    # Determine the fundamental (should be the same for both lists of partials)
    fundamental = next((partial.tone.frequency for partial in f_partials if partial.isfundamental), None)
    # Determine the step size by which the partials of g should be shifted
    # Create a list of stewise increments
    STEPS = 5000
    step = (1.5 * fundamental) / STEPS
    increments = (np.array(list(range(STEPS))) * step)[:, np.newaxis]

    def add_increments(freq_list, fundamental_freq) -> np.array:
        ratios = freq_list / fundamental_freq
        result = (increments * ratios) + freq_list
        return result

    def dissonances(
        fg_partial_pairs: list[list[Partial]],
        increment_fg: list[bool] = [False, False],
    ) -> tuple[np.array, list[list[float]]]:
        # Retrieve arrays of f and g frequencies and amplitudes
        f_freq = np.array([f_partial.tone.frequency for f_partial, _ in fg_partial_pairs])
        g_freq = np.array([g_partial.tone.frequency for _, g_partial in fg_partial_pairs])
        # f_freq = np.array([f_partial.ratio for f_partial, _ in fg_partial_pairs]) * fundamental
        # g_freq = np.array([g_partial.ratio for _, g_partial in fg_partial_pairs]) * fundamental
        f_ampl = np.array([fixed_ampl or f_partial.tone.amplitude for f_partial, _ in fg_partial_pairs])
        g_ampl = np.array([fixed_ampl or g_partial.tone.amplitude for _, g_partial in fg_partial_pairs])
        # Add increment rows to requested frequency array(s)
        f_freq = add_increments(f_freq, fundamental) if increment_fg[0] else f_freq
        g_freq = add_increments(g_freq, fundamental) if increment_fg[1] else g_freq
        dissonance_matrix = dissonance_j(f_freq, f_ampl, g_freq, g_ampl)
        return np.sum(dissonance_matrix, axis=min(1, len(dissonance_matrix.shape) - 1)), [
            f_freq,
            g_freq,
        ]  # f_freq, g_freq returned for test_print

    # Create the combinations of partials for which to calculate the dissonances.
    fg_combinations = [[f, g] for f, g in product(f_partials, g_partials)]
    ff_combinations = [[f, g] for f, g in product(f_partials, f_partials) if f.tone.frequency < g.tone.frequency]
    gg_combinations = [[f, g] for f, g in product(g_partials, g_partials) if f.tone.frequency < g.tone.frequency]

    # Create the x-axis values
    frequencies = (increments.flatten() + fundamental) / fundamental

    # Determine the pairwise dissonances.
    # Indicate which frequency/ies should be incremented step-wise.
    dissonance_fg, freqs_fg = dissonances(
        fg_partial_pairs=fg_combinations,
        increment_fg=(False, True),
    )
    dissonance_ff, freqs_ff = dissonances(
        fg_partial_pairs=ff_combinations,
        increment_fg=(False, False),
    )
    dissonance_gg, freqs_gg = dissonances(
        fg_partial_pairs=gg_combinations,
        increment_fg=(True, True),
    )

    total_dissonances = dissonance_fg + dissonance_ff + dissonance_gg

    # Reverse the left part of thefrequency graph
    if is_reverse:
        frequencies = np.flip(-frequencies)
        total_dissonances = np.flip(total_dissonances)

    return [list(frequencies), list(total_dissonances)]


def create_partials_from_list(p_list: list[tuple[float, float]]) -> list[Partial]:
    # Creates AggregatedPartials from a list of tuples (freq, ampl).
    # The first partial should be the fundamental.
    return [
        Partial(
            tone=Tone(frequency=p_list[i][0], amplitude=p_list[i][1]),
            ratio=p_list[i][0] / p_list[0][0],
            isfundamental=(i == 0),
        )
        for i in range(len(p_list))
    ]


def reduced_freq_dict(
    notelist: list[Note], ref_note: NoteName, group: AggregatedPartialDict, instrtype: InstrumentType
) -> dict[Note, Frequency]:
    # Convert all note frequencies to octave 0 relative to the ref_note
    reduced_freq_list = {
        notename: [
            note.frequency / pow(2, note.octave.index - (1 if note.name.index() < ref_note.index() else 0))
            for note in notelist
            if note.name == notename
        ]
        for notename in NoteName
    }
    # Calculate average frequency per note. Remove entries without values.
    reduced_freqs = {key: np.mean(np.array(freqs)) for key, freqs in reduced_freq_list.items() if freqs}

    # Check if the reference note occurs. If not, take the frequency from another group of this instrument
    if not reduced_freqs.get(ref_note, None):
        for key, (_, notes) in group.root.items():
            if key.startswith(instrtype.value):
                note = next((note for note in notes if note.name == ref_note), None)
                if note:
                    reduced_freqs[ref_note] = note.frequency / pow(
                        2, note.octave.index - (1 if note.name.index() < ref_note.index() else 0)
                    )
                    return reduced_freqs

    return reduced_freqs


if __name__ == "__main__":
    ...

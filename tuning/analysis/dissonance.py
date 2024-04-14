"""
References:
Jensen: https://jjensen.org/DissonanceCurve.html
Völk 2015: Florian Völk 1 2015, Updated analytical expressions for critical bandwidth and critical-band rate

"""

import time
from itertools import product
from typing import Any, Iterable

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


def dissonance_profile_old(
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

    # Reverse the first part of thefrequency graph
    if is_reverse:
        frequencies = np.flip(-frequencies)
        total_dissonances = np.flip(total_dissonances)

    return [list(frequencies), list(total_dissonances)]


def dissonance_profile(
    pairs: dict[Any, tuple[list[Partial], list[Partial]]],
    fixed_ampl=None,
    is_reverse=False,
) -> list[list[Frequency], list[float]]:
    """
    Calculated the dissonance graph values for two sets of partials.
    This method can include the calculation of the internal dissonances of f and g. However these are omitted in the current version
    because the values are constants that have no effect on the location or relative height of the consonance 'dips'.
    Args:
        f_partials (list[Partial]): The fixed set of partials
        g_partials (list[Partial]): The set of partials that will be shifted.
        fixed_ampl (_type_, optional): If given, this will replace the actual amplitudes of the partials. Defaults to None.
        is_reverse (bool, optional): Inverts the graph, with negative X axis values (frequencies).
                                     Can be used to replicate Sethares' way of displaying the dissonance graphs. Defaults to False.

    Returns:
        list[list[Frequency], list[float]]: The X and Y values of the graph.
    """

    start_time = time.time()
    # Determine the fundamental (should be the same for both lists of partials)
    fundamental = next((partial.tone.frequency for partial in pairs["pp"][0] if partial.isfundamental), None)
    # Determine the step size by which the partials of g should be shifted
    # Create a list of stewise increments
    STEPS = 5000
    step = (1.5 * fundamental) / STEPS
    increments = np.resize(
        np.array(list(range(STEPS))) * step, (1, STEPS, 1)
    )  # Vector containing the successive increments

    def add_increments_or_copydown(
        freq_array: np.ndarray, incr_indicator: bool, fundamental_freq: Frequency
    ) -> np.ndarray:
        """
        Adds a row for each increment of the list of partials by either incrementing or copying the partials.
        The list of partials can be split up in separate arrays, each with their own treatment (increment or copy).
        Example:
        freq_arrays = ([f1, f2, f3], [f4, f5]) with increment indicators (True, False)
        This will result in the following matrix (s=increment 0..N):
        [[f1    f2    f3    f4   f5],
         [f1+s  f2+s  f3+s  f4   f5],
         [f1+2s f2+2s f3+2s f4   f5],
                   . . . .
         [f1+Ns f2+Ns f3+Ns f4   f5]]
        Args:
            freq_arrays (tuple[np.ndarray]): Group of 1-D arrays to 'extend down'.
            incr_indicators (tuple[bool): Value for each array, indicates whether to increment (True) or copy the values.
            fundamental_freq (Frequency): Frequency of the fundamental.

        Returns:
            np.ndarray: A single 2-D matrix.
        """
        if incr_indicator:
            ratios = freq_array / fundamental_freq
            expanded = (increments * ratios) + freq_array
        else:
            expanded = np.tile(freq_array, (STEPS, 1))
        return expanded

    # create lists that generate all desired pairwise combinations of partials for which to calculate the dissonances.
    # The first note will be fixed, the second either shifted stepwise (f-g combination) or fixed (internal dissonance of f and g).
    # Note that the latter is omitted in the current version.
    # The lists can be split up in groups, each with a separate treatment (either be shifted or fixed).
    # Example:
    # f_partials = [fp1, fp2, fp3] and g_partials = [fg1, fg2] gives the following results:
    # first_grouped_list =  ([fp1, fp1, fp2, fp2, fp3, fp3,   fp1, fp2, gp1, gp2],) with increment indicator (False,)
    # second_grouped_list = ([gp1, gp2, gp1, gp2, gp1, gp2], [fp2, fp3, gp2, gp3]) with increment indicators (True, False)
    # This will result in the following pairs for which the dissonance will be calculated:
    # (fp1, gp1), (fp1, gp2), (fp2, gp1), (fp2, gp2), (fp3, gp1), (fp3, gp2), (fp1, fp2), (fp2, fp3), (gp1, gp2), (gp2, gp3)
    # f_partials = np.array([f_partials for f_partials, _ in pairs.values()])
    # g_partials = np.array([g_partials for _, g_partials in pairs.values()])
    f_partials = np.array([[f_partials] for f_partials, _ in pairs.values()])
    g_partials = np.array([[g_partials] for _, g_partials in pairs.values()])
    first_partial_array = np.repeat(f_partials, g_partials.shape[-1], axis=2)
    second_partial_array = np.tile(g_partials, (1, 1, f_partials.shape[-1]))
    # first_grouped_list = (sum([[f] * len(g_partials) for f in f_partials], []) + f_partials[:-1] + g_partials[:-1],)
    # second_grouped_list = (g_partials * len(f_partials), f_partials[1:] + g_partials[1:])
    set_freq = np.vectorize(lambda x: x.tone.frequency)
    set_ampl = np.vectorize(lambda x: x.tone.amplitude)
    first_freq_array = set_freq(first_partial_array)
    second_freq_array = set_freq(second_partial_array)
    first_ampl_array = set_ampl(first_partial_array)
    second_ampl_array = set_ampl(second_partial_array)

    # Expand the lists by adding a row for each frequency increment (the shifting of the second note)
    # Parameter incr_indicators determines whether the frequencies should be shifted (True) or remain fixed (False)
    first_freq = add_increments_or_copydown(
        freq_array=first_freq_array, incr_indicator=False, fundamental_freq=fundamental
    )
    second_freq = add_increments_or_copydown(
        freq_array=second_freq_array, incr_indicator=True, fundamental_freq=fundamental
    )
    # first_ampl_array = np.array(
    #     [fixed_ampl or f_partial.tone.amplitude for f_list in first_partial_array for f_partial in f_list]
    # )
    # second_ampl_array = np.array(
    #     [fixed_ampl or g_partial.tone.amplitude for g_list in second_partial_array for g_partial in g_list]
    # )

    dissonance_matrix = dissonance_j(first_freq, first_ampl_array, second_freq, second_ampl_array)
    total_dissonances = np.sum(dissonance_matrix, axis=len(dissonance_matrix.shape) - 1)

    # Create the x-axis values
    frequencies = (increments.flatten() + fundamental) / fundamental
    # Reverse the X-axis if requested
    if is_reverse:
        frequencies = np.flip(-frequencies)
        total_dissonances = np.flip(total_dissonances)

    logger.info(f"DISSONANCE CALCULATION TOOK {time.time() - start_time} SECONDS")
    return {key: [list(frequencies), list(total_dissonances[index])] for key, index in zip(pairs.keys(), range(1000))}


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

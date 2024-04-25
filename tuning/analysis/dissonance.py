"""
References:
Jensen: https://jjensen.org/DissonanceCurve.html
Völk 2015: Florian Völk 1 2015, Updated analytical expressions for critical bandwidth and critical-band rate

"""

import time
from itertools import product
from typing import Any

import numpy as np

from tuning.common.classes import Frequency, Note, NoteName, Partial, Ratio, Tone
from tuning.common.constants import (
    DEFAULT_NR_HARMONICS_DISS_GRAPH,
    DEFAULT_NR_PARTIALS_DISS_GRAPH,
    HARMONICS_AMPL_DB,
)
from tuning.common.utils import get_logger

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
    fundamental: Partial,
    is_reverse=False,
    end_shift_ratio=2.5,
) -> list[list[Frequency], list[float]]:
    """
    Calculated the dissonance graph values for two sets of partials.
    This method can include the calculation of the internal dissonances of f and g. However these are omitted in the current version
    because the values are constants that have no effect on the location or relative height of the consonance 'dips'.
    Args:
        f_partials (list[Partial]): The fixed set of partials
        g_partials (list[Partial]): The set of partials that will be shifted.
        is_reverse (bool, optional): Inverts the graph, with negative X axis values (frequencies).
                                     Can be used to replicate Sethares' way of displaying the dissonance graphs. Defaults to False.

    Returns:
        list[list[Frequency], list[float]]: The X and Y values of the graph.
    """

    start_time = time.time()
    fundamental_freq = fundamental.tone.frequency
    # Determine the step size by which the partials of g should be shifted
    # Create a list of stewise increments
    STEPS = 5000
    step = ((end_shift_ratio - 1) * fundamental_freq) / STEPS
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
    # first_partial_array =  [fp1, fp1, fp2, fp2, fp3, fp3, fp1, fp2, gp1, gp2] with increment indicator False
    # second_partial_array = [gp1, gp2, gp1, gp2, gp1, gp2, fp2, fp3, gp2, gp3] with increment indicator True
    # This will result in the following pairs for which the dissonance will be calculated:
    # (fp1, gp1), (fp1, gp2), (fp2, gp1), (fp2, gp2), (fp3, gp1), (fp3, gp2), (fp1, fp2), (fp2, fp3), (gp1, gp2), (gp2, gp3)
    max_len = max(len(p) for pair in pairs.values() for p in pair)
    # Ensure that all partial lists have the same length. Pad with zeroes.
    f_partials = np.array(
        [[np.pad(f_partials, (0, max_len - len(f_partials)), "constant")] for f_partials, _ in pairs.values()]
    )
    g_partials = np.array(
        [[np.pad(g_partials, (0, max_len - len(g_partials)), "constant")] for _, g_partials in pairs.values()]
    )
    # Create two arrays that will yield the pairs of partials to consider.
    first_partial_array = np.repeat(f_partials, g_partials.shape[-1], axis=2)
    second_partial_array = np.tile(g_partials, (1, 1, f_partials.shape[-1]))
    # Define functions to extract frequency and amplitude from the partials. Zero values should return zero.
    extract_freq = np.vectorize(lambda x: x.tone.frequency if isinstance(x, Partial) else 0)
    extract_ampl = np.vectorize(lambda x: x.tone.amplitude if isinstance(x, Partial) else 0)
    first_freq_array = extract_freq(first_partial_array)
    second_freq_array = extract_freq(second_partial_array)
    first_ampl_array = extract_ampl(first_partial_array)
    second_ampl_array = extract_ampl(second_partial_array)

    # Expand the lists by adding a row for each frequency increment (the shifting of the second group of partials)
    # Parameter incr_indicators determines whether the frequencies should be shifted (True) or remain fixed (False)
    first_freq = add_increments_or_copydown(
        freq_array=first_freq_array, incr_indicator=False, fundamental_freq=fundamental_freq
    )
    second_freq = add_increments_or_copydown(
        freq_array=second_freq_array, incr_indicator=True, fundamental_freq=fundamental_freq
    )

    dissonance_matrix = dissonance_j(first_freq, first_ampl_array, second_freq, second_ampl_array)
    total_dissonances = np.sum(dissonance_matrix, axis=len(dissonance_matrix.shape) - 1)

    # Create the x-axis values
    frequencies = (increments.flatten() + fundamental_freq) / fundamental_freq
    # Reverse the X-axis if requested
    if is_reverse:
        frequencies = np.flip(-frequencies)
        total_dissonances = np.flip(total_dissonances)

    # logger.info(f"DISSONANCE CALCULATION TOOK {time.time() - start_time} SECONDS")
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


def generate_partial_harmonic_pair(
    partials: list[Partial],
    reference_frequency: dict[NoteName, Frequency],
    nbr_partials: int = DEFAULT_NR_PARTIALS_DISS_GRAPH,
    nbr_harmonics: int = DEFAULT_NR_HARMONICS_DISS_GRAPH,
    harmonics_ampl_db: int = HARMONICS_AMPL_DB,
):
    """
    Generates a list of partials and a list of harmonics for a dissonance graph.

    Args:
        partials (list[Partial]): All available partials
        reduced_freqs (dict[NoteName, Frequency]): Frequency of each note, reduced to octave 0
        nbr_partials (int, optional): Required number of partials, including the fundamental. Defaults to MAX_NR_PARTIALS_DISS_GRAPH.
        nbr_harmonics (int, optional): Required number of harmonics, including the fundamental. Defaults to MAX_NR_HARMONICS_DISS_GRAPH.
        harmonics_ampl_db (int, optional): Fixed amplitude for harmonics. Defaults to HARMONICS_AMPL_DB.
        ref_note (NoteName, optional): Note from which to take the fundamental. Defaults to NoteName.DING.
        ref_octave (int, optional): Octave of the ref_note. Defaults to 3.

    Returns:
        _type_: _description_
    """
    partials = sorted(partials[:nbr_partials], key=lambda x: x.tone.frequency)
    for partial in partials:
        partial.tone.frequency = partial.ratio * reference_frequency
    harmonics = create_partials_from_list(
        [(reference_frequency * i, harmonics_ampl_db) for i in range(1, nbr_harmonics + 1)]
    )
    return (partials, harmonics)


def get_noteratio_dict(note_dict: dict[NoteName, Note], reference_note: Note) -> dict[NoteName, Ratio]:
    """Creates a dict of the ratio of note frequencies relative to a reference note. All notes
    are expected to span no more than one octave. Notes with a lower frequency than the reference notes
    will be moved up one octave.

    Args:
        note_dict (dict[NoteName, Note]): The notes to evaluate.
        reference_note (Note): The reference note.

    Returns:
         dict[NoteName, Ratio]: frequency ratios, where the ratio has value 1 for the reference note and all other
                                notes have a value between 1 and 2.
    """
    # Reduce all frequencies to the same octave (octave 0) and calculate ratio
    ref_freq = reference_note.frequency / pow(2, reference_note.octave.index)
    noteratio_dict = {
        notename: note.frequency / (pow(2, note.octave.index) * ref_freq) for notename, note in note_dict.items()
    }
    # correction if ref note is not DING
    noteratio_dict = {notename: freq * 2 if freq < 1 else freq for notename, freq in noteratio_dict.items()}
    return noteratio_dict


if __name__ == "__main__":
    ...

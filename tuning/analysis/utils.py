import numpy as np

from tuning.common.classes import (
    ConsonanceAnalysis,
    ConsonanceValues,
    Dissonance,
    Frequency,
    Partial,
    Tone,
)
from tuning.common.constants import NoteName


def remove_min_region(index, x_vector, y_vector) -> tuple[np.array]:
    idx1 = idx2 = index
    while idx1 > 0 and y_vector[idx1 - 1] > y_vector[idx1]:
        idx1 -= 1
    while idx2 < len(y_vector) - 1 and y_vector[idx2 + 1] > y_vector[idx2]:
        idx2 += 1
    return (
        np.delete(x_vector, np.s_[idx1 : idx2 + 1]),
        np.delete(y_vector, np.s_[idx1 : idx2 + 1]),
    )


def get_minima(x_vector, y_vector) -> list[Dissonance]:
    x_vect = np.copy(x_vector)
    y_vect = np.copy(y_vector)
    minima = []
    while len(y_vect) > 0:
        idx = np.argmin(y_vect)
        minima.append(Dissonance(ratio=x_vect[idx], dissonance=y_vect[idx]))
        x_vect, y_vect = remove_min_region(idx, x_vect, y_vect)
    return minima


def reduce_frequency(frequency: Frequency, octaves: int) -> Frequency:
    """
    Reduces frequency by the given number of octaves.

    Args:
        frequency (Frequency): The frequency to convert.
        octave_idx (int): The octave of the reference frequency (usually the fundamental)

    Returns:
        Frequency: The input frequency lowered with the given number of octaves
    """
    #
    return frequency / pow(2, octaves)


def create_consonance_analysis(
    part_freq: list[Frequency],
    harmonics_freq: list[Frequency],
) -> ConsonanceAnalysis:
    """Creates ConsonanceAnalysis object based one lists of frequencies. The list can be used
    as input for the generation of a dissonance curve.

    Args:
        part_freq (list[Frequency]): Sorted frequencies of the partials, including the fundamental.
        harmonics_freq (list[Frequency]): Sorted frequencies of the harmonics, including the fundamental.
        noteratio_dict (_type_, optional): Not used. Defaults to {NoteName.DING: 1.0}.

    Returns:
        ConsonanceAnalysis: _description_
    """
    ref_part = part_freq[0]
    partials = [
        Partial(tone=Tone(frequency=freq, amplitude=1), ratio=freq / ref_part, isfundamental=(freq == ref_part))
        for freq in part_freq
    ]
    ref_harm = harmonics_freq[0]
    harmonics = [
        Partial(tone=Tone(frequency=freq, amplitude=1), ratio=freq / ref_harm, isfundamental=(freq == ref_part))
        for freq in harmonics_freq
    ]
    noteratio_dict = {NoteName.DING: 1.0}
    return ConsonanceAnalysis(
        ref_note=NoteName.DING,
        ref_octave=0,
        ref_frequency=part_freq[0],
        fundamental=partials[0],
        all_partials=partials,
        all_harmonics=harmonics,
        noteratio_dict=noteratio_dict,
        optimal_consonances=(opt := ConsonanceValues(partials=partials, harmonics=harmonics, consonances=dict())),
        results=[opt],
    )


if __name__ == "__main__":
    x_vec = np.array(list(range(50)))
    basis = np.array([3, 5, 4, 3, 2, 1, 2, 3, 4, 3])
    arr_list = [basis for i in range(1, 6)]
    y_vec = np.concatenate(arr_list)
    print(y_vec)
    mins = get_minima(x_vec, y_vec)
    print(mins)

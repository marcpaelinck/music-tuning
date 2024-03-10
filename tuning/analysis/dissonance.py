"""
References:
Jensen: https://jjensen.org/DissonanceCurve.html
Völk 2015: Florian Völk 1 2015, Updated analytical expressions for critical bandwidth and critical-band rate

"""

import math
import time
from itertools import product

import numpy as np
import pandas as pd

from tuning.analysis.utils import get_minima
from tuning.common.classes import (
    AggregatedPartialDict,
    Instrument,
    InstrumentType,
    Partial,
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
    test_save=False,
):

    # Determine the fundamental (should be the same for both lists of partials)
    fundamental = next(
        (partial.tone.frequency for partial in f_partials if partial.isfundamental), None
    )
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
        f_ampl = np.array(
            [fixed_ampl or f_partial.tone.amplitude for f_partial, _ in fg_partial_pairs]
        )
        g_ampl = np.array(
            [fixed_ampl or g_partial.tone.amplitude for _, g_partial in fg_partial_pairs]
        )
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
    ff_combinations = [
        [f, g] for f, g in product(f_partials, f_partials) if f.tone.frequency < g.tone.frequency
    ]
    gg_combinations = [
        [f, g] for f, g in product(g_partials, g_partials) if f.tone.frequency < g.tone.frequency
    ]

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


def create_partials(p_list: list[tuple[float, float]]) -> list[Partial]:
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


def plot_diss_graph(pairs: list[tuple[list[Partial], list[Partial]]], **kwargs):
    profiler = dissonance_profile
    start_time = time.time()
    results = []
    for pair in pairs:
        ph_result = profiler(
            pair[0],
            pair[1],
            fixed_ampl=kwargs.get("amplitude", None),
        )
        pp_result = profiler(
            pair[1],
            pair[0],
            fixed_ampl=kwargs.get("amplitude", None),
            is_reverse=True,
            test_save=True if pair == pairs[-1] else False,
        )
        ph_peaks = get_minima(*ph_result)
        pp_peaks = get_minima(*pp_result)
        results.extend([pp_result + [pp_peaks], ph_result + [ph_peaks]])
        print(ph_peaks + pp_peaks)
    logger.info(f"total dissonance calculation time: {time.time() - start_time} seconds")
    plot_graphs(
        *results,
        nrows=len(results) // 2,
        ncols=2,
        plottype=PlotType.CONSONANCEPLOT,
        autox=True,
        **kwargs,
    )


def plot_dissonance_graphs(
    group: AggregatedPartialDict,
    *,
    object: InstrumentType,
    groupname: InstrumentGroupName,
    amplitude: float = None,
    **kwargs,
):
    MAX_DB = 50  # For conversion of DB relative to max. ampl, to DB relative to min. amplitude
    # TODO retrieving dict entry by using string literal is error prone
    keys = [key for key in list(group.root.keys()) if key.startswith(object.value)]
    if not keys:
        return False
    partial_harmonic_pairs = []
    plottitles = []
    for key in keys:
        plottitles.extend([f"octave {key.split('-')[1]}", ""])
        partials = group.root[key]
        for p in partials:
            p.tone.amplitude = MAX_DB - p.tone.amplitude  ##################
        fundamental = next((p for p in partials if p.isfundamental), None)
        f_freq = fundamental.tone.frequency
        harmonics = create_partials([(f_freq * i, MAX_DB) for i in range(1, 5)])
        partial_harmonic_pairs.append((partials, harmonics))
    pagetitle = f"Dissonance graph for {groupname.value} {object.value}" + (
        " (same amplitude for all partials)" if amplitude else ""
    )
    plot_diss_graph(
        partial_harmonic_pairs,
        is_decibel=True,
        amplitude=amplitude,
        plottitles=plottitles,
        pagetitle=pagetitle,
        **kwargs,
    )
    return True


def test_dissonance_graph(**kwargs):
    fundfreq = 500
    f_freqs = [(fundfreq * ratio, 1) for ratio in (1, 1.52, 3.46, 3.92)]
    g_freqs = [(fundfreq * ratio, 1) for ratio in (1, 2, 3, 4)]
    f_partials = create_partials(f_freqs)
    g_partials = create_partials(g_freqs)
    plot_diss_graph(f_partials, g_partials, **kwargs)


def test_dissonance_h_graph(**kwargs):
    f_partials = create_partials([(500 * i, 1) for i in range(1, 7)])
    ph_result = dissonance_profile(f_partials, f_partials, is_decibel=False)
    plot_graphs(ph_result, plottype=PlotType.SPECTRUMPLOT, show=True, **kwargs)


if __name__ == "__main__":
    groupname = InstrumentGroupName.SEMAR_PAGULINGAN
    aggregated_partials = read_object_from_jsonfile(
        AggregatedPartialDict, groupname, Folder.ANALYSES, AGGREGATED_PARTIALS_FILE
    )
    plot_dissonance_graphs(
        group=aggregated_partials,
        groupname=groupname,
        object=InstrumentType.JUBLAG,
        amplitude=None,
        show=True,
    )

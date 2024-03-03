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

from tuning.common.classes import AggregatedPartialDict, InstrumentType, Partial, Tone
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


def cbw_z(freq: float) -> float:
    """
    Returns the critical bandwidth (formula by Zwicker and Terhardt)
    [Völk 2015]
    """
    return 25 + 75 * pow(1 + 1.4 * pow(freq / 1000, 2), 0.69)


def cbw_v(freq: float) -> float:
    """
    Returns the critical bandwidth (formula by Völk)
    [Völk 2015]
    """
    return cbw_z(freq) * (1 - 1 / (pow(38.73 * freq / 1000, 2) + 1))


def dissonance_s(f: Tone, g: Tone, g_shift: float = 0, is_decibel=False) -> float:
    """
    Returns the dissonance value between two frequences, each with given amplitude.
    The frequency of f must be smaller or equal to that of g.
    [Sethares]
    """
    if is_decibel:
        f = Tone(frequency=f.frequency, amplitude=db_to_ampl(f.amplitude))
        g = Tone(frequency=g.frequency, amplitude=db_to_ampl(g.amplitude))
    d = abs(g.frequency - f.frequency)
    b1 = 3.5
    b2 = 5.75
    s = 0.24 / (min(f.frequency, g.frequency) * 0.021 + 19)
    # l12 = min(ampl_to_loudness(f.amplitude), ampl_to_loudness(g.amplitude))
    l12 = min(f.amplitude, g.amplitude)
    result = l12 * (math.exp(-b1 * s * d) - math.exp(-b2 * s * d))
    return result


def dissonance_s_array(freq_arrays: list[np.array], ampl_arrays: list[np.array]) -> float:
    """
    Returns the dissonance value between two frequences, each with given amplitude.
    The frequency of f must be smaller or equal to that of g.
    [Sethares]
    """
    f = freq_arrays[0]
    g = freq_arrays[1]
    b1 = 3.5
    b2 = 5.75
    d = abs(g - f)
    s = 0.24 / (np.minimum(f, g) * 0.021 + 19)
    # l12 = min(ampl_to_loudness(f.amplitude), ampl_to_loudness(g.amplitude))
    l12 = np.minimum(ampl_arrays[0], ampl_arrays[1])
    result = l12 * (np.exp(-b1 * s * d) - np.exp(-b2 * s * d))
    return result


def dissonance_j(f: Tone, g: Tone, g_shift: float = 0, is_decibel=False) -> float:
    """
    Returns the dissonance value between two frequences, each with given amplitude.
    [Jensen]
    """
    if is_decibel:
        f = Tone(frequency=f.frequency, amplitude=db_to_ampl(f.amplitude))
        g = Tone(frequency=g.frequency, amplitude=db_to_ampl(g.amplitude))
    q = abs(g.frequency - f.frequency) / (0.021 * min(f.frequency, g.frequency) + 19)
    return f.amplitude * g.amplitude * (math.exp(-0.84 * q) - math.exp(-1.38 * q))


def partial_dissonance(
    f_tones: list[Tone],
    g_tones: list[Tone],
    diss_function: callable = dissonance_s,
    is_decibel=False,
) -> float:
    """
    Creates the dissonance_profile of two sets of partials as described in [Sethares] section 10.5.

    Args:
        f_tones (list[Tone]):
        g_tones (list[Tone]:
        diss_function (callable, optional): _description_. Defaults to dissonance_s.

    Returns:
        float: the dissonance value
    """
    f_tones = sorted(f_tones, key=lambda t: t.frequency)
    g_tones = sorted(g_tones, key=lambda t: t.frequency)
    ff_frequencies = [(f, g) for f, g in product(f_tones, f_tones) if f.frequency < g.frequency]
    gg_frequencies = [(f, g) for f, g in product(g_tones, g_tones) if f.frequency < g.frequency]
    fg_frequencies = [(f, g) for f, g in product(f_tones, g_tones)]
    dissonances = (
        sum(diss_function(f, g, is_decibel=is_decibel) for f, g in ff_frequencies),
        sum(diss_function(f, g, is_decibel=is_decibel) for f, g in gg_frequencies),
        sum(diss_function(f, g, is_decibel=is_decibel) for f, g in fg_frequencies),
    )
    frequencies = (ff_frequencies, gg_frequencies, fg_frequencies)
    return [dissonances, frequencies]


def dissonance_profile_old(
    f_partials: list[Partial],
    g_partials: list[Partial],
    amplitude=None,
    is_decibel=False,
    is_reverse=False,
    test_save=False,
):
    base_freq = next(
        (partial.tone.frequency for partial in f_partials if partial.isfundamental), None
    )
    STEPS = 5000
    step = (1.5 * base_freq) / STEPS
    f_partials = sorted(f_partials, key=lambda p: p.tone.frequency)
    g_partials = sorted(g_partials, key=lambda p: p.tone.frequency)
    for p in g_partials:
        p.ratio = p.tone.frequency / g_partials[0].tone.frequency
    f_tones = [
        Tone(frequency=partial.tone.frequency, amplitude=amplitude or partial.tone.amplitude)
        for partial in f_partials
    ]
    frequency_increments = [base_freq + step * i for i in range(STEPS)]
    total_dissonances = []
    ##################
    test_freq_pairs = []
    test_diss_list = []
    ##################
    for freq in frequency_increments:
        g_tones = [
            Tone(frequency=partial.ratio * abs(freq), amplitude=amplitude or partial.tone.amplitude)
            for partial in g_partials
        ]
        dissonances, frequencies = partial_dissonance(
            f_tones, g_tones, is_decibel=is_decibel, diss_function=dissonance_s
        )
        total_dissonances.append((sum(dissonances)))

        ###############
        freq_titles_count = [len(group) for group in frequencies]
        flattened_tones = list(sum(sum(frequencies, []), ()))
        test_freq_pairs.append([t.frequency for t in flattened_tones])
        test_diss_list.append(dissonances)
    if test_save:
        print("Saving...")
        headers = [("f1", "f2"), ("g1", "g2"), ("f", "g")]
        freq_titles = [
            headers[group]
            for group in range(len(freq_titles_count))
            for i in range(freq_titles_count[group])
        ]
        freq_titles = list(sum(freq_titles, ()))
        diss_titles = ["ff", "gg", "fg"]
        diss_df = pd.DataFrame(test_diss_list, columns=diss_titles)
        freq_df = pd.DataFrame(test_freq_pairs, columns=freq_titles)
        pd.concat([diss_df, freq_df], axis=1).to_excel(
            get_path(InstrumentGroupName.SEMAR_PAGULINGAN, Folder.ANALYSES, "old_diss.xlsx")
        )
    ##################
    if is_reverse:
        frequency_increments = list(np.flip(-np.array(frequency_increments)))
        total_dissonances = list(np.flip(np.array(total_dissonances)))

    return (list(np.array(frequency_increments) / base_freq), total_dissonances)


def dissonance_profile(
    f_partials: list[Partial],
    g_partials: list[Partial],
    amplitude=None,
    is_decibel=False,
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
        fg_frequencies,
        fg_amplitudes,
        increment_ids: tuple[int] = (),
    ) -> tuple[np.array, list[list[float]]]:
        if amplitude == None:
            ampl_arrays = [fg_amplitudes[:, 0], fg_amplitudes[:, 1]]
        else:
            # Create amplitude vectors with all same values
            nr_combinations = fg_frequencies.shape[0]
            ampl_arrays = [np.ones(nr_combinations).reshape(1, nr_combinations)] * 2
        freq_arrays = [fg_frequencies[:, 0], fg_frequencies[:, 1]]
        # Add increment rows to requested frequency array(s)
        for i in increment_ids:
            freq_arrays[i] = add_increments(freq_arrays[i], fundamental)
        dissonance_matrix = dissonance_s_array(freq_arrays, ampl_arrays)
        return np.sum(dissonance_matrix, axis=1), freq_arrays

    # Retrieve the frequences and amplitudes of f and g
    f_partials = sorted(f_partials, key=lambda p: p.tone.frequency)
    g_partials = sorted(g_partials, key=lambda p: p.tone.frequency)
    f_freqs = [partial.tone.frequency for partial in f_partials]
    g_freqs = [partial.tone.frequency for partial in g_partials]
    f_ampls = [partial.tone.amplitude for partial in f_partials]
    g_ampls = [partial.tone.amplitude for partial in g_partials]

    # Determine all combinations of f and g partials for which to add up the dissonances.
    fg_combinations = np.array(np.meshgrid(f_freqs, g_freqs)).T.reshape(-1, 2)
    # Same for the internal dissonance of f with itself.
    ff_combinations = np.array(np.meshgrid(f_freqs, f_freqs)).T.reshape(-1, 2)
    ff_combinations = ff_combinations[ff_combinations[:, 0] < ff_combinations[:, 1]]
    # Same for the internal dissonance of g with itself.
    gg_combinations = np.array(np.meshgrid(g_freqs, g_freqs)).T.reshape(-1, 2)
    gg_combinations = gg_combinations[gg_combinations[:, 0] < gg_combinations[:, 1]]

    # Create the x-axis values
    frequencies = (increments.flatten() + fundamental) / fundamental

    # Create a row of partial frequencies for each increment step.
    # The g values are incremented with each step, the f values remain the same.
    dissonance_fg, freqs_fg = dissonances(
        fg_frequencies=fg_combinations,
        fg_amplitudes=None,
        increment_ids=(1,),
    )
    dissonance_ff, freqs_ff = dissonances(
        fg_frequencies=ff_combinations,
        fg_amplitudes=None,
    )
    dissonance_gg, freqs_gg = dissonances(
        fg_frequencies=gg_combinations,
        fg_amplitudes=None,
        increment_ids=(0, 1),
    )

    total_dissonances = dissonance_fg + dissonance_ff + dissonance_gg

    if is_reverse:
        frequencies = np.flip(-frequencies)
        total_dissonances = np.flip(total_dissonances)

    ##############
    if test_save:
        print("Saving...")

        def weave(freqs: list[np.array], tile=(False, False)) -> np.array:
            freqs0 = np.tile(freqs[0], (STEPS, 1)) if tile[0] else freqs[0]
            freqs1 = np.tile(freqs[1], (STEPS, 1)) if tile[1] else freqs[1]
            return np.concatenate(
                sum(
                    [
                        (freqs0[:, i].reshape(5000, -1), freqs1[:, i].reshape(5000, -1))
                        for i in range(freqs[0].shape[-1])
                    ],
                    (),
                ),
                axis=1,
            )

        combi_list = [ff_combinations, gg_combinations, fg_combinations]
        headers = [("f1", "f2"), ("g1", "g2"), ("f", "g")]
        freq_titles = [
            headers[combi] for combi in range(3) for i in range(combi_list[combi].shape[0])
        ]
        freq_titles = list(sum(freq_titles, ()))
        diss_titles = ["ff", "gg", "fg"]
        test_diss = np.column_stack(
            (
                np.tile(dissonance_ff, 5000),
                dissonance_gg,
                dissonance_fg,
            )
        )
        diss_df = pd.DataFrame(test_diss, columns=diss_titles)
        freqs_ff = weave(freqs_ff, tile=(True, True))
        freqs_gg = weave(freqs_gg)
        freqs_fg = weave(freqs_fg, tile=(True, False))
        freqs = np.concatenate([freqs_ff, freqs_gg, freqs_fg], axis=1)
        freq_df = pd.DataFrame(freqs, columns=freq_titles)
        pd.concat([diss_df, freq_df], axis=1).to_excel(
            get_path(InstrumentGroupName.SEMAR_PAGULINGAN, Folder.ANALYSES, "new_diss.xlsx")
        )
    #################
    return (list(frequencies), list(total_dissonances))


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
    # plotter(
    #     group=group,
    #     instrument=instrument,
    #     ratio=ratio,
    #     xmin=0,
    #     xmax=xmax_dict[instrument.instrumenttype],
    #     ymin=-100,
    #     ymax=0,
    #     **kwargs,
    # )
    profiler = dissonance_profile_old
    start_time = time.time()
    results = []
    i = 0
    for pair in pairs:
        i += 1
        print(f"{i=}")
        ph_result = profiler(
            pair[0],
            pair[1],
            is_decibel=kwargs.get("is_decibel", False),
            amplitude=kwargs.get("amplitude", None),
        )
        pp_result = profiler(
            pair[1],
            pair[0],
            is_decibel=kwargs.get("is_decibel", False),
            amplitude=kwargs.get("amplitude", None),
            is_reverse=True,
            test_save=True if pair == pairs[-1] else False,
        )
        results.extend([pp_result, ph_result])
    logger.info(f"total dissonance calculation time: {time.time() - start_time} seconds")
    plot_graphs(
        *results,
        nrows=len(results) // 2,
        ncols=2,
        plottype=PlotType.PLOT,
        show=True,
        autox=True,
        **kwargs,
    )


def plot_dissonance_graph(
    groupname: InstrumentGroupName,
    *,
    instrumenttype: InstrumentType,
    amplitude: float = None,
):
    aggregated = read_object_from_jsonfile(
        AggregatedPartialDict, groupname, Folder.ANALYSES, AGGREGATED_PARTIALS_FILE
    )
    # TODO retrieving dict entry by using string literal is error prone
    keys = [key for key in list(aggregated.root.keys()) if key.startswith(instrumenttype.value)]
    partial_harmonic_pairs = []
    plottitles = []
    for key in keys:
        plottitles.extend([f"octave {key.split('-')[1]}", ""])
        partials = aggregated.root[key]
        fundamental = next((p for p in partials if p.isfundamental), None)
        f_freq = fundamental.tone.frequency
        harmonics = create_partials([(f_freq * i, 1) for i in range(1, 5)])
        partial_harmonic_pairs.append((partials, harmonics))
    pagetitle = f"Dissonance graph for {groupname.value} {instrumenttype.value}" + (
        " (same amplitude for all partials)" if amplitude else ""
    )
    plot_diss_graph(
        partial_harmonic_pairs,
        is_decibel=True,
        amplitude=amplitude,
        pagetitle=pagetitle,
        plottitles=plottitles,
    )


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
    plot_graphs(ph_result, plottype=PlotType.PLOT, show=True, **kwargs)


if __name__ == "__main__":
    plot_dissonance_graph(
        groupname=InstrumentGroupName.SEMAR_PAGULINGAN,
        instrumenttype=InstrumentType.GENDERRAMBAT,
        amplitude=1,
    )

"""
References:
Jensen: https://jjensen.org/DissonanceCurve.html
Völk 2015: Florian Völk 1 2015, Updated analytical expressions for critical bandwidth and critical-band rate

"""

import math
from itertools import product

from tuning.common.classes import AggregatedPartial, AggregatedPartialDict, Tone
from tuning.common.constants import (
    AGGREGATED_PARTIALS_FILE,
    FileType,
    InstrumentGroupName,
)
from tuning.common.utils import db_to_ampl, get_path, read_object_from_jsonfile
from tuning.visualization.utils import PlotType, plot_graphs


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
    Returns the dissonance value between two frequencies, each with given amplitude.
    [Sethares]
    """
    # freq_f, ampl_f = f
    # freq_g, ampl_g = g
    if is_decibel:
        f = Tone(frequency=f.frequency, amplitude=db_to_ampl(f.amplitude))
        g = Tone(frequency=g.frequency, amplitude=db_to_ampl(g.amplitude))
    d = abs(g.frequency - f.frequency)
    b1 = 3.5
    b2 = 5.75
    s = 0.24 / (min(f.frequency, g.frequency) * 0.021 + 19)
    return min(f.amplitude, g.amplitude) * (math.exp(-b1 * s * d) - math.exp(-b2 * s * d))


def dissonance_j(f: Tone, g: Tone, g_shift: float = 0, is_decibel=False) -> float:
    """
    Returns the dissonance value between two frequencies, each with given amplitude.
    [Jensen]
    """
    # freq_f, ampl_f = f
    # freq_g, ampl_g = g
    if is_decibel:
        f = Tone(frequency=f.frequency, amplitude=db_to_ampl(f.amplitude))
        g = Tone(frequency=g.frequency, amplitude=db_to_ampl(g.amplitude))
    q = (g.frequency - f.frequency) / (0.021 * f.frequency + 19)
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
    return sum(diss_function(f, g, is_decibel=is_decibel) for f, g in product(f_tones, g_tones))


def dissonance_profile(
    f_partials: list[AggregatedPartial], g_partials: list[AggregatedPartial], is_decibel=False
):
    base_freq = next(
        (partial.tone.frequency for partial in f_partials if partial.isfundamental), None
    )
    STEPS = 5000
    step = (1.5 * base_freq) / STEPS
    f_tones = [partial.tone for partial in f_partials]
    frequencies = [base_freq + step * i for i in range(STEPS)]
    dissonances = []
    for freq in frequencies:
        g_tones = [
            Tone(frequency=partial.ratio * freq, amplitude=partial.tone.amplitude)
            for partial in g_partials
        ]
        dissonances.append(partial_dissonance(f_tones, g_tones, is_decibel=is_decibel))
    return (frequencies, dissonances)


if __name__ == "__main__":
    GROUPNAME = InstrumentGroupName.SEMAR_PAGULINGAN
    filepath = get_path(GROUPNAME, FileType.ANALYSES, AGGREGATED_PARTIALS_FILE)
    aggregated = read_object_from_jsonfile(AggregatedPartialDict, filepath)
    partials = aggregated.root["gangsa pemade"]
    result = dissonance_profile(partials, partials, is_decibel=True)
    # TODO need to use ratios for dissonance_profiel.
    # Multiply ratio with fundamental of one note (e.g. ding)
    plot_graphs(result, plottype=PlotType.PLOT, show=True)

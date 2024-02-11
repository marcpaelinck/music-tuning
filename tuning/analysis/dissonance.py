"""
References:
Jensen: https://jjensen.org/DissonanceCurve.html
Völk 2015: Florian Völk 1 2015, Updated analytical expressions for critical bandwidth and critical-band rate

"""

import math
from itertools import combinations


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


def db_to_ampl(db: float) -> float:
    """
    converts dB to amplitude
    https://blog.demofox.org/2015/04/14/decibels-db-and-amplitude/
    """
    return pow(10, db / 20)


def ampl_to_loudness(ampl: float) -> float: ...


def dissonance_s(
    f: tuple[float, float], g: tuple[float, float], g_shift: float = 0, is_db=False
) -> float:
    """
    Returns the dissonance value between two frequencies, each with given amplitude.
    [Sethares]
    """
    freq_f, ampl_f = f
    freq_g, ampl_g = g
    if is_db:
        ampl_f = db_to_ampl(ampl_f)
        ampl_g = db_to_ampl(ampl_g)
    d = abs(freq_g - freq_f)
    b1 = 3.5
    b2 = 5.75
    s = 0.24 / (min(freq_f, freq_g) * 0.021 + 19)
    return min(ampl_f, ampl_g) * (math.exp(-b1 * s * d) - math.exp(-b2 * s * d))


def dissonance_j(
    f: tuple[float, float], g: tuple[float, float], g_shift: float = 0, is_decibel=False
) -> float:
    """
    Returns the dissonance value between two frequencies, each with given amplitude.
    [Jensen]
    """
    freq_f, ampl_f = f
    freq_g, ampl_g = g
    if is_decibel:
        ampl_f = db_to_ampl(ampl_f)
        ampl_g = db_to_ampl(ampl_g)
    q = (freq_g - freq_f) / (0.021 * freq_f + 19)
    return ampl_f * ampl_g * (math.exp(-0.84 * q) - math.exp(-1.38 * q))


def dissonance_profile(
    f_partials: list[tuple[float, float]],
    g_partials: list[tuple[float, float]],
    step: float = 1,
    diss_function: callable = dissonance_s,
):
    """
    Creates the dissonance_profile of two sets of partials as described in [Sethares] section 10.5.

    Args:
        f_partials (list[tuple[float, float]]): _description_
        g_partials (list[tuple[float, float]]): _description_
        step (float, optional): _description_. Defaults to 1.
        diss_function (callable, optional): _description_. Defaults to dissonance_s.

    Returns:
        _type_: _description_
    """
    return sum(diss_function(f, g) for f, g in combinations(f_partials, g_partials))


if __name__ == "__main__":
    ...

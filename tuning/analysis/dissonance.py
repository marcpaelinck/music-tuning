"""
References:
Jensen: https://jjensen.org/DissonanceCurve.html
Völk 2015: Florian Völk 1 2015, Updated analytical expressions for critical bandwidth and critical-band rate

"""

import math
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

from tuning.common.utils import plot_separate_graphs


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
    f: tuple[float, float], g: tuple[float, float], g_shift: float = 0, is_db=False
) -> float:
    """
    Returns the dissonance value between two frequencies, each with given amplitude.
    [Jensen]
    """
    freq_f, ampl_f = f
    freq_g, ampl_g = g
    if is_db:
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
    return sum(diss_function(f, g) for f, g in combinations(f_partials, g_partials))


def plot_cbw():
    freq_list = [
        0.008,
        0.016,
        0.032,
        0.064,
        0.128,
        0.256,
        0.512,
        1.024,
        2.048,
        4.096,
        8.192,
        16.384,
    ]
    freq_list = [1000 * f for f in freq_list]
    cbwz = [cbw_z(f) for f in freq_list]
    cbwv = [cbw_v(f) for f in freq_list]
    xpoints = np.array(freq_list)
    ypoints = np.array(cbwz)
    zpoints = np.array(cbwv)
    plt.plot(xpoints, ypoints)
    plt.plot(xpoints, zpoints)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()


def plot_dissonances0():
    g_list = list(range(400, 850, 1))
    f = 400
    diss_j = [dissonance_j(f, g) for g in g_list]
    diss_s = [dissonance_s(f, g) for g in g_list]
    xpoints = np.array(g_list)
    jpoints = np.array(diss_j)
    spoints = np.array(diss_s)
    plt.plot(xpoints, jpoints, label="j", linewidth=7.0)
    plt.plot(xpoints, spoints, label="s", linewidth=1.0)
    plt.xscale("log")
    plt.legend()
    plt.show()


def plot_dissonances():
    g_list = list(range(400, 850, 1))
    f = 400
    diss_j = [dissonance_j(f, g) for g in g_list]
    diss_s = [0.01 * g for g in g_list]
    plot_separate_graphs((g_list, diss_j), (g_list, diss_s))


if __name__ == "__main__":
    # plot_cbw()
    plot_dissonances0()

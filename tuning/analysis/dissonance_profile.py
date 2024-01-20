"""
References:
Jensen: https://jjensen.org/DissonanceCurve.html
Völk 2015: Florian Völk 1 2015, Updated analytical expressions for critical bandwidth and critical-band rate

"""
import math

import matplotlib.pyplot as plt
import numpy as np


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


def dissonance(
    f: float, g: float, ampl_f: float = 1, ampl_g: float = 1, is_db=False
) -> float:
    """
    Returns the dissonance value between two frequencies, each with given amplitude.
    [Jensen]
    """
    if is_db:
        ampl_f = db_to_ampl(ampl_f)
        ampl_g = db_to_ampl(ampl_g)
    q = (g - f) / (0.021 * f + 19)
    return ampl_f * ampl_g * (math.exp(-0.84 * q) - math.exp(-1.38 * q))


def dissonance_profile(
    f_partials: list[tuple[float, float]],
    g_partials: list[tuple[float, float]],
    step: float,
):
    ...


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


def plot_dissonance():
    g_list = list(range(400, 850, 10))
    f = 400
    diss = [dissonance(f, g) for g in g_list]
    xpoints = np.array(g_list)
    ypoints = np.array(diss)
    plt.plot(xpoints, ypoints)
    plt.xscale("log")
    plt.show()


if __name__ == "__main__":
    # plot_cbw()
    plot_dissonance()

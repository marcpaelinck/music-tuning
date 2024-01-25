from xmlrpc.client import MAXINT, MININT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fftpack import fft, fftfreq
from scipy.io import wavfile

from tuning.analysis.constants import FileType, InstrumentGroup
from tuning.analysis.utils import get_path

CUTOFF_FREQUENCY = 10000


def cutoff(xval: list[float], yval: list[float], limits: tuple[float, float]):
    xvalues = list(xval) if not isinstance(xval, list) else xval
    N1, N2 = (
        xvalues.index(next((x for x in xvalues if x > limits[0]), len(xvalues) - 1)),
        xvalues.index(next((x for x in xvalues if x > limits[1]), len(xvalues) - 1)),
    )
    N1 = max(N1 - 1, 0)
    return (xval[N1:N2], yval[N1:N2])


def get_spectrum(group: InstrumentGroup, filename: str, cutoff_limits=(MININT, MAXINT)):
    spectrum_df = pd.read_csv(
        get_path(group, FileType.SPECTRUM, filename),
        sep="\t",
    )
    spectrum = spectrum_df.rename(
        columns={"Frequency (Hz)": "frequency", "Level (dB)": "amplitude"}
    ).to_dict(orient="list")
    return cutoff(
        spectrum["frequency"],
        spectrum["amplitude"],
        (cutoff_limits[0], cutoff_limits[1]),
    )


def create_spectrum(
    filepath: str, cutoff_limits=(MININT, MAXINT)
) -> tuple[list[float]]:
    sample_rate, data = wavfile.read(filepath)
    # take average in case of multiple tracks (stereo)
    # normalize between 0 and 1
    y = [np.int16(np.average(value)) for value in data]
    y_normalized = y / np.max(y)
    # calculate fourier transform (complex numbers list)
    yf = abs(fft(y_normalized))
    xf = fftfreq(len(yf), 1 / sample_rate)
    print(len(yf))
    # reduce scale to [0:1000], discard negative xf values
    return cutoff(xf, yf, cutoff_limits)


def plot_spectra(s1: tuple[list[float]], *args):
    fig, axs = plt.subplots(1 + len(args))
    fig.suptitle("Plots")

    axs[0].plot(s1[0], s1[1])
    for idx in range(len(args)):
        axs[idx + 1].plot(args[idx][0], args[idx][1])
    plt.show()


if __name__ == "__main__":
    CUTOFF1 = 0
    CUTOFF2 = 10000
    group = InstrumentGroup.TEST
    x, y = create_spectrum(
        get_path(group, FileType.SOUND, "ding.wav"),
        cutoff_limits=(CUTOFF1, CUTOFF2),
    )
    yfdb = 20 * np.log10(y / np.max(y))

    spectrum_df = pd.DataFrame({"frequency": x, "amplitude": yfdb})
    spectrum_df.to_csv(
        get_path(group, FileType.SPECTRUM, "spectrum-pemade-penumbang-ding-2.txt"),
        sep="\t",
        index=False,
        float_format="%.5f",
    )

    xgk, ygk = get_spectrum(
        InstrumentGroup.GONG_KEBYAR,
        "spectrum-pemade-penumbang-ding-2.txt",
        cutoff_limits=(CUTOFF1, CUTOFF2),
    )
    plot_spectra((x, y), (x, yfdb), (xgk, ygk))

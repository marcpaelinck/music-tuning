import matplotlib.pyplot as plt

from tuning.analysis.spectrum import import_spectrum
from tuning.common.classes import Instrument
from tuning.common.constants import TRUNCATE_FROM, TRUNCATE_TO, InstrumentGroupName
from tuning.common.utils import get_instrument_by_id, read_group_from_jsonfile


def plot_separate_graphs(s1: tuple[list[float]], *args):
    fig, axs = plt.subplots(1 + len(args))
    fig.suptitle("Plots")

    axs[0].plot(s1[0], s1[1])
    for idx in range(len(args)):
        axs[idx + 1].plot(args[idx][0], args[idx][1])
    plt.show()


def plot_note_spectra(groupname: InstrumentGroupName, instrument_code=str):
    orchestra = read_group_from_jsonfile(groupname)
    instrument = get_instrument_by_id(orchestra, instrument_code)
    spectra = [
        (note.spectrum.frequencies(), note.spectrum.amplitudes()) for note in instrument.notes
    ]
    plot_separate_graphs(*spectra)


if __name__ == "__main__":
    plot_note_spectra(InstrumentGroupName.TEST, "PEM1")

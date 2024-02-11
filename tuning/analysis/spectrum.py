import logging
from xmlrpc.client import MAXINT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fftpack import fft, fftfreq

from tuning.common.classes import (
    Frequency,
    Instrument,
    InstrumentGroup,
    Note,
    Spectrum,
    Tone,
)
from tuning.common.constants import (
    TRUNCATE_FROM,
    TRUNCATE_TO,
    FileType,
    InstrumentGroupName,
)
from tuning.common.utils import get_path

logging.basicConfig(
    format="%(asctime)s - %(name)-12s %(levelname)-7s: %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def truncate_spectrum(spectrum: Spectrum, min_freq: int = 0, max_freq: int = 10000) -> Spectrum:
    """
    truncates a spectrum to the given range
    """
    tones = spectrum.tones
    last_freq = tones[-1]
    index1, index2 = (
        tones.index(next((x for x in tones if x.frequency >= min_freq), last_freq)),
        tones.index(next((x for x in tones if x.frequency > max_freq), last_freq)),
    )
    index2 = max(index2 - 1, 0) if tones[index2].frequency > max_freq else index2
    spectrum.tones = spectrum.tones[index1:index2]
    return spectrum


def import_spectrum(
    groupname: InstrumentGroupName,
    filename: str,
    truncate: tuple[Frequency, Frequency] = (TRUNCATE_FROM, TRUNCATE_TO),
) -> Spectrum:
    """
    Imports a spectrum from a file in the spectrum folder.

    Args:
        groupname (InstrumentGroup): _description_
        filename (str): file name without path information.
        truncate (tuple[Frequency, Frequency]): min and max frequencies to keep.

    Returns:
        Spectrum: truncated spectrum
    """
    spectrum_df = pd.read_csv(
        filepath := get_path(groupname, FileType.SPECTRUM, filename),
        sep="\t",
    )
    spectrum_list = spectrum_df.set_axis(["frequency", "amplitude"], axis=1).to_dict(
        orient="records"
    )

    return truncate_spectrum(
        spectrum=Spectrum(
            spectrumfilepath=filepath,
            tones=[Tone(**record) for record in spectrum_list],
        ),
        min_freq=truncate[0],
        max_freq=truncate[1],
    )


# def save_spectrum(spectrum: Spectrum) -> None:
#     """
#     Saves spectrum data to the spectrum folder.

#     Args:
#         info (NoteInfo): _description_
#         group (InstrumentGroupName): _description_
#     """
#     spectrum_df = pd.DataFrame(
#         {
#             "frequency": spectrum.frequencies(),
#             "amplitude": spectrum.amplitudes(),
#         }
#     )

#     spectrum_df.to_csv(
#         spectrum.spectrumfilepath,
#         sep="\t",
#         index=False,
#         float_format="%.5f",
#     )


# def save_spectrum_summary(
#     group: InstrumentGroupName,
#     orchestra: InstrumentGroup,
#     filename: str = SPECTRA_INFO_FILE,
# ) -> None:
#     logger.info(f"Saving spectrum summary in {filename}.")
#     dict_repr = [
#         {header: instrument.model_dump()[attr] for attr, header in INSTRUMENT_FIELDS.items()}
#         | {header: note.model_dump()[attr] for attr, header in NOTE_FIELDS.items()}
#         | {header: note.octave.model_dump()[attr] for attr, header in OCTAVE_FIELDS.items()}
#         | {header: note.spectrum.model_dump()[attr] for attr, header in SPECTRUM_FIELDS.items()}
#         for instrument in orchestra.instruments
#         for note in instrument.notes
#     ]
#     summary_df = pd.DataFrame.from_dict(dict_repr)
#     summary_df.to_csv(
#         get_path(group=group, filetype=FileType.SPECTRUM, filename=filename), sep="\t"
#     )


def create_spectrum(note_info: Note, filepath: str, truncate=(0, MAXINT)) -> Spectrum:
    """
    Creates a spectrum file for a NoteInfo object. The spectrum analysis is performed
    on a single channel.

    Args:
        note_info_list (NoteInfoList): list of NoteInfo objects that contain spectrum data.
        track (int): track to select.
        truncate (tuple, optional): range of frequencies to keep.

    Returns:
        NoteInfoList: _description_
    """
    # amplitudes = np.float64(pd.DataFrame(note_info.sample.data)[track])
    amplitudes = np.float64(np.concatenate(note_info.sample.data))
    # normalize between 0 and 1
    ampl_normalized = amplitudes / np.max(amplitudes)
    # Calculate fourier transform (returns complex numbers list).
    # Discard the imaginary part.
    ampl = np.abs(fft(ampl_normalized))
    ampl_db = 20 * np.log10(ampl / abs(ampl).max())
    freq = fftfreq(len(ampl), 1 / note_info.sample.sample_rate)
    # Remove negative frequencies.
    # Note that fftfreq generates [0, minfreq, .., maxfreq, -maxfreq, ..., -minfreq]
    max_index = np.argmax(freq)
    spectrum = Spectrum(
        tones=[Tone(amplitude=ampl_db[i], frequency=freq[i]) for i in range(max_index + 1)],
        spectrumfilepath=filepath,
    )
    return truncate_spectrum(spectrum, min_freq=truncate[0], max_freq=truncate[1])


def create_spectrum_audacity(note: Note, filepath: str, truncate=(0, MAXINT)) -> Spectrum:
    """
    Creates a spectrum file for a NoteInfo object. The spectrum analysis is performed
    on a single channel.

    Args:
        note (Note): note object.
        filepath: path to spectrum file.
        truncate (tuple, optional): range of frequencies to keep.

    Returns:
        NoteInfoList: _description_
    """
    # concatenate all channels into one data stream
    amplitudes = np.concatenate(note.sample.data.T)
    # normalize between 0 and 1
    ampl_normalized = amplitudes / np.max(amplitudes)
    # Calculate fourier transform (returns complex numbers list).
    # Discard the imaginary part.
    BINSIZE = 65536
    # cut up into bins
    results = list()
    for bin in range(len(ampl_normalized) // BINSIZE):
        ampl = np.abs(fft(ampl_normalized[bin * BINSIZE : (bin + 1) * BINSIZE]))
        freq = fftfreq(len(ampl), 1 / note.sample.sample_rate)
        # Remove negative frequencies.
        # Remark that fftfreq generates [0, minfreq, .., maxfreq, -maxfreq, ..., -minfreq]
        max_index = np.argmax(freq)
        results.append(ampl[: max_index + 1])
    avg_ampl = np.average(np.asarray(results), axis=0)
    # maxvalue = np.iinfo(np.int16).max
    maxvalue = np.max(avg_ampl)
    ampl_db = 20 * np.log10(avg_ampl / maxvalue)
    spectrum = Spectrum(
        tones=[Tone(amplitude=ampl_db[i], frequency=freq[i]) for i in range(max_index + 1)],
        spectrumfilepath=filepath,
    )
    # spectrum = Spectrum(amplitudes=ampl_db[: max_index + 1], frequencies=freq[: max_index + 1])
    return truncate_spectrum(spectrum, min_freq=truncate[0], max_freq=truncate[1])


def plot_spectra(s1: tuple[list[float]], *args):
    fig, axs = plt.subplots(1 + len(args))
    fig.suptitle("Plots")

    axs[0].plot(s1[0], s1[1])
    for idx in range(len(args)):
        axs[idx + 1].plot(args[idx][0], args[idx][1])
    plt.show()


def get_spectrum_filepath(group: InstrumentGroup, instrument: Instrument, note: Note):
    filename = f"{instrument.name}-{instrument.code}-{note.name}-{note.octave.index}.csv"
    return get_path(group.grouptype, FileType.SPECTRUM, filename)


def create_spectra(orchestra: InstrumentGroup) -> None:
    if not orchestra.has_sound_samples:
        logger.warning("No sound samples available: call get_sound_samples first.")
        return

    for instrument in orchestra.instruments:
        logger.info(f"Processing {instrument.name} {instrument.code}.")
        if instrument.error:
            continue
        logger.info(f"Generating spectrum files.")
        for note in instrument.notes:
            note.spectrum = create_spectrum_audacity(
                note, filepath=get_spectrum_filepath(orchestra, instrument, note)
            )
            # save_spectrum(note.spectrum)
    # save_spectrum_summary(group=orchestra.grouptype, orchestra=orchestra)
    for instrument in orchestra.instruments:
        if instrument.error:
            print(f"{instrument.soundfile}: {instrument.comment}")
    orchestra.has_spectra = True


if __name__ == "__main__":
    group = InstrumentGroupName.TEST
    # x, amplitudes = create_spectrum(
    #     get_path(group, FileType.SOUND, "ding.wav"),
    #     cutoff_limits=(CUTOFF_FREQUENCY_LOW, CUTOFF_FREQUENCY_HIGH),
    # )
    # yfdb = 20 * np.log10(amplitudes / np.max(amplitudes))

    # spectrum_df = pd.DataFrame({"frequency": x, "amplitude": yfdb})
    # spectrum_df.to_csv(
    #     get_path(group, FileType.SPECTRUM, "spectrum-pemade-penumbang-ding-2.txt"),
    #     sep="\t",
    #     index=False,
    #     float_format="%.5f",
    # )

    # plot_spectra((x, amplitudes), (x, yfdb), (spectrum_gk.frequencies, spectrum_gk.amplitudes)

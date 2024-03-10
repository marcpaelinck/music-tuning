from xmlrpc.client import MAXINT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fftpack import fft, fftfreq

from tuning.common.classes import (
    AmplUnit,
    Frequency,
    FreqUnit,
    Instrument,
    InstrumentGroup,
    Note,
    Spectrum,
    Tone,
)
from tuning.common.constants import (
    TRUNCATE_FROM,
    TRUNCATE_TO,
    Folder,
    InstrumentGroupName,
)
from tuning.common.utils import (
    get_logger,
    get_path,
    read_group_from_jsonfile,
    save_group_to_jsonfile,
)

logger = get_logger(__name__)


def truncate_spectrum(spectrum: Spectrum, min_freq: int = 0, max_freq: int = 10000) -> Spectrum:
    """
    truncates a spectrum to the given range
    Note: argmax will stop at the first True ("In case of multiple occurrences of the maximum values, 
          the indices corresponding to the first occurrence are returned.") and doesn't save another list. 
    """
    index1 = np.argmax(spectrum.frequencies >= min_freq)
    index2 = -1 if spectrum.frequencies[-1] < max_freq else np.argmax(spectrum.frequencies >= max_freq)
    spectrum.frequencies = spectrum.frequencies[index1:index2]
    spectrum.amplitudes = spectrum.amplitudes[index1:index2]
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
        filepath := get_path(groupname, Folder.SPECTRUM, filename),
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
        frequencies=freq,
        amplitudes=ampl_db,
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
    maxvalue = np.max(avg_ampl)
    ampl_db = 20 * np.log10(avg_ampl / maxvalue)
    spectrum = Spectrum(
        frequencies=freq[: max_index + 1],
        amplitudes=ampl_db,
        spectrumfilepath=filepath,
        freq_unit=FreqUnit.HERTZ,
        ampl_unit=AmplUnit.DB,
    )
    return truncate_spectrum(spectrum, min_freq=truncate[0], max_freq=truncate[1])


def get_spectrum_filepath(group: InstrumentGroup, instrument: Instrument, note: Note) -> str:
    """
    Returns the path to the file containing the frequency spectrum of the given note.

    Args:
        group (InstrumentGroup): the instrument group
        instrument (Instrument): the instrument within the group
        note (Note): the note of the instrument

    Returns:
        str: relative path to the spectrum file.
    """
    filename = f"{instrument.instrumenttype.value.replace(" ","")}-{instrument.code}-{note.name}-{note.octave.index}.csv"
    return get_path(group.grouptype, Folder.SPECTRUM, filename)


def create_spectra(orchestra: InstrumentGroup) -> None:
    """
    This method creates Spectrum objects which contain the frequency spectrum 
    of the individual notes and adds each Spectrum object to the corresponding Note object. 

    Args:
        orchestra (InstrumentGroup): The instrument group.
    """
    if not orchestra.has_sound_samples:
        logger.warning("No sound samples available: call get_sound_samples first.")
        return

    for instrument in orchestra.instruments:
        logger.info(f"Processing {instrument.instrumenttype} {instrument.code}.")
        if instrument.error:
            continue
        logger.info(f"Generating spectrum files.")
        for note in instrument.notes:
            note.spectrum = create_spectrum_audacity(
                note, filepath=get_spectrum_filepath(orchestra, instrument, note)
            )
            # save_spectrum(note.spectrum)
    for instrument in orchestra.instruments:
        if instrument.error:
            print(f"{instrument.soundfile}: {instrument.comment}")
    orchestra.has_spectra = True


if __name__ == "__main__":
    groupname = InstrumentGroupName.SEMAR_PAGULINGAN
    orchestra = read_group_from_jsonfile(groupname, read_sounddata=True)
    create_spectra(orchestra)
    save_group_to_jsonfile(orchestra)

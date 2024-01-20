import os

from constants import DATA_FOLDER, SOUNDFILE_SUBFOLDER, SPECTRUM_SUBFOLDER


def get_spectrum_folder(instrument_group_folder: str):
    return os.path.join(DATA_FOLDER, instrument_group_folder, SPECTRUM_SUBFOLDER)


def get_soundwave_folder(instrument_group_folder: str):
    return os.path.join(DATA_FOLDER, instrument_group_folder, SOUNDFILE_SUBFOLDER)


def get_output_folder(instrument_group_folder: str):
    return os.path.join(DATA_FOLDER, instrument_group_folder)

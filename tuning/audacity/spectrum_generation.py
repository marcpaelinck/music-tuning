import os
from time import sleep

import pyautogui
from audacity_interface import do_command

from tuning.analysis.constants import DATA_FOLDER, SOUNDFILE_SUBFOLDER, TEST
from tuning.analysis.utils import get_filenames


def quick_test():
    """Example list of commands."""
    do_command("PlotSpectrum:")
    # do_command("Help: Command=Help")
    # do_command('Help: Command="GetInfo"')
    # do_command('SetPreference: Name=GUI/Theme Value=classic Reload=1')


def select_256():
    # with pyautogui.hold("alt"):
    #     pyautogui.press("tab")
    with pyautogui.hold("alt"):
        pyautogui.press("s")
    pyautogui.press("8")


def create_spectrum_files(instrument_group: str) -> None:
    path = os.path.join(DATA_FOLDER, instrument_group, SOUNDFILE_SUBFOLDER)
    path = os.path.abspath(path)
    filenames = get_filenames(path, "^[\w-]+.(wav|mp3)$")
    for filename in filenames:
        filepath = os.path.join(path, filename)
        do_command(f"OpenProject2: Filename={filepath}")
        do_command(f"SelectAll:")
        do_command(f"Close:")
        sleep(2)
        pyautogui.press("n")


create_spectrum_files(TEST)

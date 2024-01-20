import pyautogui
from audacity_interface import do_command


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


quick_test()
select_256()

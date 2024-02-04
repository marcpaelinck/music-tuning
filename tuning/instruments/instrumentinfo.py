from collections import defaultdict

import pandas as pd

from tuning.common.classes import Instrument, InstrumentGroup, Note, Octave, OmbakType
from tuning.common.constants import INSTRUMENT_INFO_FILE, FileType, InstrumentGroupName
from tuning.common.utils import get_path, note_from_shortcode


def get_octave_dict(group: InstrumentGroupName) -> dict[Octave]:
    row_list = pd.read_excel(
        get_path(group, FileType.SETTINGS, INSTRUMENT_INFO_FILE), sheet_name="Octaves"
    ).to_dict(orient="records")
    octave_collection: dict = defaultdict(dict)
    for row in row_list:
        octave_collection[row["octave group"]][row["octave seq"]] = Octave(
            index=row["octave seq"],
            start_freq=row["start freq"],
            end_freq=row["end freq"],
        )
    return octave_collection


def create_group(group: InstrumentGroupName, only_included=False) -> InstrumentGroup:
    """
    Parses the excel document containing information about the instruments files.
    """
    orchestra = InstrumentGroup(grouptype=group, instruments=[])
    octave_dict = get_octave_dict(group)
    fileinfo = pd.read_excel(
        get_path(group, FileType.SETTINGS, INSTRUMENT_INFO_FILE), sheet_name="Sound Files"
    ).to_dict(orient="records")
    if only_included:
        fileinfo = [instr for instr in fileinfo if instr["include"]]

    for row in fileinfo:
        shortcodes = row["notes"].split("-")
        octaves = octave_dict[row["octave group"]]
        instrument_info = Instrument(
            name=row["instrument"],
            code=row["code"],
            ombaktype=OmbakType(row["ombaktype"]),
            soundfile=row["filename"],
            notes=[
                Note(
                    name=note_from_shortcode(shortcode),
                    octave=octaves[int(shortcode[-1])],
                    order_in_soundfile=shortcodes.index(shortcode),
                    freq=0,
                    partial_index=0,
                    partials=[],
                )
                for shortcode in shortcodes
            ],
        )
        orchestra.instruments.append(instrument_info)

    return orchestra

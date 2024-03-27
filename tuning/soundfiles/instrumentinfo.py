from collections import defaultdict

import pandas as pd

from tuning.common.classes import (
    Instrument,
    InstrumentGroup,
    InstrumentType,
    Note,
    Octave,
    OmbakType,
)
from tuning.common.constants import (
    CODE,
    END_FREQ,
    FILENAME,
    INSTRUMENT,
    INSTRUMENT_GROUP,
    INSTRUMENT_INFO_FILE,
    NOTES,
    OCTAVE_SEQ,
    OMBAKTYPE,
    START_FREQ,
    Folder,
    InstrumentGroupName,
)
from tuning.common.utils import get_path, note_from_shortcode


def get_octave_dict(groupname: InstrumentGroupName) -> dict[Octave]:
    row_list = pd.read_excel(
        get_path(groupname, Folder.SETTINGS, INSTRUMENT_INFO_FILE), sheet_name="Octaves"
    ).to_dict(orient="records")
    octave_collection: dict = defaultdict(dict)
    for row in row_list:
        octave_collection[row[INSTRUMENT_GROUP]][row[OCTAVE_SEQ]] = Octave(
            index=row[OCTAVE_SEQ],
            start_freq=row[START_FREQ],
            end_freq=row[END_FREQ],
        )
    return octave_collection


def create_group_from_info_file(groupname: InstrumentGroupName) -> InstrumentGroup:
    """
    Parses the excel document containing information about the instruments files.
    """
    orchestra = InstrumentGroup(grouptype=groupname, instruments=[])
    octave_dict = get_octave_dict(groupname)
    fileinfo = pd.read_excel(
        get_path(groupname, Folder.SETTINGS, INSTRUMENT_INFO_FILE), sheet_name="Sound Files"
    ).to_dict(orient="records")

    for row in fileinfo:
        shortcodes = row[NOTES].split("-")
        octaves = octave_dict[row[INSTRUMENT_GROUP]]
        instrument_info = Instrument(
            instrumenttype=InstrumentType(row[INSTRUMENT]),
            code=row[CODE],
            ombaktype=OmbakType(row[OMBAKTYPE]),
            original_soundfilename=row[FILENAME],
            soundfilename=row[FILENAME],
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

from collections import defaultdict
from tokenize import group

import pandas as pd

from tuning.common.classes import (
    Frequency,
    Instrument,
    InstrumentGroup,
    InstrumentType,
    Note,
    NoteName,
    Octave,
    OctaveNbr,
    OmbakType,
)
from tuning.common.constants import (
    CODE,
    END_FREQ,
    FILENAME,
    FREQUENCY,
    INSTRUMENT,
    INSTRUMENT_GROUP,
    INSTRUMENT_INFO_FILE,
    NOTE,
    NOTES,
    OCTAVE_SEQ,
    OMBAKTYPE,
    START_FREQ,
    Folder,
    InstrumentGroupName,
)
from tuning.common.utils import get_path, note_from_shortcode


def get_octave_dict(groupname: InstrumentGroupName) -> dict[int, Octave]:
    oct_df = pd.read_excel(get_path(groupname, Folder.SETTINGS, INSTRUMENT_INFO_FILE), sheet_name="Octaves")
    row_list = (
        oct_df[oct_df[INSTRUMENT_GROUP] == groupname.value].drop(INSTRUMENT_GROUP, axis=1).to_dict(orient="records")
    )
    octave_dict = dict()
    for row in row_list:
        octave_dict[row[OCTAVE_SEQ]] = Octave(
            index=row[OCTAVE_SEQ],
            start_frequency=row[START_FREQ],
            end_frequency=row[END_FREQ],
        )
    return octave_dict


def get_notefreq_dict(groupname: InstrumentGroupName) -> dict[Frequency, (NoteName, OctaveNbr)]:
    note_df = pd.read_excel(get_path(groupname, Folder.SETTINGS, INSTRUMENT_INFO_FILE), sheet_name="Notes")
    row_list = (
        note_df[note_df[INSTRUMENT_GROUP] == groupname.value].drop(INSTRUMENT_GROUP, axis=1).to_dict(orient="records")
    )
    note_dict = dict()
    for row in row_list:
        note_dict[row[FREQUENCY]] = (NoteName(row[NOTE]), row[OCTAVE_SEQ])
    return note_dict


def step_create_group_from_info_file(groupname: InstrumentGroupName) -> InstrumentGroup:
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
        instrument_info = Instrument(
            instrumenttype=InstrumentType(row[INSTRUMENT]),
            code=row[CODE],
            ombaktype=OmbakType(row[OMBAKTYPE]),
            original_soundfilename=row[FILENAME],
            soundfilename=row[FILENAME],
            notes=[
                Note(
                    name=note_from_shortcode(shortcode),
                    octave=octave_dict[int(shortcode[-1])],
                    order_in_soundfile=shortcodes.index(shortcode),
                    frequency=0,
                    fundamental_index=0,
                    partials=[],
                )
                for shortcode in shortcodes
            ],
        )
        orchestra.instruments.append(instrument_info)

    return orchestra

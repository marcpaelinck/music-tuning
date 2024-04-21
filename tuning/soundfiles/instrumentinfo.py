from collections import defaultdict
from tokenize import group

import pandas as pd

from tuning.common.classes import (
    ClipInfo,
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
    INSTRUMENT_INFO_FILE,
    Folder,
    InfoFile,
    InstrumentGroupName,
)
from tuning.common.utils import get_logger, get_path, note_from_shortcode

logger = get_logger(__name__)


def get_octave_dict(groupname: InstrumentGroupName) -> dict[int, Octave]:
    oct_df = pd.read_excel(get_path(groupname, Folder.SETTINGS, INSTRUMENT_INFO_FILE), sheet_name=InfoFile.OCTAVE_SHEET)
    row_list = (
        oct_df[oct_df[InfoFile.INSTRUMENT_GROUP] == groupname.value]
        .drop(InfoFile.INSTRUMENT_GROUP, axis=1)
        .to_dict(orient="records")
    )
    octave_dict = dict()
    for row in row_list:
        octave_dict[row[InfoFile.OCTAVE_SEQ]] = Octave(
            index=row[InfoFile.OCTAVE_SEQ],
            start_frequency=row[InfoFile.START_FREQ],
            end_frequency=row[InfoFile.END_FREQ],
        )
    return octave_dict


def get_clipping_dict(groupname: InstrumentGroupName) -> dict[tuple[str, int], tuple[NoteName, OctaveNbr]]:
    clipping_df = pd.read_excel(
        get_path(groupname, Folder.SETTINGS, INSTRUMENT_INFO_FILE), sheet_name=InfoFile.CLIPPING_SHEET
    )
    row_list = (
        clipping_df[clipping_df[InfoFile.INSTRUMENT_GROUP] == groupname.value]
        .drop(InfoFile.INSTRUMENT_GROUP, axis=1)
        .to_dict(orient="records")
    )
    clipping_dict: ClipInfo = defaultdict(dict)
    for row in row_list:
        clipping_dict[row[InfoFile.FILENAME]][row[InfoFile.CLIPPING_SEQ] - 1] = (
            NoteName(row[InfoFile.NOTE]),
            row[InfoFile.OCTAVE_SEQ],
        )
    return clipping_dict


def get_notefreq_dict(groupname: InstrumentGroupName) -> dict[InstrumentType, dict[(NoteName, OctaveNbr), Frequency]]:
    note_df = pd.read_excel(get_path(groupname, Folder.SETTINGS, INSTRUMENT_INFO_FILE), sheet_name=InfoFile.NOTE_SHEET)
    row_list = (
        note_df[note_df[InfoFile.INSTRUMENT_GROUP] == groupname.value]
        .drop(InfoFile.INSTRUMENT_GROUP, axis=1)
        .to_dict(orient="records")
    )
    note_dict = defaultdict(dict)
    for row in row_list:
        instrtype_list = [InstrumentType(instr.strip()) for instr in row[InfoFile.INSTRUMENTS].split(",")]
        for instr_type in instrtype_list:
            note_dict[instr_type][(NoteName(row[InfoFile.NOTE]), row[InfoFile.OCTAVE_SEQ])] = row[InfoFile.FREQUENCY]
    return note_dict


def step_create_group_from_info_file(groupname: InstrumentGroupName) -> InstrumentGroup:
    """
    Parses the excel document containing information about the instruments files.
    """
    logger.info(f"Creating a data structure from file {INSTRUMENT_INFO_FILE}")

    octave_dict = get_octave_dict(groupname)
    clipping_dict = get_clipping_dict(groupname)
    note_dict = get_notefreq_dict(groupname)

    orchestra = InstrumentGroup(grouptype=groupname, instruments=[], clipinfo_dict=clipping_dict)
    fileinfo = pd.read_excel(
        get_path(groupname, Folder.SETTINGS, INSTRUMENT_INFO_FILE), sheet_name=InfoFile.SOUNDFILE_SHEET
    ).to_dict(orient="records")

    for row in fileinfo:
        instrument_info = Instrument(
            instrumenttype=(instr_type := InstrumentType(row[InfoFile.INSTRUMENT])),
            code=row[InfoFile.CODE],
            ombaktype=OmbakType(row[InfoFile.OMBAKTYPE]),
            original_soundfilename=row[InfoFile.FILENAME],
            soundfilename=row[InfoFile.FILENAME],
            notes=[
                Note(
                    name=name,
                    octave=octave_dict[oct_nbr],
                    frequency=freq,
                    fundamental_index=0,
                    partials=[],
                )
                for (name, oct_nbr), freq in note_dict[instr_type].items()
            ],
        )
        orchestra.instruments.append(instrument_info)

    return orchestra

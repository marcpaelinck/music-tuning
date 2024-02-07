import json
import os
import re
from ast import parse

from matplotlib import pyplot as plt
from pydantic import BaseModel

from tuning.common.classes import Instrument, InstrumentGroup, Note, NoteName, Octave
from tuning.common.constants import (
    DATA_FOLDER,
    OCTAVE_RANGE_FILE,
    FileType,
    InstrumentGroupName,
)


def get_path(group: InstrumentGroupName, filetype: FileType, filename: str = ""):
    return os.path.join(DATA_FOLDER, group.value, filetype.value, filename)


def get_filenames(folder: str, regex: str = ".*") -> list[str]:
    return [
        f
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and re.match(regex, f)
    ]


def parse_json_file(pydantic_type: type[BaseModel], filepath: str) -> BaseModel:
    # Parses a json file into a Pydantic object
    with open(filepath, "r") as infile:
        json_data = json.load(infile)
        if isinstance(json_data, list):
            return pydantic_type(json_data)
        else:
            return pydantic_type(**json_data)


def note_from_shortcode(code: str) -> NoteName:
    """
    Returns a Note object for a given short code "i", "o", "e" etc.
    Also recognizes the intermediate notes "eu" and "ai".
    """
    code = code[:-1] if code[-1] in "1234567890" else code
    return next((note for note in NoteName if note.value == f"d{code}ng"), None)


def save_group_to_jsonfile(group: InstrumentGroup):
    with open(
        get_path(group.grouptype, FileType.SETTINGS, f"{group.grouptype.value}.json"), "w"
    ) as outfile:
        outfile.write(group.model_dump_json(indent=4))


def read_group_from_jsonfile(
    groupname: InstrumentGroupName,
    read_sounddata: bool = False,
    read_spectrumdata: bool = True,
    save_spectrumdata: bool = True,
):
    with open(get_path(groupname, FileType.SETTINGS, f"{groupname.value}.json"), "r") as infile:
        jsonvalue = infile.read()
    return InstrumentGroup.model_validate_json(
        jsonvalue,
        context={
            "read_sounddata": read_sounddata,
            "read_spectrumdata": read_spectrumdata,
            "save_spectrumdata": save_spectrumdata,
        },
    )


def get_instrument_by_id(group: InstrumentGroup, code: str) -> Instrument:
    return next((instr for instr in group.instruments if instr.code == code), None)

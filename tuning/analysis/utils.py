import json
import os
import re
from ast import parse

import pandas as pd
from pydantic import BaseModel, InstanceOf

from tuning.analysis.classes import NoteInfo, Octave
from tuning.analysis.constants import DATA_FOLDER, FileType, InstrumentGroup


def get_path(group: InstrumentGroup, filetype: FileType, filename: str = ""):
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


def get_octaves(group: InstrumentGroup) -> list[Octave]:
    with open(get_path(group, FileType.OUTPUT, "octaves_ranges.json"), "r") as octfile:
        octave_info = json.load(octfile)
        return {
            key: Octave(sequence=key, start_freq=start, end_freq=end)
            for key, [start, end] in octave_info.items()
        }

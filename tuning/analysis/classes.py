import json
import os
from enum import Enum
from typing import List

from pydantic import BaseModel, RootModel

Frequency = float
Ratio = float
Amplitude = float
Interval = float


class PartialType(Enum):
    BASE_NOTE = "base note"
    PARTIAL = "partial"


class Octave(BaseModel):
    sequence: int
    start_freq: float
    end_freq: float


class Tone(BaseModel):
    frequency: float
    amplitude: float


class Partial(BaseModel):
    tone: Tone
    ratio: Ratio
    reduced_ratio: Ratio
    type: PartialType


PartialList = RootModel[List[Partial]]


class AggregatedPartial(BaseModel):
    ratio: Ratio
    ampl: Amplitude
    partials: PartialList


AggregatedPartialList = RootModel[List[AggregatedPartial]]


class SpectrumInfo(BaseModel):
    instrument: str
    tuning: str
    note: str
    octave: Octave


class Note(BaseModel):
    name: str
    spectrum: SpectrumInfo
    freq: Frequency
    index: int
    partials: PartialList


class NoteList(BaseModel):
    group: str
    comment: str
    notes: List[Note]


def parse_file(pydantic_type: type, folder: str, filename: str):
    with open(os.path.join(folder, filename), "r") as f:
        json_data = json.load(f)
        if isinstance(json_data, List):
            return pydantic_type(json_data)
        else:
            return pydantic_type(**json_data)


if __name__ == "__main__":
    ...

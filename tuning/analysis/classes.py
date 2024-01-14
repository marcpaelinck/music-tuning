import json
import os
from enum import Enum
from typing import List

from pydantic import BaseModel, RootModel

Frequency = float
Ratio = float
Amplitude = float
Interval = float


class HarmonicType(Enum):
    BASE_NOTE = "base note"
    HARMONIC = "harmonic"


class Octave(BaseModel):
    sequence: int
    start_freq: float
    end_freq: float


class Tone(BaseModel):
    frequency: float
    amplitude: float


class Harmonic(BaseModel):
    tone: Tone
    ratio: Ratio
    reduced_ratio: Ratio
    type: HarmonicType


HarmonicList = RootModel[List[Harmonic]]


class AggregatedHarmonic(BaseModel):
    ratio: Ratio
    ampl: Amplitude
    harmonics: HarmonicList


AggregatedHarmonicList = RootModel[List[AggregatedHarmonic]]


class SpectrumInfo(BaseModel):
    instrument: str
    tuning: str
    note: str
    octave: Octave


class Note(BaseModel):
    spectrum: SpectrumInfo
    freq: Frequency
    index: int
    harmonics: HarmonicList


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

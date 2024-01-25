from enum import Enum
from typing import Optional

from pydantic import BaseModel, RootModel

Frequency = float
Ratio = float
Amplitude = float
Interval = float
Filename = str


class PartialType(Enum):
    FUNDAMENTAL = "fundamental"
    PARTIAL = "partial"


class OmbakType(Enum):
    PENUMBANG = "penumbang"
    PENGISEP = "pengisep"


class Note(Enum):
    DING = "ding"
    DONG = "dong"
    DENG = "deng"
    DEUNG = "deung"
    DUNG = "dung"
    DANG = "dang"
    DAING = "daing"

    def __init__(self, val):
        super().__init__(val)
        self.index = ["ding", "dong", "deng", "deung", "dung", "dang", "daing"].index(
            val
        )

    def __lt__(self, obj):
        return (self.index) < (obj.index)

    def __gt__(self, obj):
        return (self.index) > (obj.index)

    def __le__(self, obj):
        return (self.index) <= (obj.index)

    def __ge__(self, obj):
        return (self.index) >= (obj.index)

    def __eq__(self, obj):
        return (self.index) == (obj.index)


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


PartialList = RootModel[list[Partial]]


class AggregatedPartial(BaseModel):
    ratio: Ratio
    ampl: Amplitude
    partials: PartialList


AggregatedPartialList = RootModel[list[AggregatedPartial]]


class SpectrumInfo(BaseModel):
    instrument: str
    ombaktype: OmbakType
    note: Note
    octave: Octave


WavePattern = RootModel[list[Amplitude]]


class NoteInfo(BaseModel):
    note: Note
    instrument: str
    ombaktype: OmbakType
    octave: Octave
    freq: Optional[Frequency] = None
    index: int
    wav: Optional[WavePattern] = []
    partials: Optional[PartialList] = []


class NoteInfoList(BaseModel):
    group: str
    comment: str
    notes: list[NoteInfo]


class SoundFileInfo(BaseModel):
    filename: str


if __name__ == "__main__":
    ...

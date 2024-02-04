import json
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_serializer, model_validator
from scipy.io import wavfile

from tuning.common.constants import InstrumentGroupName

Frequency = float
Ratio = float
Amplitude = float
Interval = float
Filename = str
SoundData = np.ndarray[np.ndarray[np.int16]]


class OmbakType(Enum):
    PENUMBANG = "penumbang"
    PENGISEP = "pengisep"

    def __str__(self):
        return self.value


class NoteName(Enum):
    DING = "ding"
    DONG = "dong"
    DENG = "deng"
    DEUNG = "deung"
    DUNG = "dung"
    DANG = "dang"
    DAING = "daing"

    def __str__(self):
        return self.value


class Octave(BaseModel):
    index: int
    start_freq: float
    end_freq: float


class Tone(BaseModel):
    frequency: float
    amplitude: float


class ClipRange(BaseModel):
    start: int
    end: int


class Sample(BaseModel):
    sample_rate: int
    data: SoundData = Field(default=None, exclude=True)
    soundfilepath: str
    cliprange: ClipRange

    @model_validator(mode="after")
    # Reads the sound data from file.
    # The sample data is not saved when serializing this class because this would
    # make the json file too bulky. When the class is loaded, the data is read
    # directly from the sound file using the soundfilepath and cliprange values.
    def read_sounddata(self):
        if self.soundfilepath and self.cliprange is not None and self.data is None:
            _, data = wavfile.read(self.soundfilepath)
            self.data = data[self.cliprange.start : self.cliprange.end + 1]
        return self

    class Config:
        arbitrary_types_allowed = True


class Partial(BaseModel):
    tone: Tone
    ratio: Ratio
    isfundamental: bool


# class AggregatedPartial(BaseModel):
#     ratio: Ratio
#     ampl: Amplitude
#     partials: PartialList
#
#
# AggregatedPartialList = RootModel[list[AggregatedPartial]]


class Spectrum(BaseModel):
    spectrumfilepath: str = None
    tones: list[Tone] = Field(default=None)

    @model_validator(mode="after")
    # Reads the spectrum data from file.
    # The spectrum data is not saved when serializing this class because this would
    # make the json file too bulky. When the class is loaded, the data is read
    # directly from the sound file using the soundfilepath and cliprange values.
    def read_spectrumdata(self):
        if self.spectrumfilepath and self.tones == None:
            spectrum_df = pd.read_csv(self.spectrumfilepath, sep="\t")
            spectrum_list = spectrum_df.set_axis(["frequency", "amplitude"], axis=1).to_dict(
                orient="records"
            )
            self.tones = [Tone(**record) for record in spectrum_list]
        return self

    @field_serializer("tones")
    def serialize_tones(self, data: list[Tone], _info):
        spectrum_df = pd.DataFrame(
            {
                "frequency": self.frequencies(),
                "amplitude": self.amplitudes(),
            }
        )
        spectrum_df.to_csv(
            self.spectrumfilepath,
            sep="\t",
            index=False,
            float_format="%.5f",
        )
        return None

    def frequencies(self):
        return [tone.frequency for tone in self.tones]

    def amplitudes(self):
        return [tone.amplitude for tone in self.tones]


class Note(BaseModel):
    name: NoteName
    octave: Octave
    order_in_soundfile: int
    freq: Optional[Frequency] = None
    partial_index: int = None
    sample: Optional[Sample] = None
    spectrum: Optional[Spectrum] = None
    partials: Optional[list[Partial]] = None


class Instrument(BaseModel):
    error: bool = False
    comment: str = ""
    name: str
    code: str
    ombaktype: OmbakType
    soundfile: str = ""
    notes: list[Note]


class InstrumentGroup(BaseModel):
    grouptype: InstrumentGroupName
    instruments: list[Instrument]
    has_sound_samples: bool = False
    has_spectra: bool = False


if __name__ == "__main__":
    x = 1

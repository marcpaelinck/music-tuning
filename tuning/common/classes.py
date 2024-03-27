from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    Field,
    RootModel,
    ValidationInfo,
    field_serializer,
    model_validator,
)
from scipy.io import wavfile

from tuning.common.constants import InstrumentGroupName

Frequency = float
Ratio = float
Amplitude = float
Interval = float
Filename = str
SoundData = np.ndarray[np.ndarray[np.int16]]


class InstrumentType(Enum):
    GONG = "gong"
    JEGOGAN = "jegogan"
    JUBLAG = "jublag"
    PEMADE = "gangsa pemade"
    KANTILAN = "gangsa kantilan"
    GENDERRAMBAT = "gender rambat"
    GENDERWAYANG = "gender wayang"


class OmbakType(Enum):
    PENGUMBANG = "pengumbang"
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


class FreqUnit(Enum):
    HERTZ = "Herz"
    CENT = "Cent"
    RATIO = "ratio"


class AmplUnit(Enum):
    LINEAR = "linear"
    DB = "dB"


class Octave(BaseModel, frozen=True):
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
    soundfilepath: Optional[str] = None
    cliprange: ClipRange

    @model_validator(mode="after")
    # Reads the sound data from file.
    # The sample data is not saved when serializing this class to avoid cluttering
    # the JSON serialization file. During deserialization the data is read from the
    # sound file using the cliprange values to determine the sample range.
    def read_sounddata(self, info: ValidationInfo):
        if info.context and info.context.get("read_sounddata", True):
            if self.soundfilepath and self.cliprange is not None and self.data is None:
                _, data = wavfile.read(self.soundfilepath)
                self.data = data[self.cliprange.start : self.cliprange.end + 1]
        return self

    class Config:
        arbitrary_types_allowed = True


class Partial(BaseModel):
    tone: Tone
    ratio: Ratio
    prominence: float = 0
    isfundamental: bool


class AggregatedPartial(Partial):
    instrumenttype: InstrumentType = None
    ombaktype: OmbakType = None
    octave: Octave = None
    partials: Optional[list[Partial]] = Field(default=None)


AggregatedPartialDict = RootModel[dict[str, list[AggregatedPartial]]]


class Spectrum(BaseModel):
    spectrumfilepath: str | None = None
    freq_unit: FreqUnit = FreqUnit.HERTZ
    ampl_unit: AmplUnit = AmplUnit.DB
    frequencies: Optional[np.ndarray[float]] = Field(default=None)
    reference_amplitude: float = None
    amplitudes: Optional[np.ndarray[float]] = Field(default=None)
    save_spectrum_data: bool = Field(exclude=True, default=False)

    @model_validator(mode="after")
    # Reads the spectrum data from file.
    # During deserialization the tones attribute is read from the spectrum file.
    def read_spectrumdata(self, info: ValidationInfo):
        # Skip if tones attribute already has a value
        if info.context and info.context.get("read_spectrumdata", True):
            if self.spectrumfilepath and self.frequencies == None:
                spectrum_df = pd.read_csv(self.spectrumfilepath, sep="\t")
                self.frequencies = np.array(spectrum_df["frequency"])
                self.amplitudes = np.array(spectrum_df["amplitude"])
        # if info.context and "save_spectrumdata" in info.context.keys():
        #     self.save_spectrum_data = info.context["save_spectrumdata"]
        return self

    @field_serializer("frequencies", "amplitudes")
    # when serializing this class the spectrum data is saved to a separate file
    # to avoid cluttering the json serialization file.
    def serialize_tones(self, data: np.ndarray[float], _info):
        if (
            _info.field_name == "frequencies" and self.save_spectrum_data
        ):  # _info.context.get("save_spectrumdata", True):
            # print(f"saving spectrum {self.spectrumfilepath}")
            spectrum_df = pd.DataFrame(
                {
                    "frequency": self.frequencies,
                    "amplitude": self.amplitudes,
                }
            )
            spectrum_df.to_csv(
                self.spectrumfilepath,
                sep="\t",
                index=False,
                float_format="%.5f",
            )
        return None

    class Config:
        arbitrary_types_allowed = True


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
    instrumenttype: InstrumentType = None
    error: bool = False
    comment: str = ""
    # name: str
    code: str
    ombaktype: OmbakType
    original_soundfilename: str
    soundfilename: str
    notes: list[Note]


class InstrumentGroup(BaseModel):
    grouptype: InstrumentGroupName
    instruments: list[Instrument]
    has_sound_samples: bool = False
    has_spectra: bool = False


if __name__ == "__main__":
    x = 1

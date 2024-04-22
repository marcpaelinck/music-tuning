from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationInfo, field_serializer, model_validator
from scipy.io import wavfile

from tuning.common.constants import (
    AmplUnit,
    FreqUnit,
    InstrumentGroupName,
    InstrumentType,
    NoteName,
    OmbakType,
    PartialCombiType,
)

Frequency = float
IntFrequency = int
OctaveNbr = int
Ratio = float
Amplitude = float
Interval = float
Filename = str
SoundData = np.ndarray[np.ndarray[np.int16]]


class Octave(BaseModel, frozen=True):
    index: int
    start_frequency: float
    end_frequency: float

    def __lt__(self, other):
        return self.index < other.index


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
    fundamental: Optional[Tone] = None
    fund_octave: Optional[Octave] = None
    instrument_code: Optional[str] = None
    notename: Optional[NoteName] = None
    octave_idx: Optional[int] = None
    sample_seq: Optional[int] = None

    def __str__(self):
        return f"Partial[frequency={self.tone.frequency:.4}, amplitude={self.tone.amplitude:.4}, ratio={self.ratio:.4}, prominence={self.prominence:.4}]"


class AggregatedPartial(Partial):
    # The frequency of tone and fundamental are averages
    # based on the normalized frequencies of the underlying
    # partial , i.e. all frequencies are first reduced
    # to the same octave (octave 0). This enables to perform
    # an aggregation over multiple octaves.
    # reduced_frequency is the equivalent frequency within
    # octave 0 range (reference octave).
    partials: Optional[list[Partial]] = Field(default=None)
    fundamental_tone: Optional[Tone] = None
    ratio0: Optional[float] = None


class Note(BaseModel):
    name: NoteName
    octave: Octave
    frequency: Optional[Frequency] = None
    fundamental_index: Optional[int] = None
    samples: Optional[list[Sample]] = None
    spectra: Optional[list["Spectrum"]] = None
    partials: Optional[list[list[Partial]]] = None
    aggr_partials: Optional[list[AggregatedPartial]] = None


class AggregatedPartialDict(BaseModel):
    aggregation: Optional[tuple[str, ...]] = None
    root: dict[str, tuple[list[AggregatedPartial], dict[NoteName, Note]]]
    notes_octave_0: dict[NoteName, Note]


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


# class Note(BaseModel):
#     name: NoteName
#     octave: Octave
#     order_in_soundfile: int
#     frequency: Optional[Frequency] = None
#     fundamental_index: int = None
#     samples: Optional[list[Sample]] = None
#     spectra: Optional[list[Spectrum]] = None
#     partials: Optional[list[list[Partial]]] = None
#     aggr_partials: Optional[list[AggregatedPartial]] = None


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


# A .wav file contains a sequence of note recordings from the same instrument.
# These recordings are called clips or clippings.
# ClipPosition determines the order of a clipping in such a sequence (1=first)
ClipPosition = int
FileName = str
# User-provided info about specific clippings.
# This will overrule the automatic identification of these clippings.
ClipInfo = dict[ClipPosition, tuple[NoteName, OctaveNbr]]


class InstrumentGroup(BaseModel):
    grouptype: InstrumentGroupName
    instruments: list[Instrument]
    has_sound_samples: bool = False
    has_spectra: bool = False
    clipinfo_dict: Optional[dict[FileName, ClipInfo]] = None


class Dissonance(BaseModel):
    ratio: Ratio
    dissonance: float

    class Config:
        frozen = True


class ConsonanceValues(BaseModel):
    partials: list[Partial]
    harmonics: list[Partial]
    consonances: dict[PartialCombiType, list[Dissonance]]


class ConsonanceAnalysis(BaseModel):
    ref_note: NoteName
    ref_octave: OctaveNbr
    ref_frequency: Frequency
    fundamental: Partial
    all_partials: list[Partial]
    all_harmonics: list[Partial]
    noteratio_dict: dict[NoteName, Ratio]
    results: list[ConsonanceValues]
    optimal_consonances: Optional[ConsonanceValues] = None
    optimal_matches: Optional[dict[NoteName, tuple[Ratio, PartialCombiType, Dissonance]]] = None

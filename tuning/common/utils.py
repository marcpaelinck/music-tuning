import json
import logging
import math
import os
import re

from pydantic import BaseModel
from scipy.interpolate import interp1d

from tuning.common.classes import (
    FreqUnit,
    Instrument,
    InstrumentGroup,
    Note,
    NoteName,
    Spectrum,
    Tone,
)
from tuning.common.constants import DATA_FOLDER, FileType, InstrumentGroupName


def get_logger(name) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s - %(name)-12s %(levelname)-7s: %(message)s", datefmt="%H:%M:%S"
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger(__name__)


def get_path(groupname: InstrumentGroupName, filetype: FileType, filename: str = ""):
    return os.path.join(DATA_FOLDER, groupname.value, filetype.value, filename)


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


def db_to_ampl(db: float) -> float:
    """
    converts dB to amplitude
    https://blog.demofox.org/2015/04/14/decibels-db-and-amplitude/
    """
    return pow(10, db / 20)


def convert_freq(value: float, from_unit: FreqUnit, to_unit: FreqUnit, ref_value=None) -> float:
    if from_unit is to_unit:
        return value

    match (from_unit, to_unit):
        case (FreqUnit.HERZ, FreqUnit.CENT):
            return 1200 * math.log2(value)
        case (FreqUnit.CENT, FreqUnit.HERZ):
            return math.pow(2, value / 1200)
        case _:
            logger.error(f"Conversion from {from_unit.value} to {to_unit.value} not implemented.")


def get_partial(note: Note) -> Tone:
    return note.partials[note.partial_index].tone


def save_object_to_jsonfile(object: BaseModel, filepath: str):
    tempfilepath = filepath + "x"
    with open(tempfilepath, "w") as outfile:
        outfile.write(object.model_dump_json(indent=4))
    if os.path.exists(filepath):
        os.remove(filepath)
    os.rename(tempfilepath, filepath)


def save_group_to_jsonfile(group: InstrumentGroup):
    filepath = get_path(group.grouptype, FileType.SETTINGS, f"{group.grouptype.value}.json")
    save_object_to_jsonfile(group, filepath)
    # filepath = tempfilepath.replace(".jsonx", ".json")
    # with open(tempfilepath, "w") as outfile:
    #     outfile.write(group.model_dump_json(indent=4, exclude={"hello world"}))
    # if os.path.exists(filepath):
    #     os.remove(filepath)
    # os.rename(tempfilepath, filepath)


def read_object_from_jsonfile(objecttype: type, filepath: str) -> BaseModel:
    with open(filepath, "r") as infile:
        jsonvalue = infile.read()
        return objecttype.model_validate_json(jsonvalue)


def read_group_from_jsonfile(
    groupname: InstrumentGroupName,
    read_sounddata: bool = False,
    read_spectrumdata: bool = True,
    save_spectrumdata: bool = False,
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


def convert_spectrum_freq(spectrum: Spectrum, to_unit: FreqUnit, step: float = None) -> Spectrum:
    if spectrum.freq_unit is FreqUnit.HERZ:
        freqlist = [tone.frequency for tone in spectrum.tones]
        ampllist = [tone.amplitude for tone in spectrum.tones]
        match to_unit:
            case FreqUnit.CENT:
                if to_unit is spectrum.freq_unit:
                    return spectrum
                else:
                    step = step or 5
                    minval = int(
                        convert_freq(
                            max(5, int(spectrum.tones[0].frequency)),
                            from_unit=spectrum.freq_unit,
                            to_unit=to_unit,
                        )
                    )
                    maxval = int(
                        convert_freq(
                            int(spectrum.tones[-1].frequency),
                            from_unit=spectrum.freq_unit,
                            to_unit=to_unit,
                        )
                    )
                func = interp1d(freqlist, ampllist, assume_sorted=True)
                return Spectrum(
                    spectrumfilepath=spectrum.spectrumfilepath,
                    freq_unit=to_unit,
                    ampl_unit=spectrum.ampl_unit,
                    tones=[
                        Tone(
                            frequency=db_freq,
                            amplitude=func(
                                convert_freq(db_freq, from_unit=to_unit, to_unit=spectrum.freq_unit)
                            ),
                        )
                        for db_freq in range(minval, maxval, step)
                    ],
                )

            case _:
                ...

    logger.error(
        f"Method not implemented for conversion from {spectrum.freq_unit.value} to {to_unit.value}"
    )


def get_instrument_by_id(group: InstrumentGroup, code: str) -> Instrument:
    return next((instr for instr in group.instruments if instr.code == code), None)

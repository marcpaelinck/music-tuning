import json
import logging
import math
import os
import re

import numpy as np
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
from tuning.common.constants import DATA_FOLDER, Folder, InstrumentGroupName


def get_logger(name) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s - %(name)-12s %(levelname)-7s: %(message)s", datefmt="%H:%M:%S"
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger(__name__)


def get_path(groupname: InstrumentGroupName, filetype: Folder, filename: str = ""):
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


def ampl_to_loudness(amplitude: float) -> float:
    p_e = amplitude / math.sqrt(2)
    SPL = 20 * math.log10(p_e / 20)
    return (1 / 16) * pow(2, SPL / 10)


def convert_freq(value: float, from_unit: FreqUnit, to_unit: FreqUnit, ref_value=None) -> float:
    if from_unit is to_unit:
        return value

    match (from_unit, to_unit):
        case (FreqUnit.HERTZ, FreqUnit.CENT):
            return 1200 * math.log2(value)
        case (FreqUnit.CENT, FreqUnit.HERTZ):
            return math.pow(2, value / 1200)
        case _:
            logger.error(f"Conversion from {from_unit.value} to {to_unit.value} not implemented.")


def get_partial(note: Note) -> Tone:
    return note.partials[note.partial_index].tone


def save_object_to_jsonfile(
    object: BaseModel,
    filepath: str,
):
    tempfilepath = filepath + "x"
    with open(tempfilepath, "w") as outfile:
        outfile.write(object.model_dump_json(indent=4))
    if os.path.exists(filepath):
        os.remove(filepath)
    os.rename(tempfilepath, filepath)


def save_group_to_jsonfile(group: InstrumentGroup, save_spectrumdata=False):
    if save_spectrumdata:
        # set save flag of Spectrum objects
        logger.info("Saving spectra")
        for instrument in group.instruments:
            for note in instrument.notes:
                note.spectrum.save_spectrum_data = True
    filepath = get_path(group.grouptype, Folder.SETTINGS, f"{group.grouptype.value}.json")
    save_object_to_jsonfile(group, filepath)


def read_object_from_jsonfile(
    objecttype: type,
    groupname: InstrumentGroupName,
    filetype: Folder,
    filename: str,
) -> BaseModel:
    filepath = get_path(groupname, filetype, filename)
    with open(filepath, "r") as infile:
        jsonvalue = infile.read()
        return objecttype.model_validate_json(jsonvalue)


def read_group_from_jsonfile(
    groupname: InstrumentGroupName,
    read_sounddata: bool = False,
    read_spectrumdata: bool = False,
):
    """
    Imports an InstrumentGroup object from json.

    Args:
        groupname (InstrumentGroupName): Name of the group to import. This determines the forlder containing the json file.
        read_sounddata (bool, optional): If True, the .wav soundfiles will be imported. This can take several seconds. Defaults to False.
        read_spectrumdata (bool, optional): Same for spectrum data. Defaults to False.
        save_spectrumdata (bool, optional): Set this value to False if you don't want the spectrum data to be saved by the save_group_to_jsonfile
                                            method.
                                            Ideally this argument should be passed to the save_group_to_jsonfile method,
                                            but context values can't be passed to the Pydantic serializer. As a workaround, the
                                            value is set in Spectrum objects when they are created. Defaults to True.

    Returns:
        _type_: _description_
    """
    logger.info(
        f"Reading {groupname.value} info{' with sounddata' if read_sounddata else ''}{' with spectrum data' if read_spectrumdata else ''}."
    )
    with open(get_path(groupname, Folder.SETTINGS, f"{groupname.value}.json"), "r") as infile:
        jsonvalue = infile.read()
    return InstrumentGroup.model_validate_json(
        jsonvalue,
        context={
            "read_sounddata": read_sounddata,
            "read_spectrumdata": read_spectrumdata,
            "save_spectrumdata": save_spectrumdata,
        },
    )


def convert_spectrum_freq(spectrum: Spectrum, to_unit: FreqUnit, step: float = 5) -> Spectrum:
    """
    Returns a new Spectrum object, in which the frequency values are converted to the new unit.
    Currentlhy only converts from Hertz to Cent.

    Args:
        spectrum (Spectrum): The spectrum that should be converted.
        to_unit (FreqUnit): Unit to convert to.
        step (float, optional): The step width for the new unit. Amplitude values will be interpolated.
                                Defaults to 5.

    Returns:
        Spectrum: New converted spectrum.
    """
    # TODO unit test
    if spectrum.freq_unit is FreqUnit.HERTZ:
        match to_unit:
            case FreqUnit.CENT:
                if to_unit is spectrum.freq_unit:
                    return spectrum
                else:
                    minval = int(
                        convert_freq(
                            max(5, int(spectrum.frequencies[0])),
                            from_unit=spectrum.freq_unit,
                            to_unit=to_unit,
                        )
                    )
                    maxval = int(
                        convert_freq(
                            int(spectrum.frequencies[-1]),
                            from_unit=spectrum.freq_unit,
                            to_unit=to_unit,
                        )
                    )
                func = interp1d(spectrum.frequencies, spectrum.amplitudes, assume_sorted=True)
                cent_freqs = np.array([cent_freq for cent_freq in range(minval, maxval, step)])
                # Convert cent value back to hertz and find amplitude by interpolation
                cent_ampls = np.array(
                    [
                        func(convert_freq(cent_freq, from_unit=to_unit, to_unit=spectrum.freq_unit))
                        for cent_freq in cent_freqs
                    ]
                )
                return Spectrum(
                    spectrumfilepath=spectrum.spectrumfilepath,
                    freq_unit=FreqUnit.CENT,
                    ampl_unit=spectrum.ampl_unit,
                    frequencies=cent_freqs,
                    amplitudes=cent_ampls,
                )

            case _:
                ...

    logger.error(
        f"Method not implemented for conversion from {spectrum.freq_unit.value} to {to_unit.value}"
    )


def get_instrument_by_id(group: InstrumentGroup, code: str) -> Instrument:
    return next((instr for instr in group.instruments if instr.code == code), None)

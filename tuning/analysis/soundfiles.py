"""
This module enables to detect the notes in .wav files and to return
a separate wave pattern for each note.
"""
import json
import logging
import re

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.io import wavfile

from tuning.analysis.classes import Filename, Note, NoteInfo, NoteInfoList, OmbakType
from tuning.analysis.constants import FileType, InstrumentGroup
from tuning.analysis.utils import get_octaves, get_path

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def note_from_shortcode(code: str) -> Note:
    """
    Returns a Note object for a given short code "i", "o", "e" etc.
    Also recognizes the intermediate notes "eu" and "ai".
    """
    return next((note for note in Note if note.value == f"d{code}ng"), None)


class Boundary(BaseModel):
    type: str
    index: int


class Range(BaseModel):
    start: int
    end: int


def parse_wav_file(group: InstrumentGroup, filename: str, noteinfo_list: NoteInfoList):
    """
    Detects the separate notes in the file and returns
    the NoteInfoList with added WavePatterns
    """
    logger.info(f"Parsing file {filename}.")
    sample_rate, data = wavfile.read(get_path(group, FileType.SOUND, filename))
    # Detect start and end positions of contiguous series with amplitude > average.
    logger.info("--- Detecting boundaries of sound range.")
    amplitudes = abs(data)
    threshold = 0.5 * np.average(amplitudes)
    boundaries: list[Boundary] = [
        Boundary(type="Start" if is_start else "End", index=idx)
        for idx in range(len(amplitudes) - 1)
        if (is_start := (amplitudes[idx][0] < threshold < amplitudes[idx + 1][0]))
        or (amplitudes[idx + 1][0] < threshold < amplitudes[idx][0])
    ]
    # Create ranges from the start and stop boundaries
    logger.info("--- Creating sound ranges.")
    ranges = [
        Range(start=boundaries[i].index, end=boundaries[i + 1].index)
        for i in range(len(boundaries) - 1)
        if boundaries[i].type == "Start" and boundaries[i + 1].type == "End"
    ]
    # Merge ranges that are separated by less than 0.1 second
    logger.info("--- Merging sound ranges.")
    threshold = 0.1 * sample_rate
    idx = 0
    while idx < len(ranges) - 1:
        range1 = ranges[idx]
        range2 = ranges[idx + 1]
        if range2.start - range1.end < threshold:
            range1.end = range2.start
            ranges.remove(range2)
        else:
            idx += 1
    # Drop ranges that are shorter than 1 second. This removes unintended noises
    # such as switching the recorder on or off
    logger.info("--- Dropping short sound ranges.")
    threshold = sample_rate
    ranges = [block for block in ranges if block.end - block.start > threshold]
    # Check that the number of series corresponds with the given info.
    logger.info("The following sound ranges were detected:")
    for soundrange in ranges:
        logger.info(
            f"--- ({(soundrange.start/sample_rate):.2f} sec., {(soundrange.end/sample_rate):.2f} sec.)"
        )
    if len(ranges) != len(noteinfo_list.notes):
        logger.error(
            f"--- {group} .wav file {filename}: {len(noteinfo_list.notes)} notes are given, but {len(ranges)} were detected."
        )
        for soundrange in ranges:
            logger.error(
                f"--- {(soundrange.start/sample_rate):.2f} sec. - {(soundrange.end/sample_rate):.2f} sec."
            )

        return None
    # Retrieve the wave patterns
    logger.info("Number of sound ranges matches with expected number of notes.")
    for index in range(len(noteinfo_list.notes)):
        noteinfo_list.notes[index].wav.extend(
            amplitudes[ranges[index].start : ranges[index].end]
        )


def parse_file_info(group: InstrumentGroup) -> list[tuple[Filename, NoteInfoList]]:
    octaves = get_octaves(group)
    fileinfo = pd.read_excel(get_path(group, FileType.SOUND, "info.xlsx")).to_dict(
        orient="records"
    )
    wavfile_info_list = []
    for file in fileinfo:
        shortcodes = re.findall("([aeiou]{1,2})(\\d)", file["notes"])
        notes = NoteInfoList(
            group=group,
            comment="",
            notes=[
                NoteInfo(
                    note=note_from_shortcode(code[0]),
                    instrument=file["instrument"],
                    ombaktype=OmbakType(file["ombaktype"]),
                    octave=octaves[code[1]],
                    freq=0,
                    index=0,
                    partials=[],
                )
                for code in shortcodes
            ],
        )
        wavfile_info_list.append((file["filename"], notes))

    return wavfile_info_list


if __name__ == "__main__":
    wavfile_info_list = parse_file_info(InstrumentGroup.TEST)
    for item in wavfile_info_list:
        parse_wav_file(
            group=InstrumentGroup.TEST, filename=item[0], noteinfo_list=item[1]
        )

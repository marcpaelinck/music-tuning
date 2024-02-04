"""
This module enables to detect the notes in .wav files and to return
a separate wave pattern for each note.
"""
import logging
import os
from operator import ge

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.pylab import f
from scipy.io import wavfile

from tuning.common.classes import (
    ClipRange,
    Instrument,
    InstrumentGroup,
    Sample,
    SoundData,
)
from tuning.common.constants import (
    AMPLITUDE_THRESHOLD_TO_AVERAGE_RATIO,
    MINIMUM_SAMPLE_DURATION,
    SHORT_BLANKS_DURATION_THRESHOLD,
    FileType,
    InstrumentGroupName,
)
from tuning.common.utils import get_path
from tuning.instruments.enhance_soundfiles import (
    equalize_note_amplitudes,
    reconstruct_clipped_regions,
)

logging.basicConfig(
    format="%(asctime)s - %(name)-12s %(levelname)-7s: %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_soundclip_ranges(
    data: SoundData, threshold: float, include: callable = ge
) -> list[ClipRange]:
    """
    Returns ranges in which the sound data exceeds the given threshold.

    Args:
        data (SoundData): sound data that should be examined.
        threshold (float): amplitude value.
        include (callable, optional): comparison function. Indicates how data should relate
                                      to the threshold in order to be included. Default ge.

    Returns:
        list[ClipRange]: _description_
    """
    amplitudes = pd.DataFrame(data).abs().max(axis=1).to_frame(name="amplitude")

    # Mark start and end positions of contiguous clip ranges with amplitude >= threshold.
    # If the recording ends with high amplitude, add "End" label to the last row.
    logger.info("--- Detecting boundaries of sound ranges.")
    # -- Add column containing amplitude value of the next row.
    events = amplitudes.assign(next=amplitudes.shift(-1))
    lastindex = events.index[-1]
    # -- Add start or end labels.
    events["event"] = np.where(
        # -- Add start label when threshold is being violated.
        ~include(events.amplitude, threshold) & include(events.next, threshold),
        "Start",
        # -- Add end label when violation ends.
        np.where(
            (include(events.amplitude, threshold) & ~include(events.next, threshold))
            | ((events.index == lastindex) & include(events.amplitude, threshold)),
            "End",
            "",
        ),
    )
    # Add an end label to the last row if the last value still exceeds the threshold.
    events["event"] = np.where(
        (events.amplitude < threshold) & (events.next >= threshold),
        "Start",
        np.where(
            ((events.amplitude >= threshold) & (events.next < threshold))
            | ((events.index == lastindex) & (events.amplitude >= threshold)),
            "End",
            "",
        ),
    )
    #  Remove all non-labeled rows
    events = events[~(events.event == "")].drop(["next"], axis=1)
    if events.empty:
        return []

    # Remove possible unmatched final start (note that 0 <= #odd rows - #even rows <=1)
    logger.info("--- Creating clip ranges.")
    # Create ranges from the indices of consecutive start and stop labels.
    start_events = events.iloc[::2]  # odd numbered rows
    end_events = events.iloc[1::2]  # even numbered rows
    # The following assertions ensure that the start and stop labels alternate
    # and that each start label has a corresponding end label.
    assert list(start_events.event.unique()) == ["Start"]
    assert list(end_events.event.unique()) == ["End"]
    assert len(start_events) == len(end_events)
    # Create clip ranges from the start and stop boundaries
    intervals = pd.DataFrame({"start": start_events.index, "end": end_events.index}).to_dict(
        orient="records"
    )
    return [ClipRange(start=interval["start"], end=interval["end"]) for interval in intervals]


def detect_individual_notes(sample_rate: int, data: SoundData) -> list[ClipRange]:
    """
    Detects the separate notes in the sound file, then creates WavePattern objects
    for each note and adds them to the given NoteInfoList.
    """
    logger.info(f"- Parsing individual notes.")
    clipranges = create_soundclip_ranges(
        data, float(AMPLITUDE_THRESHOLD_TO_AVERAGE_RATIO * abs(data).mean())
    )

    # Merge clip ranges that are separated by less than 0.1 second
    logger.info("--- Merging sound ranges.")
    threshold = SHORT_BLANKS_DURATION_THRESHOLD * sample_rate
    idx = 0
    while idx < len(clipranges) - 1:
        range1 = clipranges[idx]
        range2 = clipranges[idx + 1]
        if range2.start - range1.end < threshold:
            range1.end = range2.end
            clipranges.remove(range2)
        else:
            idx += 1

    # Drop clip ranges that are shorter than 1 second. This removes unintended noises
    # such as switching the recorder on or off.
    logger.info("--- Dropping short clip ranges (random noises).")
    threshold = MINIMUM_SAMPLE_DURATION * sample_rate
    clipranges = [block for block in clipranges if block.end - block.start > threshold]
    # Check that the number of series corresponds with the given info.
    logger.info("--- The following clip ranges were created:")
    for cliprange in clipranges:
        logger.info(
            f"---    ({(cliprange.start/sample_rate):.2f} sec., {(cliprange.end/sample_rate):.2f} sec.)"
        )
    return clipranges


def process_wav_file(
    group: InstrumentGroup,
    instrument: Instrument,
    clip_fix: bool = True,
    equalize_notes: bool = True,
    save: bool = False,
    suffix: str = "-ENHANCED",
) -> list[Sample]:
    """
    Performs several modifications to a .wav file.
    clip_fix: reconstructs clipped regions, these are areas where the amplitude exceeds the maximum value.
              Clipping will be performed for each note separately.
    equalize_notes: increases the amplitude of each separate note so that all their maximum values
                      are equal. Ignored if parse_ranges is False.
    save: saves the processed data.
    suffix: text to be added to the original file name.
    """
    logger.info(f"Processing file {instrument.soundfile}.")
    sample_rate, data = wavfile.read(
        get_path(group.grouptype, FileType.SOUND, instrument.soundfile)
    )

    if clip_fix:
        data = reconstruct_clipped_regions(data)

    clipranges = detect_individual_notes(sample_rate, data)
    if len(clipranges) != len(instrument.notes):
        logger.error(
            f"--- {group.grouptype} .wav file {instrument.soundfile}: {len(instrument.notes)} "
            + f"notes are given, but {len(clipranges)} were detected."
        )
        instrument.comment = (
            f"ERROR: {len(instrument.notes)} notes are given, but {len(clipranges)} were detected."
        )
        instrument.error = True
    else:
        logger.info("Number of clip ranges matches with expected number of notes.")
        # add_samples_to_instrument(instrument, clipranges, sample_rate, data)

    if equalize_notes:
        data = equalize_note_amplitudes(sample_rate, data, clipranges)

    name, extension = os.path.splitext(instrument.soundfile)
    save_filename = get_path(group.grouptype, FileType.SOUND, name + suffix + extension)

    if save:
        wavfile.write(save_filename, sample_rate, data)

    # Clip data into separate notes
    clippings = [
        Sample(
            data=data[cliprange.start : cliprange.end],
            sample_rate=sample_rate,
            soundfilepath=save_filename,
            cliprange=cliprange,
        )
        for cliprange in clipranges
    ]
    return clippings


def get_sound_samples(group: InstrumentGroup) -> InstrumentGroup:
    for instrument in group.instruments:
        samples = process_wav_file(group, instrument)
        for note in instrument.notes:
            note.sample = samples[note.order_in_soundfile]
    group.has_sound_samples = True
    return group


def add_subplot(
    figure: Figure,
    x: list[float],
    y: list[float],
    clipranges: list[ClipRange],
    offset: float,
):
    axs = figure.add_axes((0.2, offset, 0.6, 0.2))
    axs.plot(x, y)
    for cliprange in clipranges:
        axs.axvline(x=cliprange.start, color="b")
        axs.axvline(x=cliprange.end, color="r")


GROUP = InstrumentGroupName.TEST

if __name__ == "__main__":
    ...
    # orchestra = read_soundfile_info(GROUP)
    # instrument = next((instr for instr in orchestra.instruments if instr.code == "PEM1"), None)
    # process_wav_file(group=GROUP, instrument=instrument)
    # for note in instrument.notes:
    #     note.spectrum = create_spectrum(note)
    #     note.spectrumfile = (
    #         f"{instrument.name}-{instrument.code}-{note.name}-{note.octave.index}.csv"
    #     )
    #     save_spectrum(
    #         note.spectrum,
    #         filename=note.spectrumfile,
    #         group=GROUP,
    #     )

    # figure = plt.figure()
    # offset = 0.1
    # for instrument in orchestra:
    #     instrument, sample_rate, amplitudes, clipranges = parse_individual_notes(
    #         group=GROUP, instrument=instrument
    #     )
    #     add_subplot(
    #         figure,
    #         x=[t / sample_rate for t in range(len(amplitudes))],
    #         y=amplitudes,
    #         ranges=[(rng.start / sample_rate, rng.end / sample_rate) for rng in clipranges],
    #         offset=offset,
    #     )
    #     offset += 0.3
    # plt.show()

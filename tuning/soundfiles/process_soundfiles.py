"""
This module enables to detect the notes in .wav files and to return
a separate wave pattern for each note.
"""

import os

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
    ENHANCED_SOUNDFILE_SUFFIX,
    MINIMUM_SAMPLE_DURATION,
    SHORT_BLANKS_DURATION_THRESHOLD,
    SOUNDFILE_CLIP_FIX,
    SOUNDFILE_EQUALIZE_NOTES,
    Folder,
    InstrumentGroupName,
)
from tuning.common.utils import get_logger, get_path
from tuning.soundfiles.enhance_soundfiles import (
    equalize_note_amplitudes,
    reconstruct_clipped_regions,
)
from tuning.soundfiles.utils import create_soundclip_ranges

logger = get_logger(__name__)


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
    clip_fix: bool = SOUNDFILE_CLIP_FIX,
    equalize_notes: bool = SOUNDFILE_EQUALIZE_NOTES,
    save: bool = ENHANCED_SOUNDFILE_SUFFIX,
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
    sample_rate, data = wavfile.read(get_path(group.grouptype, Folder.SOUND, instrument.soundfile))

    if clip_fix:
        logger.info("Repairing regions that exceeded maximum recording level.")
        data = reconstruct_clipped_regions(data)

    logger.info("--- Detecting note boundaries.")
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
        logger.info("Equalizing the amplitude of the individual noise.")
        data = equalize_note_amplitudes(sample_rate, data, clipranges)

    name, extension = os.path.splitext(instrument.soundfile)
    save_filename = get_path(group.grouptype, Folder.SOUND, name + suffix + extension)

    if save:
        logger.info("Saving the enhanced file.")
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

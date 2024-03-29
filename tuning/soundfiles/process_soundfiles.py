"""
This module enables to detect the notes in .wav files and to return
a separate wave pattern for each note.
"""

import os

import numpy as np
from matplotlib.figure import Figure
from matplotlib.pylab import f
from scipy.io import wavfile

from tuning.common.classes import (
    ClipRange,
    Frequency,
    Instrument,
    InstrumentGroup,
    IntFrequency,
    NoteName,
    OctaveNbr,
    Sample,
    SoundData,
)
from tuning.common.constants import (
    AMPLITUDE_THRESHOLD_TO_AVERAGE_RATIO,
    ENHANCED_SOUNDFILE_SUFFIX,
    EXPECTED_SAMPLE_DURATION,
    MINIMUM_SAMPLE_DURATION,
    NOTE_PATTERN,
    SHORT_BLANKS_DURATION_THRESHOLD,
    SOUNDFILE_CLIP_FIX,
    SOUNDFILE_EQUALIZE_NOTES,
    Folder,
    InstrumentGroupName,
)
from tuning.common.utils import (
    get_logger,
    get_path,
    read_group_from_jsonfile,
    save_group_to_jsonfile,
)
from tuning.soundfiles.enhance_soundfiles import (
    equalize_note_amplitudes,
    reconstruct_clipped_regions,
)
from tuning.soundfiles.instrumentinfo import (
    create_group_from_info_file,
    get_notefreq_dict,
)
from tuning.soundfiles.utils import create_soundclip_ranges

logger = get_logger(__name__)


def determine_fundamental_freq(
    sample: SoundData, sample_rate: float, freq_range: tuple[float, float]
) -> int:
    """
    Searches for the frequency of the fundamental

    Args:
        sample (SoundData): sample to analze
        sample_rate (float): sample rate of the sound sample
        freq_range (tuple[float, float]): limit the search to this range

    Returns:
        int: rounded value for the frequency
    """

    # Compute Fourier transform of windowed signal on one channel
    data = sample[:, 0]
    yval = np.fft.rfft(data)
    xval = np.fft.rfftfreq(data.size, 1 / sample_rate)

    # Find the peak and interpolate to get a more accurate peak
    # Limit the search to the given freq_range
    lb_idx = (np.abs(xval - freq_range[0])).argmin()
    ub_idx = (np.abs(xval - freq_range[1])).argmin()
    fund_idx = np.argmax(abs(yval[lb_idx:ub_idx])) + lb_idx

    # Convert to equivalent frequency
    return int(xval[fund_idx])


def matches_note_pattern(clip: SoundData) -> bool:
    length = len(clip)
    intervals = [
        (int(sum(NOTE_PATTERN[:i] * length)), int(sum(NOTE_PATTERN[: i + 1] * length)) - 1)
        for i in range(len(NOTE_PATTERN))
    ]
    maxima = [np.average(abs(clip[interval[0] : interval[1]])) for interval in intervals]
    # return all(maxima[i] > 1.1 * maxima[i + 2] for i in range(segments - 2))
    return all(maxima[i] > maxima[i + 1] for i in range(len(intervals) - 1))


def detect_individual_notes(
    sample_rate: int, data: SoundData, notefreq_dict: dict[(NoteName, int), Frequency]
) -> list[tuple[tuple[NoteName, OctaveNbr], IntFrequency, ClipRange]]:
    """
    Detects the separate notes in the sound file, then creates WavePattern objects
    for each note and adds them to the given NoteInfoList.
    """
    note_freq_array = np.abs(np.array(list(notefreq_dict.keys())))
    note_values_list = list(notefreq_dict.values())

    def find_note_by_freq(freq: Frequency) -> tuple[NoteName, OctaveNbr] | None:
        index = (np.abs(note_freq_array - freq)).argmin()
        if 0 <= index < len(notefreq_dict):
            return note_values_list[index]
        return None

    logger.info(f"- Parsing individual notes.")
    clipranges = create_soundclip_ranges(
        data, float(AMPLITUDE_THRESHOLD_TO_AVERAGE_RATIO * abs(data).mean())
    )

    # Merge clip ranges that are separated by less than 0.1 second
    logger.info("--- Merging sound ranges.")
    minimum_length = SHORT_BLANKS_DURATION_THRESHOLD * sample_rate
    idx = 0
    while idx < len(clipranges) - 1:
        range1 = clipranges[idx]
        range2 = clipranges[idx + 1]
        if range2.start - range1.end < minimum_length:
            range1.end = range2.end
            clipranges.remove(range2)
        else:
            idx += 1

    # Drop clip ranges that are shorter than the minimum recording duration. This removes unintended noises
    # such as switching the recorder on or off.
    logger.info("--- Dropping short clip ranges (random noises).")
    minimum_length = MINIMUM_SAMPLE_DURATION * sample_rate
    expected_length = EXPECTED_SAMPLE_DURATION * sample_rate
    note_freqs = list(notefreq_dict.keys())
    freq_range = (min(note_freqs) * 0.9, max(note_freqs) * 1.1)

    clipranges = [
        block
        for block in clipranges
        if (block.end - block.start > minimum_length)
        and (
            (block.end - block.start > expected_length)
            or matches_note_pattern(data[block.start : block.end])
        )
    ]
    frequencies = [
        determine_fundamental_freq(data[block.start : block.end], sample_rate, freq_range)
        for block in clipranges
    ]
    noteinfo = [find_note_by_freq(freq) for freq in frequencies]

    # Check that the number of series corresponds with the given info.
    logger.info("--- The following clip ranges were detected:")
    for i in range(len(clipranges)):
        logger.info(
            f"---    ({(clipranges[i].start/sample_rate):.2f} sec., {(clipranges[i].end/sample_rate):.2f} sec.) freq={(frequencies[i])} note={noteinfo[i]}"
        )
    return noteinfo, frequencies, clipranges


def process_wav_file(
    group: InstrumentGroup,
    instrument: Instrument,
    notefreq_dict: dict[(NoteName, int), Frequency],
    clip_fix: bool = SOUNDFILE_CLIP_FIX,
    equalize_notes: bool = SOUNDFILE_EQUALIZE_NOTES,
    save: bool = True,
    suffix: str = ENHANCED_SOUNDFILE_SUFFIX,
) -> list[Sample]:
    """
    Performs several modifications to a .wav file.
    clip_fix: reconstructs clipped regions, these are areas where the amplitude exceeds the maximum value.
              Clipping will be performed for each note separately.
    equalize_notes: increases the amplitude of each separate note so that all their maximum values
                      are equal. Ignored if parse_ranges is False.
    save: saves the processed data if True.
    suffix: text to be added to the original file name.
    """
    logger.info(f"Processing file {instrument.soundfilename}.")
    sample_rate, data = wavfile.read(
        get_path(group.grouptype, Folder.SOUND, instrument.soundfilename)
    )

    if clip_fix:
        logger.info("Repairing regions that exceeded maximum recording level.")
        data = reconstruct_clipped_regions(data)

    logger.info("--- Detecting note boundaries.")
    noteinfo, frequencies, clipranges = detect_individual_notes(sample_rate, data, notefreq_dict)

    if len(noteinfo) != len(instrument.notes):
        logger.error(
            f"--- {group.grouptype} .wav file {instrument.soundfilename}: {len(instrument.notes)} "
            + f"notes are given, but {len(noteinfo)} were detected."
        )
        instrument.comment = (
            f"ERROR: {len(instrument.notes)} notes are given, but {len(noteinfo)} were detected."
        )
        instrument.error = True
    else:
        logger.info("Number of clip ranges matches with expected number of notes.")

    if equalize_notes:
        logger.info("Equalizing the amplitude of the individual noise.")
        data = equalize_note_amplitudes(sample_rate, data, clipranges)

    name, extension = os.path.splitext(instrument.soundfilename)
    save_filename = name + suffix + extension
    instrument.soundfilename = save_filename
    save_filepath = get_path(group.grouptype, Folder.SOUND, name + suffix + extension)

    if save:
        logger.info("Saving the enhanced file.")
        wavfile.write(save_filepath, sample_rate, data)

    # Clip data into separate notes
    for i in range(len(clipranges)):
        note = next(
            (
                note
                for note in instrument.notes
                if note.name == noteinfo[i][0] and note.octave.index == noteinfo[i][1]
            ),
            None,
        )
        if not note:
            logger.warning(
                f"({noteinfo[i][0].name}, {noteinfo[i][1]}) detected but does occur in {instrument.instrumenttype.value}. Clip range will be ignored."
            )
        else:
            note.sample = Sample(
                data=data[clipranges[i].start : clipranges[i].end],
                sample_rate=sample_rate,
                soundfilepath=save_filepath,
                cliprange=clipranges[i],
            )
            note.freq = frequencies[i]


def process_sound_files(group: InstrumentGroup) -> InstrumentGroup:
    notefreq_dict = get_notefreq_dict(group.grouptype)
    for instrument in group.instruments:
        process_wav_file(group=group, instrument=instrument, notefreq_dict=notefreq_dict)
    group.has_sound_samples = True


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


if __name__ == "__main__":
    ...
    # Set this value before running
    # GROUPNAME = InstrumentGroupName.TEST
    # orchestra = create_group_from_info_file(GROUPNAME)
    # process_sound_files(orchestra)

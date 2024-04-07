"""
This module enables to detect the notes in .wav files and to return
a separate wave pattern for each note.
"""

import os
import statistics
from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.pylab import f
from scipy.io import wavfile

from tuning.common.classes import (
    ClipInfo,
    ClipRange,
    FileName,
    Frequency,
    Instrument,
    InstrumentGroup,
    InstrumentType,
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
    SKIP_STRIKE_DURATION,
    SOUNDFILE_CLIP_FIX,
    SOUNDFILE_EQUALIZE_NOTES,
    Folder,
    InstrumentGroupName,
)
from tuning.common.utils import (
    get_logger,
    get_path,
    step_read_group_from_jsonfile,
    step_save_group_to_jsonfile,
)
from tuning.soundfiles.enhance_soundfiles import (
    equalize_note_amplitudes,
    reconstruct_clipped_regions,
)
from tuning.soundfiles.instrumentinfo import get_notefreq_dict
from tuning.soundfiles.utils import create_soundclip_ranges

logger = get_logger(__name__)


def determine_fundamental_freq(
    sample: SoundData,
    sample_rate: float,
    notefreq_dict: dict[(NoteName, OctaveNbr), Frequency],
    note_oct: tuple[NoteName, OctaveNbr] | None,
) -> int:
    """
    Searches for the frequency of the fundamental for a given note sample.
    This is the frequency with the highest amplitude within the estimated frequency range of the
    instrument's notes (fundamentals).

    Args:
        sample (SoundData): note sample to analyze
        sample_rate (float): sample rate of the sound sample
        notefreq_dict (dict[(NoteName, OctaveNbr), Frequency]): estimated frequency per note.
        note_oct (tuple[NoteName, OctaveNbr] | None): note and octave info for the note sample

    Returns:
        int: rounded value for the frequency
    """

    # Compute Fourier transform of windowed signal on one channel
    data = sample[:, 0]
    yval = np.fft.rfft(data)
    xval = np.fft.rfftfreq(data.size, 1 / sample_rate)

    # Find the peak and interpolate to get a more accurate peak
    # Limit the search to the given freq_range
    if note_oct:
        # Note for this clip is given by user. Restrict search range around
        # estimated note frequency.
        freq_range = (notefreq_dict[note_oct] * 0.9, notefreq_dict[note_oct] * 1.1)
    else:
        # Set search range to the entire frequency range of the instrument.
        note_freqs = list(notefreq_dict.values())
        freq_range = (min(note_freqs) * 0.9, max(note_freqs) * 1.1)
    # Determine range indices in the xval/yval arrays
    lb_idx = (np.abs(xval - freq_range[0])).argmin()
    ub_idx = (np.abs(xval - freq_range[1])).argmin()
    # Find the index of the highest amplitude in the given range
    fundamental_idx = np.argmax(abs(yval[lb_idx:ub_idx])) + lb_idx

    # Return corresponding frequency
    return int(xval[fundamental_idx])

def envelope(clip: SoundData, rate, threshold):
    """
    Inputs audio data in the form of a numpy array. Converts to pandas series
    to find the rolling average and apply the absolute value to the signal at all points.
    
    Additionally takes in the sample rate and threshold (amplitude). Data below the threshold
    will be filtered out. This is useful for filtering out environmental noise from recordings. 
    """
    mask = []
    signal = pd.Series(np.abs(clip)) # Convert to series to find rolling average and apply absolute value to the signal at all points. 
    signal_mean = signal.rolling(window = int(rate/10), min_periods = 1, center = True).mean() # Take the rolling average of the series within our specified window.
    
    for mean in signal_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return np.array(mask)


def matches_note_pattern(clip: SoundData, start: float) -> bool:
    length = len(clip)
    intervals = [
        (int(sum(NOTE_PATTERN[:i] * length)), int(sum(NOTE_PATTERN[: i + 1] * length)) - 1)
        for i in range(len(NOTE_PATTERN))
    ]
    maxima = [np.average(abs(clip[interval[0] : interval[1]])) for interval in intervals]
    # return all(maxima[i] > maxima[i + 2] for i in range(len(intervals) - 2))
    matches = all(maxima[0] > maxima[i + 1] for i in range(len(intervals) - 1))
    return matches


def remove_strikes(cliprange_list: list[ClipRange], sample_rate: int):
    return [ClipRange(start=c.start + int(SKIP_STRIKE_DURATION * sample_rate), end=c.end) for c in cliprange_list]


def detect_individual_notes(
    sample_rate: int,
    data: SoundData,
    notefreq_dict: dict[(NoteName, OctaveNbr), Frequency],
    clipinfo: ClipInfo,
) -> dict[tuple[NoteName, OctaveNbr], list[tuple[Frequency, ClipRange]]]:
    """
    Detects the separate notes in the sound file, then creates WavePattern objects
    for each note and adds them to the given NoteInfoList.
    """
    note_freq_array = np.abs(np.array(list(notefreq_dict.values())))
    note_values_list = list(notefreq_dict.keys())

    def find_note_by_freq(freq: Frequency) -> tuple[NoteName, OctaveNbr] | None:
        index = (np.abs(note_freq_array - freq)).argmin()
        if 0 <= index < len(notefreq_dict):
            return note_values_list[index]
        return None

    logger.info(f"- Parsing individual notes.")
    cliprange_list = create_soundclip_ranges(
        data=data, sample_rate=sample_rate, threshold=float(AMPLITUDE_THRESHOLD_TO_AVERAGE_RATIO * abs(data).mean())
    )

    # Merge clip ranges that are separated by less than 0.1 second
    logger.info(f"--- Detected {len(cliprange_list)} clip ranges.")
    # logger.info("--- Merging sound ranges.")
    # minimum_length = SHORT_BLANKS_DURATION_THRESHOLD * sample_rate
    # idx = 0
    # while idx < len(cliprange_list) - 1:
    #     range1 = cliprange_list[idx]
    #     range2 = cliprange_list[idx + 1]
    #     if range2.start - range1.end < minimum_length:
    #         range1.end = range2.end
    #         cliprange_list.remove(range2)
    #     else:
    #         idx += 1
    # logger.info(f"--- Remaining: {len(cliprange_list)} clip ranges.")

    # Drop clip ranges that are shorter than the minimum recording duration. This removes unintended noises
    # such as switching the recorder on or off.
    logger.info("--- Dropping short clip ranges (random noises).")
    minimum_length = MINIMUM_SAMPLE_DURATION * sample_rate
    expected_length = EXPECTED_SAMPLE_DURATION * sample_rate

    # Create a list of clipranges that are recognized as note patterns
    cliprange_list_copy = cliprange_list  ##########
    cliprange_list = [
        block
        for block in cliprange_list
        if (block.end - block.start > minimum_length)
        and (
            (block.end - block.start > expected_length)
            and matches_note_pattern(data[block.start : block.end], block.start)
        )
    ]

    ##########
    diff = [
        (
            block,
            (block.end - block.start <= expected_length),
            not matches_note_pattern(data[block.start : block.end], block.start),
        )
        for block in cliprange_list_copy if (block not in cliprange_list) and (block.end - block.start > minimum_length)]
    for cliprange, tooshort, nomatch in diff:
        logger.warning(
            f"---        ({(cliprange.start/sample_rate):.2f} sec., {(cliprange.end/sample_rate):.2f} sec.) {"is too short" if tooshort else ""} {"does not match pattern" if nomatch else ""}"
        )

    ##########

    # Skip the first part of the clip recording (the strike)
    cliprange_list = remove_strikes(cliprange_list, sample_rate)

    # Determine the note and octave based on the fundamental frequency
    frequency_list = [
        determine_fundamental_freq(data[block.start : block.end], sample_rate, notefreq_dict, clipinfo.get(pos, None))
        for block, pos in zip(cliprange_list, range(len(cliprange_list)))
    ]
    noteinfo_list = [find_note_by_freq(freq) for freq in frequency_list]

    # Replace clipping information if explicit info is available
    # for (filename, clip_seq), (notename, octave_seq) in clipping_info.items():
    #     if filename == filename:
    #         del noteinfo_list[clip_seq]
    #         noteinfo_list.insert(clip_seq, (notename, octave_seq))

    # create a dictionary with entries per note
    notedict: dict[tuple[NoteName, OctaveNbr], list[tuple[Frequency, ClipRange]]] = defaultdict(list)
    for noteinfo, frequency, cliprange in zip(noteinfo_list, frequency_list, cliprange_list):
        notedict[noteinfo].append((frequency, cliprange))

    # Check that the number of series corresponds with the given info.
    logger.info("--- The following clip ranges were detected:")
    for noteinfo, freq_clip_list in notedict.items():
        # for i in range(len(cliprange_list)):
        logger.info(f"---    note={noteinfo}:")
        for frequency, cliprange in freq_clip_list:
            logger.info(
                f"---        ({(cliprange.start/sample_rate):.2f} sec., {(cliprange.end/sample_rate):.2f} sec.) freq={(frequency)}"
            )
    return notedict


def process_wav_file(
    group: InstrumentGroup,
    instrument: Instrument,
    notefreq_dict: dict[(NoteName, OctaveNbr), Frequency],
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
    sample_rate, data = wavfile.read(get_path(group.grouptype, Folder.SOUND, instrument.soundfilename))

    if clip_fix:
        logger.info("Repairing regions that exceeded maximum recording level.")
        data = reconstruct_clipped_regions(data=data, sample_rate=sample_rate)

    logger.info("--- Detecting note boundaries.")
    noteinfo_dict = detect_individual_notes(
        sample_rate=sample_rate,
        data=data,
        notefreq_dict=notefreq_dict,
        clipinfo=group.clipinfo_dict.get(instrument.soundfilename, {}),
    )

    if len(noteinfo_dict) != len(instrument.notes):
        logger.error(
            f"--- {group.grouptype} .wav file {instrument.soundfilename}: {len(instrument.notes)} "
            + f"notes are given, but {len(noteinfo_dict)} were detected."
        )
        instrument.comment = f"ERROR: {len(instrument.notes)} notes are given, but {len(noteinfo_dict)} were detected."
        instrument.error = True
    else:
        logger.info("Number of clip ranges matches with expected number of notes.")

    if equalize_notes:
        logger.info("Equalizing the amplitude of the individual notes.")
        cliprange_list = [clip for clip_list in noteinfo_dict.values() for _, clip in clip_list]
        data = equalize_note_amplitudes(sample_rate, data, cliprange_list)

    name, extension = os.path.splitext(instrument.soundfilename)
    save_filename = name + suffix + extension
    instrument.soundfilename = save_filename
    save_filepath = get_path(group.grouptype, Folder.SOUND, name + suffix + extension)

    if save:
        logger.info("Saving the enhanced file.")
        wavfile.write(save_filepath, sample_rate, data)

    # Clip data into separate notes
    for noteinfo, freq_clip_list in noteinfo_dict.items():
        # for i in range(len(cliprange_list)):
        note = (
            None
            if not noteinfo
            else next(
                (note for note in instrument.notes if note.name == noteinfo[0] and note.octave.index == noteinfo[1]),
                None,
            )
        )
        if not note:
            logger.warning(
                f"{noteinfo} detected but does occur in {instrument.instrumenttype.value}. Clip range will be ignored."
            )
        else:
            note.samples = list()
            freq_sum = 0
            for frequency, cliprange in freq_clip_list:
                note.samples.append(
                    Sample(
                        data=data[cliprange.start : cliprange.end],
                        sample_rate=sample_rate,
                        soundfilepath=save_filepath,
                        cliprange=cliprange,
                    )
                )
                freq_sum += frequency
            note.frequency = freq_sum / len(freq_clip_list)


def step_process_sound_files(group: InstrumentGroup) -> InstrumentGroup:
    notefreq_dict = get_notefreq_dict(group.grouptype)
    for instrument in group.instruments:
        process_wav_file(group=group, instrument=instrument, notefreq_dict=notefreq_dict[instrument.instrumenttype])
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


if __name__ == "__main__":
    ...
    # Set this value before running
    # GROUPNAME = InstrumentGroupName.TEST
    # orchestra = create_group_from_info_file(GROUPNAME)
    # process_sound_files(orchestra)

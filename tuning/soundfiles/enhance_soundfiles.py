import numpy as np
from scipy.interpolate import CubicSpline

from tuning.common.classes import ClipRange, SoundData
from tuning.soundfiles.utils import create_soundclip_ranges


def reconstruct_clipped_regions(
    data: SoundData, thresh_ratio: float = 0.95, reduce: float = None
) -> SoundData:
    """
    Attempts to reconstruct clipped regions by interpolating the lost signal.
    Clipping mostly occurs when the recording level is set too high.
    Most likely to be effective with lightly clipped audio.
    Args:
        data (np.ndarray): sound data to be fixed. (array of tuples containing amplitude values)
        thresh_ratio (float, optional): Indicates how close to the maximum sample magnitude any sample
                                        must be to be considered clipped. Defaults to 0.95.
        reduce (float, optional): Reduce amplitude to provide headroom for the fixed reconstruction (dB).
                                  None for automatic detection. Defaults to None.

    Returns:
        np.ndarray: the fixed data.
    """
    maxvalue = np.int16(np.iinfo(data.dtype).max)
    clipranges = create_soundclip_ranges(data=data, threshold=maxvalue * thresh_ratio)
    # make a copy of the array and cast to int32 to allow extrapolation outside of the int16 limits
    restored_data = data.copy().astype(np.int32)
    for cliprange in clipranges:
        x_orig = np.array(
            list(range(cliprange.start - 5, cliprange.start))
            + list(range(cliprange.end + 1, cliprange.end + 6))
        )
        y_orig = np.take(data, x_orig, axis=0)
        # function to predict missing values
        interpolated = CubicSpline(x_orig, y_orig)
        # indices to pass through function
        x_restored = list(range(cliprange.start - 5, cliprange.end + 6))
        # new sample values
        restored_data[x_restored[0] : x_restored[-1] + 1] = [interpolated(x) for x in x_restored]

    # scale the data so that it fits within the
    restored_max = abs(restored_data).max()
    restored_data = np.divide(restored_data, restored_max)
    restored_data = np.multiply(restored_data, maxvalue).astype(np.int16)
    return restored_data


def equalize_note_amplitudes(
    sample_rate: float, data: SoundData, clipranges: list[ClipRange]
) -> SoundData:
    """
    Equalizes the amplitude of the individual notes.

    Args:
        sample_rate (float):
        data (SoundDataType): the original sound data
        clipranges (list[ClipRange]): contains the intervals containing the data for each note.

    Returns:
        np.ndarray: equalized data.
    """
    maxvalue = np.int16(np.iinfo(data.dtype).max * 0.95)
    strike_duration = int(sample_rate * 0.05)
    for span in clipranges:
        if span.start + strike_duration < span.end:
            start = span.start + strike_duration
            data[span.start : start] = 0
        else:
            start = span.start
        values = data[start : span.end + 1]
        ratio = maxvalue / np.abs(values).max()
        data[start : span.end + 1] = np.int16(values * ratio)
    return data

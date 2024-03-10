from operator import ge

import numpy as np
import pandas as pd

from tuning.common.classes import ClipRange, SoundData


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

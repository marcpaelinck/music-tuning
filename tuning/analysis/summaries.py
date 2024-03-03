import numpy as np
import pandas as pd

from tuning.common.classes import AggregatedPartialDict, NoteName
from tuning.common.constants import (
    AGGREGATED_PARTIALS_FILE,
    Folder,
    InstrumentGroupName,
)
from tuning.common.utils import (
    db_to_ampl,
    read_group_from_jsonfile,
    read_object_from_jsonfile,
)


def avg_note_freq_per_instrument_type(
    groupname: InstrumentGroupName,
    split_ombak=False,
    ratios=False,
) -> pd.DataFrame:
    """
    Returns a table containing the average frequency of each note over all similar instruments.
    (penumbang and pengisep are)

    Args:
        groupname (InstrumentGroupName): _description_
        split_ombak (bool): penumbang and pengisep instruments are averaged if False.
        ratio (bool): if True, returns ratio of note frequencies compared to the frequency of the first note (DING).

    Returns:
        pd.DataFrame: _description_
    """
    orchestra = read_group_from_jsonfile(groupname, read_sounddata=False, read_spectrumdata=False)
    summary = [
        {
            "type": instr.instrumenttype,
            "code": instr.code,
            "note": note.name,
            "octave": note.octave.index,
            "freq": note.partials[note.partial_index].tone.frequency,
        }
        | ({"ombak": instr.ombaktype.value} if split_ombak else {})
        for instr in orchestra.instruments
        for note in instr.notes
    ]
    summary_df = pd.DataFrame(summary)
    pivot = summary_df.pivot(
        index=["type", "code", "octave"] + (["ombak"] if split_ombak else []),
        columns="note",
        values="freq",
    )
    averages = pivot.groupby(["type", "octave"] + (["ombak"] if split_ombak else [])).aggregate(
        func=np.average
    )
    if ratios:
        averages = averages.div(averages[NoteName.DING], axis=0)
    return averages


def aggregated_partials(groupname: InstrumentGroupName):
    aggr_partials = read_object_from_jsonfile(
        AggregatedPartialDict, groupname, Folder.ANALYSES, AGGREGATED_PARTIALS_FILE
    )
    aggr_dict = [
        {
            "instr": ap.instrument,
            "oct": ap.octave,
            "freq": ap.tone.frequency,
            "ampl_db": ap.tone.amplitude,
            "ampl": db_to_ampl(ap.tone.amplitude),
            "nr_part": len(ap.partials),
            "is_fund": ap.isfundamental,
        }
        for values in aggr_partials.root.values()
        for ap in sorted(
            values, key=lambda p: (p.instrument, p.octave, p.tone.amplitude), reverse=True
        )
    ]
    aggr_df = pd.DataFrame(aggr_dict)
    return aggr_df


if __name__ == "__main__":
    GROUP = InstrumentGroupName.SEMAR_PAGULINGAN
    aggr_df = aggregated_partials(GROUP)
    print(aggr_df)

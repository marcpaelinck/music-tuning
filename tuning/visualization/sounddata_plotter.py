import math
from enum import Enum, auto
from functools import partial
from itertools import chain
from typing import Any, Optional

import numpy as np
from scipy.io import wavfile

from tuning.analysis.aggregation import get_aggr_partials
from tuning.analysis.dissonance import (
    dissonance_j,
    dissonance_profile,
    dissonance_s,
    generate_partial_harmonic_pair,
    reduced_freq_dict,
)
from tuning.analysis.utils import get_minima
from tuning.common.classes import (
    AggregatedPartial,
    AggregatedPartialDict,
    ConsonanceAnalysis,
    Instrument,
    InstrumentGroup,
    InstrumentType,
    Partial,
    Tone,
)
from tuning.common.constants import (
    DEFAULT_NR_HARMONICS_DISS_GRAPH,
    DEFAULT_NR_PARTIALS_DISS_GRAPH,
    Folder,
    InstrumentGroupName,
    NoteName,
    PartialCombiType,
)
from tuning.common.utils import get_logger, get_path
from tuning.visualization.utils import PlotItem, PlotType, create_pdf_new, plot_graphs

logger = get_logger(__name__)


def plot_sounddata(
    group: InstrumentGroup,
    filename: str,
    original_file: bool = True,
) -> bool:
    """
    Creates plots of the sound data with an indication of the detected clippings with note names.

    Args:
        group (InstrumentGroup): The group of instruments to plot.
        filename (str): Name of the PDF file (without path)
        original_file (bool, optional): If True, the original sound files will be used, otherwise
                                        the enhanced files. Defaults to True.
    """

    document = dict()
    document[PlotItem.CONTENT]: DocContent = list()  # type: ignore

    for instrument in group.instruments:
        page = dict()
        document[PlotItem.CONTENT].append(page)
        page[PlotItem.CONTENT]: PageContent = list()  # type: ignore
        page[PlotItem.PAGETITLE] = (
            f"Sound graph for {group.grouptype.value} {instrument.instrumenttype.value} {instrument.code}"
        )

        sample_rate, data = wavfile.read(
            get_path(
                group.grouptype,
                Folder.SOUND,
                instrument.original_soundfilename if original_file else instrument.soundfilename,
            )
        )
        channels = [c for c in range(data.shape[-1])]
        page[PlotItem.XMIN] = 0
        page[PlotItem.XMAX] = math.ceil(len(data) / sample_rate)
        page[PlotItem.YMIN] = ymin = math.floor(np.min(data))
        page[PlotItem.YMAX] = ymax = math.ceil(np.max(data))

        sounddata = [
            ([i / sample_rate for i in range(len(data[:, channel]))], data[:, channel]) for channel in channels
        ]
        page[PlotItem.CONTENT].append({PlotItem.CONTENT: sounddata, PlotItem.PLOTTYPE: PlotType.SOUNDDATAPLOT})

        cliprange_borders = [
            list(
                zip(
                    *[
                        (border / sample_rate, ymin, ymax, "r")
                        for note in instrument.notes
                        for sample in note.samples
                        for border in (sample.cliprange.start, sample.cliprange.end)
                    ]
                )
            )
            for _ in channels
        ]
        plottexts = [
            [
                {
                    "x": (sample.cliprange.start + sample.cliprange.end) / (2 * sample_rate),
                    "y": ymax - 0.1 * (ymax - ymin),
                    "s": note.name.short,
                    "horizontalalignment": "center",
                    "fontsize": "small",
                    "color": "b",
                }
                for note in instrument.notes
                for sample in note.samples
            ],
            [],
        ]
        page[PlotItem.CONTENT].append(
            {PlotItem.CONTENT: cliprange_borders, PlotItem.PLOTTYPE: PlotType.VLINES, PlotItem.PLOTTEXTS: plottexts}
        )

    filepath = get_path(groupname=group.grouptype, folder=Folder.ANALYSES, filename=filename)
    create_pdf_new(document=document, filepath=filepath)


if __name__ == "__main__":
    ...

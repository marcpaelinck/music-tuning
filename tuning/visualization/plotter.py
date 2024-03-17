import math
import os
from enum import Enum, auto
from functools import partial

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tuning.analysis.dissonance import (
    dissonance_j,
    dissonance_s,
    plot_dissonance_graphs,
)
from tuning.common.classes import (
    AggregatedPartialDict,
    Instrument,
    InstrumentGroup,
    InstrumentType,
    Tone,
)
from tuning.common.constants import (
    AGGREGATED_PARTIALS_FILE,
    Folder,
    InstrumentGroupName,
)
from tuning.common.utils import (
    get_logger,
    get_path,
    read_group_from_jsonfile,
    read_object_from_jsonfile,
)
from tuning.visualization.utils import PlotType, create_pdf, plot_graphs

logger = get_logger(__name__)


class Graph(Enum):
    SPECTRUM = auto()
    DISSONANCE = auto()


def plot_dissonance_functions():
    fundfreq = 100
    frequencies = [freq for freq in range(fundfreq, int(2.15 * fundfreq), 1)]
    g_list = [Tone(frequency=freq, amplitude=1) for freq in frequencies]
    f = Tone(frequency=fundfreq, amplitude=1)
    diss_j = [(dissonance_j(f, g)) for g in g_list]
    diss_s = [(dissonance_s(f, g)) for g in g_list]
    plot_graphs(
        (frequencies, diss_j), (frequencies, diss_s), plottype=PlotType.SPECTRUMPLOT, show=True
    )


def plot_note_spectra(
    group: InstrumentGroup,
    object: Instrument,
    *,
    what: list[str] = ["spectra", "partials"],
    max_partials: int = 100,
    ratio: bool = True,  # X-axis is ratios if True, otherwise frequencies
    show: bool = False,
) -> bool:
    xmin = 0
    xmax = max(np.max(note.spectrum.frequencies) for note in object.notes)
    ymin = -100
    ymax = max(np.max(note.spectrum.amplitudes) for note in object.notes)

    plot_content = []
    if "partials" in what:
        xmax = (
            max(
                partial.ratio if ratio else partial.tone.frequency
                for note in object.notes
                for partial in note.partials[:max_partials]
            )
            * 1.05
        )
        partials = [
            list(
                zip(
                    *[
                        (
                            p.ratio if ratio else p.tone.frequency,
                            0,
                            p.tone.amplitude,
                            "r" if p.isfundamental else "y",
                        )
                        for p in note.partials[:max_partials]
                    ]
                )
            )
            for note in object.notes
        ]
        plot_content.append((partials, PlotType.VLINES))

    if "spectra" in what:
        spectra = [(note.spectrum.frequencies, note.spectrum.amplitudes) for note in object.notes]
        if ratio:
            fund_freqs = [note.partials[note.partial_index].tone.frequency for note in object.notes]
            spectra = [
                ([freq / fund_freq for freq in spectrum[0]], spectrum[1])
                for spectrum, fund_freq in zip(spectra, fund_freqs)
            ]
        plot_content.append((spectra, PlotType.SPECTRUMPLOT))

    title = f"{group.grouptype.value} - {object.instrumenttype} {object.code}"

    figure = None
    axes = None
    for content in plot_content:
        is_last = content is plot_content[-1]
        figure, axes = plot_graphs(
            *content[0],
            plottype=content[1],
            pagetitle=title,
            plottitles=[note.name.value for note in object.notes] if is_last else None,
            show=show if is_last else False,
            figure=figure,
            axes=axes,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        )
    return True


if __name__ == "__main__":

    # Set these values before running
    GROUPNAME = InstrumentGroupName.SEMAR_PAGULINGAN
    GRAPH = Graph.DISSONANCE  # SPECTRUM or DISSONANCE

    if GRAPH == Graph.SPECTRUM:
        group = read_group_from_jsonfile(
            GROUPNAME, read_sounddata=False, read_spectrumdata=True, save_spectrumdata=False
        )
        pdf_filename = "spectra_with_partials_ratio.pdf"
        create_pdf(
            group=group,
            iterlist=group.instruments,
            plotter=partial(plot_note_spectra, max_partials=7),
            filepath=get_path(groupname=GROUPNAME, filetype=Folder.ANALYSES, filename=pdf_filename),
        )
    elif GRAPH == Graph.DISSONANCE:
        group = read_object_from_jsonfile(
            AggregatedPartialDict, GROUPNAME, Folder.ANALYSES, AGGREGATED_PARTIALS_FILE
        )
        pdf_filename = "dissonance_graph.pdf"
        create_pdf(
            group=group,
            iterlist=InstrumentType,
            plotter=plot_dissonance_graphs,
            filepath=get_path(groupname=GROUPNAME, filetype=Folder.ANALYSES, filename=pdf_filename),
            groupname=InstrumentGroupName.SEMAR_PAGULINGAN,
        )

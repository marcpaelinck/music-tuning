import math
from enum import Enum, auto
from functools import partial
from typing import Callable, Iterable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel
from scipy.io import wavfile

from tuning.analysis.aggregation import dict_key
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
    read_object_from_jsonfile,
    step_read_group_from_jsonfile,
)
from tuning.visualization.utils import PlotType, create_pdf, plot_graphs

logger = get_logger(__name__)


class Graph(Enum):
    SOUNDFILE = auto()
    SPECTRUM = auto()
    DISSONANCE = auto()


def plot_dissonance_functions():
    fundfreq = 100
    frequencies = [freq for freq in range(fundfreq, int(2.15 * fundfreq), 1)]
    g_list = [Tone(frequency=freq, amplitude=1) for freq in frequencies]
    f = Tone(frequency=fundfreq, amplitude=1)
    diss_j = [(dissonance_j(f, g)) for g in g_list]
    diss_s = [(dissonance_s(f, g)) for g in g_list]
    plot_graphs((frequencies, diss_j), (frequencies, diss_s), plottype=PlotType.SPECTRUMPLOT, show=True)


def plot_note_spectra(
    group: InstrumentGroup,
    object: Instrument,
    *,
    aggregatedpartials: AggregatedPartialDict = AggregatedPartialDict({}),
    what: list[str] = ["spectra", "partials"],
    max_partials: int = 100,
    ratio: bool = True,  # X-axis is ratios if True, otherwise frequencies
    show: bool = False,
) -> bool:
    xmin = 0
    xmax = max(np.max(spectrum.frequencies) for note in object.notes for spectrum in note.spectra)
    ymin = -100
    ymax = max(np.max(spectrum.amplitudes) for note in object.notes for spectrum in note.spectra)

    plot_content = []
    if "partials" in what:
        xmax = (
            max(
                partial.ratio if ratio else partial.tone.frequency
                for note in object.notes
                for partial in note.aggr_partials[:max_partials]
            )
            * 1.05
        )
        note_partials = [
            list(
                zip(
                    *(
                        [
                            (
                                p.ratio if ratio else p.tone.frequency,
                                0,
                                p.tone.amplitude,
                                "r" if p.isfundamental else "y" if note.aggr_partials.index(p) < 4 else "c",
                            )
                            for p in note.aggr_partials[:max_partials]
                        ]
                        + [
                            (ap.ratio if ratio else ap.tone.frequency, 0, ap.tone.amplitude, "r")
                            for ap in aggregatedpartials.root.get(
                                dict_key(object.instrumenttype, object.ombaktype, note.octave), [[]]
                            )[0]
                        ]
                    )
                )
            )
            for note in object.notes
        ]

        plot_content.append((note_partials, PlotType.VLINES))

    if "spectra" in what:
        spectra = [(spectrum.frequencies, spectrum.amplitudes) for note in object.notes for spectrum in note.spectra]
        if ratio:
            fund_freqs = [note.aggr_partials[note.fundamental_index].tone.frequency for note in object.notes]
            spectra = [
                ([freq / fund_freq for freq in spectrum[0]], spectrum[1])
                for spectrum, fund_freq in zip(spectra, fund_freqs)
            ]
        plot_content.append((spectra, PlotType.SPECTRUMPLOT))

    title = f"{group.grouptype.value} - {object.instrumenttype} {object.code}"
    plottexts = [
        [
            {"x": x, "y": y, "s": f"{x:1.3f}", "fontsize": "x-small", "ha": "left"}
            for x, y in zip(partials[0], partials[2])
        ]
        + [
            {
                "x": xmin + 0.01 * (xmax - xmin),
                "y": ymax - 0.1 * (ymax - ymin),
                "s": note.name.value,
                "fontsize": "small",
                "color": "b",
            }
        ]
        for partials, note in zip(note_partials, object.notes)
    ]

    figure = None
    axes = None
    for content in plot_content:
        is_last = content is plot_content[-1]
        figure, axes = plot_graphs(
            *content[0],
            plottype=content[1],
            pagetitle=title,
            plottexts=(plottexts if is_last else None),
            show=show if is_last else False,
            figure=figure,
            axes=axes,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        )
    return True


def plot_sounddata(
    group: InstrumentGroup,
    object: Instrument,
    original_file: bool = True,
    *,
    show: bool = False,
) -> bool:
    sample_rate, data = wavfile.read(
        get_path(
            group.grouptype,
            Folder.SOUND,
            object.original_soundfilename if original_file else object.soundfilename,
        )
    )
    channels = [c for c in range(data.shape[-1])]
    xmin = 0
    xmax = math.ceil(len(data) / sample_rate)
    ymin = math.floor(np.min(data))
    ymax = math.ceil(np.max(data))

    plot_content = []

    sounddata = [([i / sample_rate for i in range(len(data[:, channel]))], data[:, channel]) for channel in channels]
    plot_content.append((sounddata, PlotType.SOUNDDATAPLOT))
    cliprange_borders = [
        list(
            zip(
                *[
                    (border / sample_rate, ymin, ymax, "r")
                    for note in object.notes
                    for sample in note.samples
                    for border in (sample.cliprange.start, sample.cliprange.end)
                ]
            )
        )
        for _ in channels
    ]
    plot_content.append((cliprange_borders, PlotType.VLINES))

    title = f"{group.grouptype.value} - {object.instrumenttype} {object.code}"
    plottexts = [
        [
            {
                "x": (sample.cliprange.start + sample.cliprange.end) / (2 * sample_rate),
                "y": ymax - 0.1 * (ymax - ymin),
                "s": note.name.value,
                "horizontalalignment": "center",
                "fontsize": "small",
                "color": "b",
            }
            for note in object.notes
            for sample in note.samples
        ],
        [],
    ]

    figure = None
    axes = None
    for content in plot_content:
        is_last = content is plot_content[-1]
        figure, axes = plot_graphs(
            *content[0],
            plottype=content[1],
            pagetitle=title,
            plottexts=plottexts,
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
    GRAPH = Graph.SOUNDFILE  # SOUNDFILE, SPECTRUM or DISSONANCE

    if GRAPH == Graph.SOUNDFILE:
        group = step_read_group_from_jsonfile(GROUPNAME, read_sounddata=True)
        pdf_filename = "sounddata.pdf"
        create_pdf(
            group=group,
            iterlist=group.instruments,
            plotter=plot_sounddata,
            filepath=get_path(groupname=GROUPNAME, filetype=Folder.ANALYSES, filename=pdf_filename),
        )
    elif GRAPH == Graph.SPECTRUM:
        group = step_read_group_from_jsonfile(GROUPNAME, read_sounddata=False, read_spectrumdata=True)
        pdf_filename = "spectra_with_partials_ratio.pdf"
        create_pdf(
            group=group,
            iterlist=group.instruments,
            plotter=partial(plot_note_spectra, max_partials=7),
            filepath=get_path(groupname=GROUPNAME, filetype=Folder.ANALYSES, filename=pdf_filename),
        )
    elif GRAPH == Graph.DISSONANCE:
        group = read_object_from_jsonfile(AggregatedPartialDict, GROUPNAME, Folder.ANALYSES, AGGREGATED_PARTIALS_FILE)
        pdf_filename = "dissonance_graph.pdf"
        create_pdf(
            group=group,
            iterlist=InstrumentType,
            plotter=plot_dissonance_graphs,
            filepath=get_path(groupname=GROUPNAME, filetype=Folder.ANALYSES, filename=pdf_filename),
            groupname=InstrumentGroupName.SEMAR_PAGULINGAN,
        )

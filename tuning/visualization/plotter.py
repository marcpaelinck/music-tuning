import math
from enum import Enum, auto
from functools import partial
from itertools import chain
from typing import Callable, Iterable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel
from scipy.io import wavfile

from tuning.analysis.aggregation import dict_key_str, get_aggr_partials
from tuning.analysis.dissonance import (
    dissonance_j,
    dissonance_s,
    plot_dissonance_graphs,
)
from tuning.common.classes import (
    AggregatedPartial,
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
    NoteName,
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


def matchespartial(
    aggr_partial: AggregatedPartial, instr_code: str, notename: NoteName, octave_idx: int, sample_seq: int
) -> bool:
    """
    Determines if an aggregated partial matches a partial of a specific instrument sample.

    Args:
        aggr_partial (AggregatedPartial): _description_
        code (str): instrument code
        sample_seq (int): sequence of the sample for thein the sound file

    Returns:
        bool: _description_
    """
    return any(
        [
            p
            for p in aggr_partial.partials
            if p.instrument_code == instr_code
            and p.notename == notename
            and p.octave_idx == octave_idx
            and p.sample_seq == sample_seq
        ]
    )


def plot_note_spectra(
    group: InstrumentGroup,
    item: Instrument,
    *,
    aggregatedpartials: AggregatedPartialDict = AggregatedPartialDict({}),
    what: list[str] = ["spectra", "partials"],
    max_partials: int = 100,
    ratio: bool = True,  # X-axis is ratios if True, otherwise frequencies
    show: bool = False,
) -> bool:

    instrument = item
    plot_content = []
    logger.info(f"--- generating data for {instrument.instrumenttype.value} {instrument.code}")

    xmin = 0
    # xmax = max(np.max(xval) for plot, _ in plot_content for xval, _ in plot) * 1.05
    xmax = min(
        max([p.ratio for note in instrument.notes for part_list in note.partials[:max_partials] for p in part_list]), 10
    )
    ymin = -100
    ymax = max(np.max(spectrum.amplitudes) for note in instrument.notes for spectrum in note.spectra)

    if "partials" in what:
        # Define vlines for the partials of each spectrum graph and for the aggregated partials
        aggr = partial(get_aggr_partials, aggregatedpartials=aggregatedpartials, instrument=instrument)
        note_partials = [
            [
                [ap.ratio if ratio else ap.tone.frequency for ap in aggr(note)]
                + [p.ratio if ratio else p.tone.frequency for p in partials[:max_partials]],
                [0] * (len(partials[:max_partials]) + len(aggr(note))),
                # [ap.tone.amplitude for ap in aggr(note)] +
                [ymax for ap in aggr(note)] + [p.tone.amplitude for p in partials[:max_partials]],
                [
                    "m" if matchespartial(ap, instrument.code, note.name, note.octave.index, sample_seq) else "y"
                    for ap in aggr(note)
                ]
                + ["r" if p.isfundamental else "c" for p in partials[:max_partials]],
            ]
            for note in instrument.notes
            for partials, sample_seq in zip(note.partials, list(range(len(note.partials))))
        ]

    if "spectra" in what:
        fund_list = {
            note.name: [next(p for p in p_list if p.isfundamental) for p_list in note.partials]
            for note in instrument.notes
        }
        spectra = {
            note.name: [(spectrum.frequencies, spectrum.amplitudes) for spectrum in note.spectra]
            for note in instrument.notes
        }
        if ratio:
            spectra = {
                notename: [
                    ([freq / fundamental.tone.frequency for freq in frequencies], amplitudes)
                    for (frequencies, amplitudes), fundamental in zip(spectra[notename], fund_list[notename])
                ]
                for notename in spectra.keys()
            }
        plot_content.append((list(chain.from_iterable(spectra.values())), PlotType.SPECTRUMPLOT))

        plot_content.append((note_partials, PlotType.VLINES))

    title = (
        f"{group.grouptype.value} - {instrument.instrumenttype.value} {instrument.ombaktype.value} {instrument.code}"
    )
    plottexts = [
        [
            {"x": p.ratio, "y": p.tone.amplitude, "s": f"{p.ratio:1.3f}", "fontsize": "x-small", "ha": "left"}
            for p in partials[:max_partials]
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
        # for partials, note in zip(note_partials, object.notes)
        for note in instrument.notes
        for partials in note.partials
    ]

    figure = None
    axes = None
    for content in plot_content:
        is_last = content is plot_content[-1]
        if content is plot_content[0]:
            logger.info(f"--- Creating {len(content[0])} {content[1].name} graphs")
        else:
            logger.info(f"--- Adding {content[1].name} to {len(content[0])} graphs")
        figure, axes = plot_graphs(
            *content[0],
            plottype=content[1],
            pagetitle=title,
            plottexts=(plottexts if is_last else None),
            show=False,
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
    item: Instrument,
    original_file: bool = True,
    *,
    show: bool = False,
) -> bool:
    sample_rate, data = wavfile.read(
        get_path(
            group.grouptype,
            Folder.SOUND,
            item.original_soundfilename if original_file else item.soundfilename,
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
                    for note in item.notes
                    for sample in note.samples
                    for border in (sample.cliprange.start, sample.cliprange.end)
                ]
            )
        )
        for _ in channels
    ]
    plot_content.append((cliprange_borders, PlotType.VLINES))

    title = f"{group.grouptype.value} - {item.instrumenttype} {item.code}"
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
            for note in item.notes
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
    ...

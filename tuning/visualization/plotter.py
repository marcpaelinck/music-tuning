import math
from enum import Enum, auto
from functools import partial
from itertools import chain

import numpy as np
from scipy.io import wavfile

from tuning.analysis.aggregation import dict_key_retrieve_instrgroup, get_aggr_partials
from tuning.analysis.dissonance import (
    create_partials_from_list,
    dissonance_j,
    dissonance_profile,
    dissonance_s,
    reduced_freq_dict,
)
from tuning.analysis.utils import get_minima
from tuning.common.classes import (
    AggregatedPartial,
    AggregatedPartialDict,
    Instrument,
    InstrumentGroup,
    InstrumentType,
    Tone,
)
from tuning.common.constants import Folder, InstrumentGroupName, NoteName
from tuning.common.utils import get_logger, get_path
from tuning.visualization.utils import PlotType, plot_graphs

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


def plot_dissonance_graphs(
    group: AggregatedPartialDict,
    *,
    item: InstrumentType,
    groupname: InstrumentGroupName,
    amplitude: float = None,
    ref_octave: int = 3,
    ref_note: NoteName = NoteName.DING,
    **kwargs,
):
    instrumenttype = item
    HARMONICS_AMPL_DB = 50  # Amplitude for harmonics
    # TODO retrieving dict entry by using string literal is error prone
    # Key is a string with format "instrumenttype#ombak-octave"
    keys = [key for key in list(group.root.keys()) if key.startswith(instrumenttype.value)]
    if not keys:
        return False

    # Calculate the average note frequencies for the reference octave (octave 0)
    notes = [note for key in keys for note in group.root[key][1]]
    reduced_freqs = reduced_freq_dict(notes, ref_note, group, instrumenttype)

    partial_harmonic_pairs = dict()
    for key in keys:
        instrument_group = dict_key_retrieve_instrgroup(key)
        partials, notes = group.root[key]
        # fundamental = next((p for p in partials if p.isfundamental), None)
        # reference_freq = fundamental.tone.frequency
        reference_freq = reduced_freqs[ref_note] * pow(2, ref_octave)
        for partial in partials:
            partial.tone.frequency = partial.ratio * reference_freq
        harmonics = create_partials_from_list([(reference_freq * i, HARMONICS_AMPL_DB) for i in range(1, 5)])
        partial_harmonic_pairs[instrument_group] = (partials, harmonics)

    pagetitle = f"Dissonance graph for {groupname.value} {instrumenttype.value}" + (
        " (same amplitude for all partials)" if amplitude else ""
    )
    # Reduce all notes to octave 0 then calculate average frequency per note.
    # Then determine ratio relative to DING.
    noteratio_dict = {notename: freq / reduced_freqs[ref_note] for notename, freq in reduced_freqs.items()}

    profiler = dissonance_profile
    plot_content = list()
    plot_texts = list()

    diss_graphs = list()
    diss_graph_texts = list()
    diss_graph_titles = list()
    for group, pair in partial_harmonic_pairs.items():
        pp_result = profiler(
            pair[1],
            pair[0],
            fixed_ampl=kwargs.get("amplitude", None),
            is_reverse=True,
        )
        ph_result = profiler(
            pair[0],
            pair[1],
            fixed_ampl=kwargs.get("amplitude", None),
        )
        pp_peaks = get_minima(*pp_result)
        ph_peaks = get_minima(*ph_result)
        diss_graphs.extend([pp_result + [pp_peaks], ph_result + [ph_peaks]])
        diss_graph_titles.extend([group + " - partial-partial", group + " - partial-harmonic"])

    ymin = min(min(result[1]) for result in diss_graphs)
    ymax = max(max(result[1]) for result in diss_graphs)

    vlines_graph = list()
    vlines_text = list()
    for graph_data, title in zip(diss_graphs, diss_graph_titles):
        diss_graph_texts.append(
            [
                {
                    "x": x,
                    "y": y,
                    "s": f"{abs(x):1.3f}",
                    "fontsize": "x-small",
                    "ha": "center",
                    "va": "top",
                }
                for x, y in graph_data[2]
            ]
            + [
                {
                    "x": min(graph_data[0]) + 0.01 * (max(graph_data[0]) - min(graph_data[0])),
                    "y": ymax - 0.05 * (ymax - ymin),
                    "s": title,
                    "fontsize": "small",
                    "color": "b",
                }
            ]
        )
        sign = graph_data[0][0] / abs(graph_data[0][0])
        notenames = list(noteratio_dict.keys())
        vlines_graph.append(
            [
                list(np.array(list(noteratio_dict.values())) * sign),
                [0] * len(noteratio_dict.values()),
                [ymax] * len(noteratio_dict.values()),
                ["r" if notename is ref_note.kempyung(notenames) else "y" for notename in noteratio_dict.keys()],
            ]
        )
        vlines_text.append(
            [
                {
                    "x": sign * ratio,
                    "y": ymax - 0.1,
                    "s": notename.short,
                    "fontsize": "small",
                    "color": "r" if notename is ref_note.kempyung(notenames) else "y",
                    "ha": "center",
                    "va": "bottom",
                }
                for notename, ratio in noteratio_dict.items()
            ]
        )
    plot_content.append((diss_graphs, PlotType.CONSONANCEPLOT))
    plot_texts.append(diss_graph_texts)
    plot_content.append((vlines_graph, PlotType.VLINES))
    plot_texts.append(vlines_text)

    figure = None
    axes = None
    for content, texts in zip(plot_content, plot_texts):
        figure, axes = plot_graphs(
            *content[0],
            plottype=content[1],
            nrows=len(diss_graphs) // 2,
            ncols=2,
            pagetitle=pagetitle,
            plottitles=instrument_group,
            plottexts=texts,
            autox=True,
            figure=figure,
            axes=axes,
            ymin=ymin,
            ymax=ymax,
            **kwargs,
        )


if __name__ == "__main__":
    ...

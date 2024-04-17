from functools import partial
from itertools import chain
from typing import Optional

import numpy as np

from tuning.analysis.aggregation import get_aggr_partials
from tuning.common.classes import (
    AggregatedPartial,
    AggregatedPartialDict,
    ConsonanceAnalysis,
    Instrument,
    InstrumentGroup,
    Note,
)
from tuning.common.constants import Folder, NoteName
from tuning.common.utils import get_logger, get_path
from tuning.visualization.utils import (
    DocContent,
    PageContent,
    PlotComponent,
    PlotItem,
    PlotType,
    create_pdf_new,
)

logger = get_logger(__name__)


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
    *,
    aggregatedpartials: Optional[AggregatedPartialDict] = AggregatedPartialDict(root={}),
    filename: str,
    analysis: Optional[ConsonanceAnalysis] = None,
    what: list[str] = ["spectra", "aggregatedpartials"],
    max_partials: int = 100,
    ratio: bool = True,  # X-axis is ratios if True, otherwise frequencies
    pageby: type = Instrument,  # Instrument or Note
    **kwargs,
):

    document = dict()
    document[PlotItem.CONTENT]: DocContent = list()  # type: ignore

    # Option to paginate by instrument or by instrument/note combination
    if pageby is Instrument:
        groupby = [(instrument, None) for instrument in group.instruments]
    elif pageby is Note:
        groupby = [(instrument, note) for instrument in group.instruments for note in instrument.notes]
    else:
        raise NotImplementedError(f"Cannot group by {pageby.__name__}")

    for instrument, note in groupby:
        notelist = [note] if note else instrument.notes

        page = dict()
        document[PlotItem.CONTENT].append(page)
        page[PlotItem.CONTENT]: PageContent = list()  # type: ignore
        page[PlotItem.PAGETITLE] = (
            f"{group.grouptype.value} - {instrument.instrumenttype.value} {instrument.ombaktype.value} {instrument.code} {note.name if note else ""}"
        )

        logger.info(f"--- generating data for {instrument.instrumenttype.value} {instrument.code}")

        page[PlotItem.XMIN] = 0
        page[PlotItem.XMAX] = min(
            max(
                [p.ratio for note in notelist for part_list in note.partials[:max_partials] for p in part_list]
            ),
            10,
        )
        page[PlotItem.YMIN] = -100
        page[PlotItem.YMAX] = max(np.max(spectrum.amplitudes) for note in notelist for spectrum in note.spectra)

        if "aggregatedpartials" in what:
            # Define vlines for the partials of each spectrum graph and for the aggregated partials
            aggr = partial(get_aggr_partials, aggregatedpartials=aggregatedpartials, instrument=instrument)
            note_partials = [
                [
                    [ap.ratio if ratio else ap.tone.frequency for ap in aggr(note)]
                    + [p.ratio if ratio else p.tone.frequency for p in partials[:max_partials]],
                    [0] * (len(partials[:max_partials]) + len(aggr(note))),
                    # [ap.tone.amplitude for ap in aggr(note)] +
                    [page[PlotItem.YMAX] for ap in aggr(note)] + [p.tone.amplitude for p in partials[:max_partials]],
                    [
                        "m" if matchespartial(ap, instrument.code, note.name, note.octave.index, sample_seq) else "y"
                        for ap in aggr(note)
                    ]
                    + ["r" if p.isfundamental else "c" for p in partials[:max_partials]],
                ]
                for note in notelist
                for partials, sample_seq in zip(note.partials, list(range(len(note.partials))))
            ]

        if "analysis-partials" in what:
            # Define vlines for the given partials
            note_partials = [
                [
                    xvals := [
                        p.ratio for p in analysis.optimal_consonances.partials + analysis.optimal_consonances.harmonics
                    ],
                    [0] * len(xvals),
                    [
                        p.tone.amplitude
                        for p in analysis.optimal_consonances.partials + analysis.optimal_consonances.harmonics
                    ],
                    ["m"] * len(analysis.optimal_consonances.partials)
                    + ["y"] * len(analysis.optimal_consonances.harmonics),
                ]
                for note in notelist
                for partials, sample_seq in zip(note.partials, list(range(len(note.partials))))
            ]

        content: PlotComponent = {PlotItem.CONTENT: note_partials, PlotItem.PLOTTYPE: PlotType.VLINES}
        page[PlotItem.CONTENT].append(content)

        if "spectra" in what:
            fund_list = {
                note.name: [next(p for p in p_list if p.isfundamental) for p_list in note.partials]
                for note in notelist
            }
            spectra = {
                note.name: [(spectrum.frequencies, spectrum.amplitudes) for spectrum in note.spectra]
                for note in notelist
            }
            if ratio:
                spectra = {
                    notename: [
                        ([freq / fundamental.tone.frequency for freq in frequencies], amplitudes)
                        for (frequencies, amplitudes), fundamental in zip(spectra[notename], fund_list[notename])
                    ]
                    for notename in spectra.keys()
                }
        plottexts = [
            [
                {"x": p.ratio, "y": p.tone.amplitude, "s": f"{p.ratio:1.3f}", "fontsize": "x-small", "ha": "left"}
                for p in partials[:max_partials]
            ]
            + [
                {
                    "x": page[PlotItem.XMIN] + 0.01 * (page[PlotItem.XMAX] - page[PlotItem.XMIN]),
                    "y": page[PlotItem.YMAX] - 0.1 * (page[PlotItem.YMAX] - page[PlotItem.YMIN]),
                    "s": note.name.value,
                    "fontsize": "small",
                    "color": "b",
                }
            ]
            for note in notelist
            for partials in note.partials
        ]
        content: PlotComponent = {
            PlotItem.CONTENT: list(chain.from_iterable(spectra.values())),
            PlotItem.PLOTTYPE: PlotType.SPECTRUMPLOT,
            PlotItem.PLOTTEXTS: plottexts,
        }
        page[PlotItem.CONTENT].append(content)

    filepath = get_path(groupname=group.grouptype, folder=Folder.ANALYSES, filename=filename)
    create_pdf_new(document=document, filepath=filepath)

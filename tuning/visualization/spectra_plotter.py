from functools import partial
from itertools import count
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
from tuning.common.constants import Folder, NoteName, PageSizes
from tuning.common.utils import get_logger, get_path
from tuning.visualization.utils import (
    ComponentObject,
    ComponentType,
    DocumentObject,
    GraphObject,
    GroupName,
    PageObject,
    PltProperty,
    create_pdf_with_plots,
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


def step_plot_note_spectra(
    group: InstrumentGroup,
    *,
    aggregatedpartials: Optional[AggregatedPartialDict],
    filename: str,
    analysis: Optional[ConsonanceAnalysis] = None,
    max_partials: int = 100,
    ratio: bool = True,  # X-axis is ratios if True, otherwise frequencies
    pageby: type = Instrument,  # Instrument or Note
    skip: list[GroupName] = list(),
):
    logger.info(f"Creating plots of the frequency spectra and saving them to file {filename}")

    document = DocumentObject(id=1)
    document.properties[PltProperty.SKIP] = skip

    # Option to paginate by instrument or by instrument/note combination
    if pageby is Instrument:
        groupby = [(instrument, None) for instrument in group.instruments]
    elif pageby is Note:
        groupby = [(instrument, note) for instrument in group.instruments for note in instrument.notes]
    else:
        raise NotImplementedError(f"Cannot group by {pageby.__name__}")

    for instrument, note in groupby:
        logger.info(f"--- generating data for {instrument.instrumenttype.value} {instrument.code}")
        notelist = [note] if note else instrument.notes

        document.add_child(page := PageObject())
        page.properties[PltProperty.PAGESIZE] = PageSizes.A4
        page.properties[PltProperty.TITLE] = (
            f"{group.grouptype.value} - {instrument.instrumenttype.value} {instrument.ombaktype.value} {instrument.code} {note.name if note else ' '}"
        )
        page.properties[PltProperty.XMIN] = 0
        page.properties[PltProperty.XMAX] = max(
            [p.ratio for note in notelist for part_list in note.partials[:max_partials] for p in part_list]
        )
        page.properties[PltProperty.YMIN] = -10
        page.properties[PltProperty.YMAX] = max(
            np.max(spectrum.amplitudes) for note in notelist for spectrum in note.spectra
        )
        if aggregatedpartials:
            aggr = partial(get_aggr_partials, aggregatedpartials=aggregatedpartials, instrument=instrument)
            page.properties[PltProperty.XMAX] = max(
                page.properties[PltProperty.XMAX],
                max([ap.ratio if ratio else ap.tone.frequency for ap in aggr(note) for note in notelist]),
            )
        if analysis:
            page.properties[PltProperty.XMAX] = max(
                page.properties[PltProperty.XMAX],
                max([p.ratio for p in analysis.optimal_consonances.partials + analysis.optimal_consonances.harmonics]),
            )

        for note in notelist:
            for spectrum, partials, sample_seq in zip(note.spectra, note.partials, count()):
                page.add_child(graph := GraphObject())

                # Draw spectrum plots
                fundamental = next(p for p in partials if p.isfundamental)
                xy_values = (
                    [np.array(spectrum.frequencies) / fundamental.tone.frequency, spectrum.amplitudes]
                    if ratio
                    else [spectrum.frequencies, spectrum.amplitudes]
                )
                max_idx = np.argmax(xy_values[0] > page.properties[PltProperty.XMAX] * 1.05) or len(xy_values[0] - 1)
                xy_values = [xy_values[0][:max_idx], xy_values[1][:max_idx]]
                graph.add_child(component := ComponentObject(groups=[GroupName.SPECTRA_PLOT]))
                component.properties[PltProperty.COMPONENTTYPE] = ComponentType.LINE
                component.content.extend(xy_values)

                # Add Vlines for the first max_partials partials
                note_partials = [
                    [p.ratio if ratio else p.tone.frequency for p in partials[:max_partials]],
                    [page.properties[PltProperty.YMIN]] * len(partials[:max_partials]),
                    # [ap.tone.amplitude for ap in aggr(note)] +
                    [p.tone.amplitude for p in partials[:max_partials]],
                    ["r" if p.isfundamental else "c" for p in partials[:max_partials]],
                ]
                graph.add_child(component := ComponentObject(groups=[GroupName.SPECTRA_PARTIALS]))
                component.properties[PltProperty.COMPONENTTYPE] = ComponentType.VLINES
                component.content.extend(note_partials)

                # Write values for the note partials
                graph.add_child(component := ComponentObject(groups=[GroupName.SPECTRA_PARTIALS]))
                component.properties[PltProperty.COMPONENTTYPE] = ComponentType.TEXT
                component.content = [
                    {
                        "x": p.ratio,
                        "y": p.tone.amplitude,
                        "s": f"{p.ratio:1.3f}",
                        "fontsize": "x-small",
                        "ha": "left",
                    }
                    for p in partials[:max_partials]
                ]

                # Write graph title (=note name)
                graph.add_child(component := ComponentObject(groups=[GroupName.GRAPH_TITLE]))
                component.properties[PltProperty.COMPONENTTYPE] = ComponentType.TEXT
                component.content = [
                    {
                        "x": page.properties[PltProperty.XMIN]
                        + 0.01 * (page.properties[PltProperty.XMAX] - page.properties[PltProperty.XMIN]),
                        "y": page.properties[PltProperty.YMAX]
                        - 0.1 * (page.properties[PltProperty.YMAX] - page.properties[PltProperty.YMIN]),
                        "s": note.name.value,
                        "fontsize": "small",
                        "color": "b",
                    }
                ]

                if aggregatedpartials:
                    # Define vlines for the aggregated partials
                    aggr_partials = np.array(
                        get_aggr_partials(note, aggregatedpartials=aggregatedpartials, instrument=instrument)
                    )
                    # Determine which aggregated partials match with partials for the current note.
                    match_ind = np.array(
                        [
                            matchespartial(ap, instrument.code, note.name, note.octave.index, sample_seq)
                            for ap in aggr_partials
                        ]
                    )
                    matches = aggr_partials[match_ind]
                    nonmatches = aggr_partials[~match_ind]
                    for p_list, color, text in zip((matches, nonmatches), ("m", "y"), ("with", "without")):
                        aggr_partials = [
                            [ap.ratio if ratio else ap.tone.frequency for ap in p_list],
                            # [page.properties[PlotItem.YMIN]] * len(p_list),
                            [ap.tone.amplitude for ap in p_list],
                            [page.properties[PltProperty.YMAX]] * len(p_list),
                            [color] * len(p_list),
                        ]
                        if any(aggr_partials):
                            graph.add_child(component := ComponentObject(groups=[GroupName.SPECTRA_AGGR_PARTIALS]))
                            component.properties[PltProperty.COMPONENTTYPE] = ComponentType.VLINES
                            component.properties[PltProperty.LABEL] = (
                                f"Aggr. partial {text} matching partial for this note."
                            )
                            component.content.extend(aggr_partials)

                if analysis:
                    # Define vlines for the given partials
                    component.content = [
                        xvals := [
                            p.ratio
                            for p in analysis.optimal_consonances.partials + analysis.optimal_consonances.harmonics
                        ],
                        [0] * len(xvals),
                        [
                            p.tone.amplitude
                            for p in analysis.optimal_consonances.partials + analysis.optimal_consonances.harmonics
                        ],
                        ["m"] * len(analysis.optimal_consonances.partials)
                        + ["y"] * len(analysis.optimal_consonances.harmonics),
                    ]
                    graph.add_child(component := ComponentObject(groups=[GroupName.SPECTRA_ANALYSIS_PARTIALS]))
                    component.properties[PltProperty.COMPONENTTYPE] = ComponentType.VLINES
                    component.content.extend(aggr_partials)

        page.properties[PltProperty.NBR_ROWS] = len(page.content)
        page.properties[PltProperty.NBR_COLUMNS] = 1

    filepath = get_path(groupname=group.grouptype, folder=Folder.ANALYSES_SPECTRA, filename=filename)
    create_pdf_with_plots(document=document, filepath=filepath)

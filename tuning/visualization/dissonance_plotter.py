import math
from itertools import chain
from xmlrpc.client import MAXINT

import numpy as np

from tuning.analysis.dissonance import (
    dissonance_profile,
    generate_partial_harmonic_pair,
    get_noteratio_dict,
)
from tuning.analysis.utils import get_minima
from tuning.common.classes import (
    AggregatedPartialDict,
    ConsonanceAnalysis,
    Frequency,
    InstrumentType,
)
from tuning.common.constants import (
    DEFAULT_NR_HARMONICS_DISS_GRAPH,
    DEFAULT_NR_PARTIALS_DISS_GRAPH,
    Folder,
    InstrumentGroupName,
    NoteName,
    PageSizes,
    PartialCombiType,
)
from tuning.common.utils import get_logger, get_path, step_save_object_to_jsonfile
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


def get_line_format(
    combi: PartialCombiType, item: PltProperty, nr_partials: int = None, nr_harmonics: int = None
) -> str:
    """
    Returns formatting components for a specific dissonance curve

    Args:
        combi (PartialCombiType): The combination of partials/harmonics
        item (PlotItem): The formatting item to return
        nr_partials (int): Number of partials, only needed for the title. Default is None.
        nr_harmonics (int): Number of harmonics, only needed for the title. Default is None.

    Returns:
        str: The requested formatting item.
    """
    match combi:
        case PartialCombiType.PARTIALS_PARTIALS:
            return {
                PltProperty.COLOR: "b",
                PltProperty.LABEL: f"partials[{nr_partials}]-partials[{nr_partials}]",
            }.get(item, None)
        case PartialCombiType.PARTIALS_HARMONICS:
            return {
                PltProperty.COLOR: "g",
                PltProperty.LABEL: f"partials[{nr_partials}]-harmonics[{nr_harmonics}]",
            }.get(item, None)
        case PartialCombiType.HARMONICS_PARTIALS:
            return {
                PltProperty.COLOR: "c",
                PltProperty.LABEL: f"harmonics[{nr_harmonics}]-partials[{nr_partials}]",
            }.get(item, None)
    return None


def get_stat(
    stat: callable,
    diss_results: dict[str, dict[PartialCombiType, list[Frequency], list[float]]],
    index: int,
    keys: list[str],
    combi_filter: list[PartialCombiType] = PartialCombiType,
):
    """Returns an aggregation result for the given combination of aggregation keys, partial combination and instrument type.
    Args:
        stat (callable): Aggregation function.
        diss_results (_type_): Dissonance plot values.
        index (_type_): Indicates which values to use. 0 = frequencies, 1 = dissonance.
        keys (list[str]): List of aggregation keys over which to iterate.

    Returns:
        _type_: _description_
    """
    return stat(chain.from_iterable([diss_results[key][combi][index] for key in keys for combi in combi_filter]))


def generate_dissonance_data_optimized(analysis_results: ConsonanceAnalysis, end_shift_ratio: Frequency = 2.5):
    if analysis_results:
        # Retrieve the combinations of partials and harmonics for each aggregation group
        key = InstrumentType.ALL.value
        partials = analysis_results.optimal_consonances.partials
        harmonics = analysis_results.optimal_consonances.harmonics
        noteratio_dict = {InstrumentType.ALL.value: analysis_results.noteratio_dict}
        ref_note = analysis_results.ref_note
        ref_octave = analysis_results.ref_octave
        partial_harmonic_pairs = {
            PartialCombiType.PARTIALS_PARTIALS: (partials, partials),
            PartialCombiType.PARTIALS_HARMONICS: (partials, harmonics),
            PartialCombiType.HARMONICS_PARTIALS: (harmonics, partials),
        }
        fundamental = next((partial for partial in partials if partial.isfundamental), None)
        diss_results = {key: dissonance_profile(partial_harmonic_pairs, fundamental, end_shift_ratio=end_shift_ratio)}
    return (
        noteratio_dict,
        partials,
        harmonics,
        ref_note,
        ref_octave,
        diss_results,
    )


def generate_dissonance_data(
    group: AggregatedPartialDict,
    nr_partials: int = DEFAULT_NR_PARTIALS_DISS_GRAPH,
    nr_harmonics: int = DEFAULT_NR_HARMONICS_DISS_GRAPH,
    ref_note: NoteName = NoteName.DING,
    ref_octave: int = 3,
    end_shift_ratio: int = 2.5,  # indicates the end frequency of the shifted frequency relative to the fixed fundamental.
):
    # Generate the requested combinations of partials and harmonics for each aggregation group
    # First calculate the average note frequencies for the reference octave (octave 0)
    noteratio_dict = dict()
    diss_results = dict()

    for key, (agg_partials, notes_oct_0) in group.root.items():
        refnote_oct_0 = notes_oct_0.get(ref_note, None) or group.notes_octave_0[ref_note]
        reference_frequency = refnote_oct_0.frequency * pow(2, ref_octave)
        # Then determine the ratio of each note relative to the reference note and retrieve the partials.
        partials, harmonics = generate_partial_harmonic_pair(
            agg_partials,
            nbr_partials=nr_partials,
            nbr_harmonics=nr_harmonics,
            reference_frequency=reference_frequency,
        )
        partial_harmonic_pairs = {
            PartialCombiType.PARTIALS_PARTIALS: (partials, partials),
            PartialCombiType.PARTIALS_HARMONICS: (partials, harmonics),
            PartialCombiType.HARMONICS_PARTIALS: (harmonics, partials),
        }
        fundamental = next((partial for partial in partials if partial.isfundamental), None)
        diss_results[key] = dissonance_profile(partial_harmonic_pairs, fundamental, end_shift_ratio=end_shift_ratio)

        # list of note frequencies relative to the referenc note
        note_dict = {
            notename: (notes_oct_0.get(notename, None) or group.notes_octave_0[notename])
            for notename in group.notes_octave_0.keys()
        }
        noteratio_dict[key] = get_noteratio_dict(note_dict=note_dict, reference_note=note_dict[ref_note])

    return (
        noteratio_dict,
        partials,
        harmonics,
        ref_note,
        ref_octave,
        diss_results,
    )


def step_plot_dissonance_graphs(
    *,
    groupname: InstrumentGroupName,
    filename: str,
    generator: callable = create_pdf_with_plots,
    analysis_results: ConsonanceAnalysis = None,
    # All following arguments will be ignored if analysis_results is given.
    aggr_partials: AggregatedPartialDict = None,
    nr_partials: int = DEFAULT_NR_PARTIALS_DISS_GRAPH,
    nr_harmonics: int = DEFAULT_NR_HARMONICS_DISS_GRAPH,
    amplitude: float = None,
    ref_note: NoteName = NoteName.DING,
    ref_octave: int = 3,
    end_shift_ratio: float = 2.5,
    include_only: list[GroupName] = list(),
    skip: list[GroupName] = list(),
):
    """
    Generates dissonance plot data using aggregated partials as input, and calls generator to process the data.
    For an explanation of the structure that is built in this method, please read the comment
    of method :func:`utils.create_pdf_with_plots`.
    Note: Each page will contain the graphs for one instrument type
    Args:
        groupname (InstrumentGroupName): The instrument group.
        filename (str): Name for the PDF file that will contain the output (without path component).
        aggr_partials (AggregatedPartialDict): Aggregated partials from which partials will be selected
                        to create the graphs.
        analysis_results (ConsonanceAnalysis, optional): Alternative input for the list of partials to use.
                        This structure contains a fixed set of partials and harmonics that should be used
                        for the consonance calculation. Parameter aggr_partials and all following
                        parameters will be ignored if this value is passed. Defaults to None.
        nr_partials (int, optional): Number of partials to select. The first `nr_partials` partials will be
                        selected from the given list of partials. Defaults to DEFAULT_NR_PARTIALS_DISS_GRAPH.
        nr_harmonics (int, optional): Number of harmonics to create, based on the reference note.
                        Defaults to DEFAULT_NR_HARMONICS_DISS_GRAPH.
        amplitude (float, optional): Default amplitude to use for all partials. This overrides the individual
                        amplitudes. Defaults to None.
        ref_note (NoteName, optional): Note that will be used to provide the frequency of the fundamental of
                        the frequencies that remain fixed in the consonance ComponentObject(). Defaults to NoteName.DING.
        ref_octave (int, optional): Octave sequence of the reference note. Defaults to 3.
        end_shift_ratio (float): Indicates the end frequency of the shifted fundamental relative to the fixed fundamental.
        skip (list[GroupName], optional): List of groups of objects that should not be included in the plot(s).
        include_only (list[GroupName], optional): List of groups of objects that should be included in the plot.
                        Other objects will be excluded. The skip parameter can be used to exclude subgroups of
                        the included object groups.
    """
    logger.info("Creating dissonance graphs")

    #
    # Generate the data for the graphs
    #
    if analysis_results:
        # Retrieve the combinations of partials and harmonics for each aggregation group
        (noteratio_dict, partials, harmonics, ref_note, ref_octave, diss_results) = generate_dissonance_data_optimized(
            analysis_results, end_shift_ratio=end_shift_ratio
        )
        nr_partials = len(partials)
        nr_harmonics = len(harmonics)
        aggregation = [InstrumentType.ALL.value]
        logger.info(f"Graphs will be based on optimal average partial combinations over all instruments")
    else:
        # Generate the requested combinations of partials and harmonics for each aggregation group
        # First calculate the average note frequencies for the reference octave (octave 0)
        (noteratio_dict, partials, harmonics, ref_note, ref_octave, diss_results) = generate_dissonance_data(
            aggr_partials, nr_partials, nr_harmonics, ref_note, ref_octave, end_shift_ratio=end_shift_ratio
        )
        aggregation = aggr_partials.aggregation
        logger.info(
            f"Data is aggregated {'by [' + ', '.join(aggr_partials.aggregation) + ']' if aggr_partials.aggregation else 'over all instruments'}"
        )

    #
    # Create the graphs
    #
    document = DocumentObject()
    document.include_only_list = include_only
    document.skip_list = skip

    for instrumenttype in InstrumentType:
        # TODO retrieving dict entry by using string literal is error prone
        # Key is a string with format "instrumenttype#ombak-octave"
        keys = [key for key in diss_results.keys() if key.startswith(instrumenttype.value)]
        if not keys:
            # No instrument for this key
            continue

        logger.info(f"--- generating data for {instrumenttype.value}")

        # Generate the data for the graphs and set values if necessary

        document.add_child(page := PageObject())
        page.properties[PltProperty.PAGESIZE] = PageSizes.A4.value
        page.properties[PltProperty.TITLE] = f"Dissonance graph for {groupname.value} {instrumenttype.value}" + (
            " (same amplitude for all partials)" if amplitude else ""
        )
        (x_min, x_max) = page.properties[PltProperty.XRANGE] = (
            0.95 * get_stat(min, diss_results, 0, keys),
            1.05 * get_stat(max, diss_results, 0, keys),
        )
        (y_min, y_max) = page.properties[PltProperty.YRANGE] = (
            0.95 * get_stat(min, diss_results, 1, keys),
            1.05 * get_stat(max, diss_results, 1, keys),
        )

        for subgroup in keys:
            graph_groups = {GroupName.DISSONANCE_GRAPH}
            if len(page.children) == 0:
                graph_groups.add(GroupName.DISSONANCE_ANIMATION_GRAPH)
                page.properties[PltProperty.PAGESIZE] = PageSizes.ANIMATION.value
                filter = [PartialCombiType.PARTIALS_HARMONICS]
                (y_min, y_max) = page.properties[PltProperty.YRANGE] = (
                    0.95 * get_stat(min, diss_results, 1, keys, filter),
                    1.05 * get_stat(max, diss_results, 1, keys, filter),
                )
            page.add_child(graph := GraphObject(groups=graph_groups))
            diss_result = diss_results[subgroup]

            # Add the dissonance line graphs
            for combi in PartialCombiType:
                # Add a line
                componentgroup = GroupName.from_enum(combi)
                graph.add_child(component := ComponentObject(groups={GroupName.DISSONANCE_LINE, componentgroup}))
                component.properties[PltProperty.COMPONENTTYPE] = ComponentType.LINE
                component.content = diss_result[combi]
                component.properties[PltProperty.COLOR] = get_line_format(combi, PltProperty.COLOR)
                component.properties[PltProperty.LABEL] = get_line_format(
                    combi, PltProperty.LABEL, nr_partials, nr_harmonics
                )

                # add values of minima
                graph.add_child(
                    component := ComponentObject(
                        groups={GroupName.DISSONANCE_TEXT, GroupName.DISSONANCE_TEXT_MINIMA, componentgroup}
                    )
                )
                component.properties[PltProperty.COMPONENTTYPE] = ComponentType.TEXT
                component.content = [
                    {
                        "x": dissonance.ratio,
                        "y": dissonance.dissonance,
                        "s": f"{abs(dissonance.ratio):1.3f}",
                        "fontsize": "x-small",
                        "ha": "center",
                        "va": "top",
                    }
                    for dissonance in get_minima(*diss_result[combi])
                ]
                diss = sorted(get_minima(*diss_result[combi]), key=lambda d: d.ratio)

            # Add graph title
            graph.add_child(component := ComponentObject(groups={GroupName.DISSONANCE_TEXT, GroupName.GRAPH_TITLE}))
            component.properties[PltProperty.COMPONENTTYPE] = ComponentType.TEXT
            component.content.append(
                {
                    "x": x_min + 0.01 * (x_max - x_min),
                    "y": y_max - 0.05 * (y_max - y_min),
                    "s": subgroup,
                    "fontsize": "small",
                    "color": "b",
                }
            )

            # Add vertical lines to indicate note ratios
            notenames = list(noteratio_dict[subgroup].keys())
            graph.add_child(component := ComponentObject(groups={GroupName.DISSONANCE_NOTELINES}))
            component.properties[PltProperty.COMPONENTTYPE] = ComponentType.VLINES
            component.content = [
                list(np.array(list(noteratio_dict[subgroup].values()))),
                [0] * len(noteratio_dict[subgroup].values()),
                [page.properties[PltProperty.YRANGE][1]] * len(noteratio_dict[subgroup].values()),
                [
                    "r" if notename is ref_note.kempyung(notenames) else "y"
                    for notename in noteratio_dict[subgroup].keys()
                ],
            ]

            # Add note abbreviations to vertical lines
            graph.add_child(
                component := ComponentObject(groups={GroupName.DISSONANCE_TEXT, GroupName.DISSONANCE_NOTELINES})
            )
            component.properties[PltProperty.COMPONENTTYPE] = ComponentType.TEXT
            component.content = [
                {
                    "x": ratio,
                    "y": page.properties[PltProperty.YRANGE][1] - 0.1,
                    "s": notename.short,
                    "fontsize": "small",
                    "color": "r" if notename is ref_note.kempyung(notenames) else "y",
                    "ha": "center",
                    "va": "bottom",
                }
                for notename, ratio in noteratio_dict[subgroup].items()
            ]

            # ANIMATION: Add vertical lines for partials and harmonics
            # Remove the label of the line compos (don't need a legend)
            line_compos = [
                child
                for child in graph.children
                if child.get_property_value(PltProperty.COMPONENTTYPE) == ComponentType.LINE
                and child.get_property_value(PltProperty.LABEL)
            ]
            for compo in line_compos:
                del compo.properties[PltProperty.LABEL]

            if component.properties.get(PltProperty.LABEL, None):
                del component.properties[PltProperty.LABEL]
            if page.properties.get(PltProperty.TITLE, None):
                del page.properties[PltProperty.TITLE]
            page.add_child(ani_graph := GraphObject(groups={GroupName.DISSONANCE_ANIMATION_GRAPH}))
            ani_graph.add_child(
                component := ComponentObject(groups={GroupName.DISSONANCE_ANIMATION_GRAPH, GroupName.PARTIALS})
            )
            page.properties[PltProperty.CONSTRAINED_LAYOUT] = True
            ani_graph.properties[PltProperty.AXIS_NOFRAME] = ["top", "left", "right"]
            ani_graph.properties[PltProperty.AXIS_NOTICKS_X] = True
            ani_graph.properties[PltProperty.AXIS_NOTICKS_Y] = True

            partial_freqs = sorted(p.ratio for p in partials)
            ani_graph.properties[PltProperty.XRANGE] = (0.9, 1.3 * max(partial_freqs))
            _, y_max_ani = ani_graph.properties[PltProperty.YRANGE] = (0, 1)

            component.properties[PltProperty.COMPONENTTYPE] = ComponentType.VLINES
            component.content = [
                list(np.array(partial_freqs)),
                [0] * len(partial_freqs),
                [y_max_ani] * len(partial_freqs),
                ["b"] * len(partial_freqs),
            ]
            ani_graph.add_child(
                component := ComponentObject(groups={GroupName.DISSONANCE_ANIMATION_GRAPH, GroupName.HARMONICS})
            )
            component.properties[PltProperty.COMPONENTTYPE] = ComponentType.VLINES
            harmonic_freqs = sorted(p.ratio for p in harmonics)
            component.content = [
                list(np.array(harmonic_freqs)),
                [0] * len(harmonic_freqs),
                [y_max_ani] * len(harmonic_freqs),
                ["y"] * len(harmonic_freqs),
            ]

        page.properties[PltProperty.NBR_ROWS] = nr_cols = len([child for child in page.children if not child.skip])
        page.properties[PltProperty.NBR_COLUMNS] = 1
        if nr_cols == 2:
            page.properties[PltProperty.HEIGHT_RATIOS] = [4, 1]

    filename = filename.format(ref_note=ref_note, ref_octave=ref_octave, aggregation="-".join(aggregation or ("all",)))
    filepath = get_path(groupname=groupname, folder=Folder.ANALYSES_DISS, filename=filename)
    step_save_object_to_jsonfile(
        document, groupname=groupname, folder=Folder.ANALYSES_DISS, filename=filename[:-4] + ".json"
    )
    generator(document=document, filepath=filepath)


if __name__ == "__main__":
    ...

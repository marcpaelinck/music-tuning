from itertools import chain
from xmlrpc.client import MAXINT

import numpy as np

from tuning.analysis.dissonance import (
    dissonance_j,
    dissonance_profile,
    dissonance_s,
    generate_partial_harmonic_pair,
    get_noteratio_dict,
)
from tuning.analysis.utils import get_minima
from tuning.common.classes import (
    AggregatedPartial,
    AggregatedPartialDict,
    ConsonanceAnalysis,
    InstrumentType,
    Tone,
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
from tuning.common.utils import get_logger, get_path
from tuning.visualization.utils import (
    ComponentObject,
    DocumentObject,
    GraphObject,
    GroupName,
    PageObject,
    PlotItem,
    PlotType,
    create_pdf_with_plots,
    plot_graphs,
)

logger = get_logger(__name__)


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


def get_line_format(combi: PartialCombiType, item: PlotItem, nr_partials: int = None, nr_harmonics: int = None) -> str:
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
                PlotItem.COLOR: "b",
                PlotItem.LABEL: f"partials[{nr_partials}]-partials[{nr_partials}]",
            }.get(item, None)
        case PartialCombiType.PARTIALS_HARMONICS:
            return {
                PlotItem.COLOR: "g",
                PlotItem.LABEL: f"partials[{nr_partials}]-harmonics[{nr_harmonics}]",
            }.get(item, None)
        case PartialCombiType.HARMONICS_PARTIALS:
            return {
                PlotItem.COLOR: "c",
                PlotItem.LABEL: f"harmonics[{nr_harmonics}]-partials[{nr_partials}]",
            }.get(item, None)
    return None


def get_stat(stat: callable, diss_results, index, keys, insttype_str):
    return stat(
        chain.from_iterable(
            [
                diss_results[key][combi][index]
                for key in keys
                if key.startswith(insttype_str)
                for combi in PartialCombiType
            ]
        )
    )


def generate_dissonance_data_optimized(analysis_results: ConsonanceAnalysis):
    if analysis_results:
        # Retrieve the combinations of partials and harmonics for each aggregation group
        key = InstrumentType.ALL.value
        partials = analysis_results.optimal_consonances.partials
        harmonics = analysis_results.optimal_consonances.harmonics
        noteratio_dict = {InstrumentType.ALL.value: analysis_results.noteratio_dict}
        nr_partials = len(analysis_results.optimal_consonances.partials)
        nr_harmonics = len(analysis_results.optimal_consonances.harmonics)
        ref_note = analysis_results.ref_note
        ref_octave = analysis_results.ref_octave
        partial_harmonic_pairs = {
            PartialCombiType.PARTIALS_PARTIALS: (partials, partials),
            PartialCombiType.PARTIALS_HARMONICS: (partials, harmonics),
            PartialCombiType.HARMONICS_PARTIALS: (harmonics, partials),
        }
        fundamental = next((partial for partial in partials if partial.isfundamental), None)
        diss_results = {key: dissonance_profile(partial_harmonic_pairs, fundamental)}
    return (
        noteratio_dict,
        nr_partials,
        nr_harmonics,
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
        diss_results[key] = dissonance_profile(partial_harmonic_pairs, fundamental)

        # list of note frequencies relative to the referenc note
        note_dict = {
            notename: (notes_oct_0.get(notename, None) or group.notes_octave_0[notename])
            for notename in group.notes_octave_0.keys()
        }
        noteratio_dict[key] = get_noteratio_dict(note_dict=note_dict, reference_note=note_dict[ref_note])

    return (
        noteratio_dict,
        nr_partials,
        nr_harmonics,
        ref_note,
        ref_octave,
        diss_results,
    )


def step_plot_dissonance_graphs(
    *,
    groupname: InstrumentGroupName,
    filename: str,
    analysis_results: ConsonanceAnalysis = None,
    # All following arguments will be ignored if analysis_results is given.
    aggr_partials: AggregatedPartialDict,
    nr_partials: int = DEFAULT_NR_PARTIALS_DISS_GRAPH,
    nr_harmonics: int = DEFAULT_NR_HARMONICS_DISS_GRAPH,
    amplitude: float = None,
    ref_note: NoteName = NoteName.DING,
    ref_octave: int = 3,
    skip: list[GroupName] = list(),
):
    """
    Generates dissonance plots using aggregated partials as input, and saves them in a PDF document.
    For an explanation of the structure that is being built in this method, please read the comment
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
    """
    logger.info("Creating dissonance graphs")

    #
    # Generate the data for the graphs
    #
    if analysis_results:
        # Retrieve the combinations of partials and harmonics for each aggregation group
        (noteratio_dict, nr_partials, nr_harmonics, ref_note, ref_octave, diss_results) = (
            generate_dissonance_data_optimized(analysis_results)
        )
        logger.info(f"Graphs will be based on optimal average partial combinations over all instruments")
    else:
        # Generate the requested combinations of partials and harmonics for each aggregation group
        # First calculate the average note frequencies for the reference octave (octave 0)
        (noteratio_dict, nr_partials, nr_harmonics, ref_note, ref_octave, diss_results) = generate_dissonance_data(
            aggr_partials, nr_partials, nr_harmonics, ref_note, ref_octave
        )
        logger.info(
            f"Data is aggregated {'by [' + ', '.join(aggr_partials.aggregation) + ']' if aggr_partials.aggregation else 'over all instruments'}"
        )

    #
    # Create the graphs
    #
    document = DocumentObject(id=1)
    document.properties[PlotItem.SKIP_GROUPS] = skip

    for instrumenttype in InstrumentType:
        # TODO retrieving dict entry by using string literal is error prone
        # Key is a string with format "instrumenttype#ombak-octave"
        keys = [key for key in aggr_partials.root.keys() if key.startswith(instrumenttype.value)]
        if not keys:
            continue

        logger.info(f"--- generating data for {instrumenttype.value}")

        # Generate the data for the graphs and set values if necessary

        document.add_content(page := PageObject())
        page.properties[PlotItem.PAGESIZE] = PageSizes.A4
        page.properties[PlotItem.TITLE] = f"Dissonance graph for {groupname.value} {instrumenttype.value}" + (
            " (same amplitude for all partials)" if amplitude else ""
        )
        x_min = page.properties[PlotItem.XMIN] = 0.95 * get_stat(min, diss_results, 0, keys, instrumenttype.value)
        x_max = page.properties[PlotItem.XMAX] = 1.05 * get_stat(max, diss_results, 0, keys, instrumenttype.value)
        y_min = page.properties[PlotItem.YMIN] = 0.95 * get_stat(min, diss_results, 1, keys, instrumenttype.value)
        y_max = page.properties[PlotItem.YMAX] = 1.05 * get_stat(max, diss_results, 1, keys, instrumenttype.value)

        for subgroup in keys:
            page.add_content(graph := GraphObject())
            diss_result = diss_results[subgroup]

            # Add the dissonance line graphs
            for combi in PartialCombiType:
                # Add a line
                componentgroupname = {
                    PartialCombiType.PARTIALS_HARMONICS: GroupName.DISSONANCE_PLOT_PH,
                    PartialCombiType.HARMONICS_PARTIALS: GroupName.DISSONANCE_PLOT_HP,
                    PartialCombiType.PARTIALS_PARTIALS: GroupName.DISSONANCE_PLOT_PP,
                }[combi]
                graph.add_content(component := ComponentObject(group=componentgroupname))
                component.properties[PlotItem.PLOTTYPE] = PlotType.LINE
                component.content = diss_result[combi]
                component.properties[PlotItem.COLOR] = get_line_format(combi, PlotItem.COLOR)
                component.properties[PlotItem.LABEL] = get_line_format(combi, PlotItem.LABEL, nr_partials, nr_harmonics)

                # add values of minima
                graph.add_content(component := ComponentObject(group=GroupName.DISSONANCE_TEXT_MINIMA))
                component.properties[PlotItem.PLOTTYPE] = PlotType.TEXT
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

            # Add graph title
            graph.add_content(component := ComponentObject(group=GroupName.GRAPH_TITLE))
            component.properties[PlotItem.PLOTTYPE] = PlotType.TEXT
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
            graph.add_content(component := ComponentObject(group=GroupName.DISSONANCE_NOTELINES))
            component.properties[PlotItem.PLOTTYPE] = PlotType.VLINES
            component.content = [
                list(np.array(list(noteratio_dict[subgroup].values()))),
                [0] * len(noteratio_dict[subgroup].values()),
                [page.properties[PlotItem.YMAX]] * len(noteratio_dict[subgroup].values()),
                [
                    "r" if notename is ref_note.kempyung(notenames) else "y"
                    for notename in noteratio_dict[subgroup].keys()
                ],
            ]

            # Add note abbreviations to vertical lines
            graph.add_content(component := ComponentObject(group=GroupName.DISSONANCE_NOTELINES))
            component.properties[PlotItem.PLOTTYPE] = PlotType.TEXT
            component.content = [
                {
                    "x": ratio,
                    "y": page.properties[PlotItem.YMAX] - 0.1,
                    "s": notename.short,
                    "fontsize": "small",
                    "color": "r" if notename is ref_note.kempyung(notenames) else "y",
                    "ha": "center",
                    "va": "bottom",
                }
                for notename, ratio in noteratio_dict[subgroup].items()
            ]

        page.properties[PlotItem.NBR_ROWS] = len(page.content)
        page.properties[PlotItem.NBR_COLUMNS] = 1

    filename = filename.format(
        ref_note=ref_note, ref_octave=ref_octave, aggregation="-".join(aggr_partials.aggregation or ("all",))
    )
    filepath = get_path(groupname=groupname, folder=Folder.ANALYSES_DISS, filename=filename)
    create_pdf_with_plots(document=document, filepath=filepath)


if __name__ == "__main__":
    ...

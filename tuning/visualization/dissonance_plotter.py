import numpy as np

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
    InstrumentType,
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
from tuning.visualization.utils import (
    DocContent,
    PageContent,
    PlotItem,
    PlotType,
    create_pdf_new,
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


def plot_dissonance_graphs(
    group: AggregatedPartialDict,
    *,
    groupname: InstrumentGroupName,
    filename: str,
    analysis_results: ConsonanceAnalysis = None,  # overrules the next five arguments
    nr_partials: int = DEFAULT_NR_PARTIALS_DISS_GRAPH,
    nr_harmonics: int = DEFAULT_NR_HARMONICS_DISS_GRAPH,
    amplitude: float = None,
    ref_note: NoteName = NoteName.DING,
    ref_octave: int = 3,
    **kwargs,
):

    document = dict()
    document[PlotItem.CONTENT]: DocContent = list()  # type: ignore

    for instrumenttype in InstrumentType:
        # TODO retrieving dict entry by using string literal is error prone
        # Key is a string with format "instrumenttype#ombak-octave"
        keys = [key for key in list(group.root.keys()) if key.startswith(instrumenttype.value)]
        if not keys:
            continue

        logger.info(f"--- generating data for {instrumenttype.value}")
        page = dict()
        document[PlotItem.CONTENT].append(page)
        page[PlotItem.CONTENT]: PageContent = list()  # type: ignore
        page[PlotItem.PAGETITLE] = f"Dissonance graph for {groupname.value} {instrumenttype.value}" + (
            " (same amplitude for all partials)" if amplitude else ""
        )
        page[PlotItem.AUTOX] = True

        if analysis_results:
            # Retrieve the combinations of partials and harmonics for each aggregation group
            partial_harmonic_pairs = {
                InstrumentType.ALL.value: (
                    analysis_results.optimal_consonances.partials,
                    analysis_results.optimal_consonances.harmonics,
                )
            }
            noteratio_dict = analysis_results.noteratio_dict
        else:
            # Generate the requested combinations of partials and harmonics for each aggregation group
            # First calculate the average note frequencies for the reference octave (octave 0)
            allnotes = [note for _, notes in group.root.values() for note in notes]
            groupnotes = [note for key in keys for note in group.root[key][1]]
            reduced_freqs = reduced_freq_dict(
                group_notes=groupnotes, all_notes=allnotes, ref_note=ref_note, ref_octave=ref_octave
            )
            # Then determine the ratio of each note relative to the reference note and generate the partials.
            noteratio_dict = {notename: freq / reduced_freqs[ref_note] for notename, freq in reduced_freqs.items()}
            partial_harmonic_pairs = {
                key: generate_partial_harmonic_pair(
                    group.root[key][0],
                    nbr_partials=nr_partials,
                    nbr_harmonics=nr_harmonics,
                    reduced_freqs=reduced_freqs,
                    ref_note=ref_note,
                    ref_octave=ref_octave,
                )
                for key in keys
            }

        diss_graphs = list()
        diss_graph_texts = list()
        diss_graph_titles = list()
        for subgroup, (partials, harmonics) in partial_harmonic_pairs.items():
            pairs = {
                PartialCombiType.PARTIALS_PARTIALS: (partials, partials),
                PartialCombiType.PARTIALS_HARMONICS: (partials, harmonics),
                PartialCombiType.HARMONICS_PARTIALS: (harmonics, partials),
            }
            fundamental = next((partial for partial in partials if partial.isfundamental), None)
            diss_result = dissonance_profile(pairs, fundamental)
            settings = {
                PartialCombiType.PARTIALS_PARTIALS: {
                    "color": "b",
                    "title": f"partials[{nr_partials}]-partials[{nr_partials}]",
                },
                PartialCombiType.PARTIALS_HARMONICS: {
                    "color": "g",
                    "title": f"partials[{nr_partials}]-harmonics[{nr_harmonics}]",
                },
                PartialCombiType.HARMONICS_PARTIALS: {
                    "color": "c",
                    "title": f"harmonics[{nr_harmonics}]-partials[{nr_partials}]",
                },
            }
            diss_graphs.append(
                [
                    diss_result[combi]
                    + [get_minima(*diss_result[combi])]
                    + [settings[combi]["color"]]
                    + [settings[combi]["title"]]
                    for combi in PartialCombiType
                ]
            )
            diss_graph_titles.extend([subgroup])

        ymin = min(min(line[1]) for graph in diss_graphs for line in graph)
        ymax = max(max(line[1]) for graph in diss_graphs for line in graph)
        page[PlotItem.YMIN] = ymin
        page[PlotItem.YMAX] = ymax

        vlines_graph = list()
        vlines_text = list()
        for graph_data, title in zip(diss_graphs, diss_graph_titles):
            text_data = []
            for line_data in graph_data:
                text_data.extend(
                    [
                        {
                            "x": dissonance.ratio,
                            "y": dissonance.dissonance,
                            "s": f"{abs(dissonance.ratio):1.3f}",
                            "fontsize": "x-small",
                            "ha": "center",
                            "va": "top",
                        }
                        for dissonance in line_data[2]
                    ]
                )
                text_data.extend(
                    [
                        {
                            "x": min(line_data[0]) + 0.01 * (max(line_data[0]) - min(line_data[0])),
                            "y": ymax - 0.05 * (ymax - ymin),
                            "s": title,
                            "fontsize": "small",
                            "color": "b",
                        }
                    ]
                )
            diss_graph_texts.append(text_data)
            page[PlotItem.CONTENT].append(
                {
                    PlotItem.CONTENT: diss_graphs,
                    PlotItem.PLOTTEXTS: diss_graph_texts,
                    PlotItem.PLOTTYPE: PlotType.CONSONANCEPLOT,
                }
            )

            sign = 1  # graph_data[0][0] / abs(graph_data[0][0])
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
        page[PlotItem.CONTENT].append(
            {PlotItem.CONTENT: vlines_graph, PlotItem.PLOTTEXTS: vlines_text, PlotItem.PLOTTYPE: PlotType.VLINES}
        )

    filename = filename.format(
        ref_note=ref_note, ref_octave=ref_octave, aggregation="-".join(group.aggregation or ("all",))
    )
    filepath = get_path(groupname=groupname, folder=Folder.ANALYSES, filename=filename)
    create_pdf_new(document=document, filepath=filepath)


if __name__ == "__main__":
    ...

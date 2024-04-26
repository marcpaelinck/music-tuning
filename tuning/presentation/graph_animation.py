import itertools
from enum import Enum
from functools import partial
from typing import Any, Generator

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axis import Axis
from matplotlib.lines import Line2D

from tuning.analysis.utils import create_consonance_analysis
from tuning.common.classes import Frequency
from tuning.common.constants import (
    DEFAULT_ANIMATION_FPS,
    Folder,
    InstrumentGroupName,
    NoteName,
    PageSizes,
)
from tuning.common.utils import (
    get_logger,
    get_path,
    mergedict,
    step_read_object_from_jsonfile,
    step_save_object_to_jsonfile,
)
from tuning.visualization.dissonance_plotter import step_plot_dissonance_graphs
from tuning.visualization.utils import (
    ComponentType,
    DocumentObject,
    GraphObject,
    GroupName,
    HierachicalPlotObject,
    OutputType,
    PageObject,
    PlotObject,
    PltProperty,
    plot_component,
)

logger = get_logger(__name__)

plt.rcParams["animation.ffmpeg_path"] = "C:\\Users\\marcb\\AppData\\Roaming\\ffmpeg\\bin\\ffmpeg.exe"


def get_children_old(parent: HierachicalPlotObject, include: list[GroupName] = [], skip: list[GroupName] = []):
    returnval = [
        child
        for child in parent.children
        if (not include or any(group.value in include for group in child.groups))
        and not any(group.value in skip for group in child.groups)
    ]
    return returnval


def get_children(parents: list[HierachicalPlotObject], groups: list[GroupName] = [], skip: bool | None = None):
    return [
        child
        for parent in parents
        for child in parent.children
        if (not groups or any(group in groups for group in child.groups)) and (skip == None or child.skip == skip)
    ]


def data_gen(data):
    for x, y in zip(*data):
        yield x, y


def init(ax: Axis, line, vlines, xdata: list, ydata: list, vlinedata: list):
    # ax.set_ylim(0, 1)
    # ax.set_xlim(0.9, 6.6)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    for s, x in zip(vlines.get_segments(), vlinedata[0]):
        s[0][0] = x
        s[1][0] = x
    # vlines.set_data(*vlinedata[:2])
    return (line, vlines)


def create_fig(
    graphs: list[GraphObject],
    page: PageObject,
    document=DocumentObject,
    line_compo_id: int = -1,
    harmonics_compo_id: int = -1,
):
    figure = axes = line = vlines = None
    handles = list()
    for graph in graphs:
        for component in graph.children:
            if not component.skip:
                figure, axes, handles = plot_component(
                    component.content,
                    outputtype=OutputType.ANIMATION,
                    figure=figure,
                    axes=axes,
                    handles=handles,
                    component_args=mergedict(component.properties, component.animationproperties),
                    graph_args=mergedict(graph.properties, graph.animationproperties),
                    page_args=mergedict(page.properties, page.animationproperties),
                    doc_args=mergedict(document.properties, document.animationproperties),
                )
                if component.id == line_compo_id:
                    line = handles[-1]
                if component.id == harmonics_compo_id:
                    vlines = handles[-1]
    return figure, axes, line, vlines


def run(
    data: list[float],
    xdata: list[float],
    ydata: list[float],
    ax: Axis,
    line: Line2D,
    vlines: dict[str, Any],
    countdown: Generator,  # count down for the intro before the animation starts
):
    # Do nothing during the intro time
    if next(countdown) > 0:
        return (line,)

    # update the data
    x, y = data
    delta = (x - xdata[-1]) if xdata else 0
    xdata.append(x)
    ydata.append(y)

    line.set_data(xdata, ydata)
    seg_new = np.array(vlines.get_segments())

    # Keep respective ratios intact
    seg_new[:, :, 0] = seg_new[:, :, 0] * (1 + delta / seg_new[0, 0, 0])
    seg_new = list(seg_new)
    vlines.set_segments(seg_new)
    return (line,)


# Only save last 100 frames, but run forever
def plot_animated_diss_graph(
    document: DocumentObject,
    filepath: str,
):
    """Create an MP4 animation

    Args:
        document (DocumentObject): Document containing the graph information
        filepath (str): Destination path for the MP4 file
        duration (int, optional): Required duration of the video in seconds. Defaults to 30 seconds.
        intro (int, optional): Duration of the first part in seconds, during which an empty graph
                               is displayed, before the animation starts. Defaults to 1 second.
                               The total duration of the video will be the sum of duration and intro.
        fps (int, optional): # frames per second. Defaults to 15.
    """
    page = document.children[0]
    # page.properties.pop(PltProperty.PAGESIZE)
    # page.properties[PltProperty.NBR_ROWS] = 1
    # include_list = document.include_only_list
    # skip_list = document.skip_list

    graphs = get_children([page], skip=False)
    components = get_children(
        graphs, groups=[GroupName.DISSONANCE_LINE, GroupName.PARTIALS, GroupName.HARMONICS], skip=False
    )
    line_compo = next((compo for compo in components if GroupName.DISSONANCE_LINE in compo.groups), None)
    partials_compo = next((compo for compo in components if GroupName.PARTIALS in compo.groups), None)
    harmonics_compo = next((compo for compo in components if GroupName.HARMONICS in compo.groups), None)
    if not (line_compo and partials_compo and harmonics_compo):
        logger.error("Missing partials and harmonics component")
        exit()
    duration = document.get_property_value(PltProperty.DURATION)
    intro = document.get_property_value(PltProperty.INTRO)
    if not duration or not intro:
        raise Exception("Missing video duration or intro")

    # Generate data for the animation and add dummy data for the intro
    fps = DEFAULT_ANIMATION_FPS
    step_size = int(len(line_compo.content[0]) / (duration * fps))
    data_with_intro = [(intro * step_size * fps * [0]) + data for data in line_compo.content]
    line_data = [[x for i, x in enumerate(data) if not i % step_size] for data in data_with_intro]

    xdata, ydata = [], []
    harmonics_data = [vline.copy() for vline in harmonics_compo.content]

    fig, axis, line, harmonics_vlines = create_fig(graphs, page, document, line_compo.id, harmonics_compo.id)
    run_func = partial(
        run,
        xdata=xdata,
        ydata=ydata,
        ax=axis,
        line=line,
        vlines=harmonics_vlines,
        countdown=itertools.count(intro * fps, -1),
    )
    init_func = partial(
        init, ax=axis, line=line, vlines=harmonics_vlines, xdata=xdata, ydata=ydata, vlinedata=harmonics_data
    )
    data_gen_func = partial(data_gen, data=line_data)
    (line, harmonics_vlines) = init(axis, line, harmonics_vlines, xdata, ydata, harmonics_data)
    ani = animation.FuncAnimation(
        fig, run_func, data_gen_func, interval=10, init_func=init_func, save_count=len(line_data[0])
    )
    filepath = filepath[:-4] + ".mp4"
    metadata = dict(title=filepath[:-4], artist="Matplotlib", comment="Animated dissonance graph")
    FFwriter = animation.FFMpegWriter(fps=fps, metadata=metadata)
    ani.save(filepath, writer=FFwriter)
    logger.info(f"Animation file saved to {filepath}")
    # plt.show()


if __name__ == "__main__":
    part_ratios = [1.0]
    harm_ratios = [1.0]
    fundamental_freq = 400

    analysis_results = create_consonance_analysis(partial_ratios=part_ratios, harmonic_ratios=harm_ratios)
    step_save_object_to_jsonfile(
        object=analysis_results,
        groupname=InstrumentGroupName.TEST,
        folder=Folder.ANALYSES_DISS,
        filename="PL_dissonance_plot.json",
        logginginfo="analysis Plomp-Levelt dissonance plot",
    )
    step_plot_dissonance_graphs(
        groupname=InstrumentGroupName.TEST,
        analysis_results=analysis_results,
        # filename="dissonance_graph-{ref_note}{ref_octave}_aggr[{aggregation}].pdf",
        filename="dissonance_graph_PL.pdf",
        ref_octave=2,
        ref_note=NoteName.DING,
        end_shift_ratio=1.5,
        include_only=[GroupName.DISSONANCE_GRAPH],
        skip=[GroupName.DISSONANCE_LINE_PP, GroupName.DISSONANCE_LINE_HP],
    )

    groupname = InstrumentGroupName.GENDERWAYANG
    folder = Folder.ANALYSES_DISS
    json_filename = "dissonance_graph-DING2_aggr[InstrumentType-OmbakType].json"
    mp4_filename = "dissonance_graph-DING2_aggr[InstrumentType-OmbakType].mp4"
    document = step_read_object_from_jsonfile(
        groupname=groupname, objecttype=DocumentObject, folder=folder, filename=json_filename
    )
    plot_animated_diss_graph(document, get_path(groupname, folder, mp4_filename))

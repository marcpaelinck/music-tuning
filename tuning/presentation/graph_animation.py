import itertools
from enum import Enum
from functools import partial

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axis import Axis

from tuning.common.classes import Frequency
from tuning.common.constants import Folder, InstrumentGroupName, PageSizes
from tuning.common.utils import get_logger, get_path, step_read_object_from_jsonfile
from tuning.visualization.utils import (
    ComponentType,
    DocumentObject,
    GraphObject,
    GroupName,
    HierachicalPlotObject,
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
                    figure=figure,
                    axes=axes,
                    handles=handles,
                    component_args=component.properties,
                    graph_args=graph.properties,
                    page_args=page.properties,
                    doc_args=document.properties,
                )
                if component.id == line_compo_id:
                    line = handles[-1]
                if component.id == harmonics_compo_id:
                    vlines = handles[-1]
    return figure, axes, line, vlines


def run(data, xdata, ydata, ax, line, vlines):
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
def plot_animated_diss_graph(document: DocumentObject, filepath: str, duration: int = 30, fps=15):
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

    step_size = int(len(line_compo.content[0]) / (duration * fps))
    line_data = [
        [x for x, i in zip(line_compo.content[0], itertools.count()) if not i % step_size],
        [y for y, i in zip(line_compo.content[1], itertools.count()) if not i % step_size],
    ]
    xdata, ydata = [], []
    harmonics_data = [vline.copy() for vline in harmonics_compo.content]

    fig, axis, line, harmonics_vlines = create_fig(graphs, page, document, line_compo.id, harmonics_compo.id)
    run_func = partial(run, xdata=xdata, ydata=ydata, ax=axis, line=line, vlines=harmonics_vlines)
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
    groupname = InstrumentGroupName.GENDERWAYANG
    folder = Folder.ANALYSES_DISS
    json_filename = "dissonance_graph-DING2_aggr[InstrumentType-OmbakType].json"
    mp4_filename = "dissonance_graph-DING2_aggr[InstrumentType-OmbakType].mp4"
    document = step_read_object_from_jsonfile(
        groupname=groupname, objecttype=DocumentObject, folder=folder, filename=json_filename
    )
    plot_animated_diss_graph(document, get_path(groupname, folder, mp4_filename))

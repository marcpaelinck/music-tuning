import itertools
from functools import partial

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axis import Axis

from tuning.common.constants import Folder, InstrumentGroupName, PageSizes
from tuning.common.utils import get_path, step_read_object_from_jsonfile
from tuning.visualization.utils import (
    DocumentObject,
    GraphObject,
    PageObject,
    PlotItem,
    PlotType,
    plot_graphs,
)

plt.rcParams["animation.ffmpeg_path"] = "C:\\Users\\marcb\\Desktop\\ffmpeg\\bin\\ffmpeg.exe"


def data_gen(data):
    for x, y in zip(*data):
        yield x, y


def init(ax: Axis, line, vlines, xdata: list, ydata: list, vlinedata: list):
    ax.set_ylim(0, 3500)
    ax.set_xlim(0.9, 2.6)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    for s, x in zip(vlines.get_segments(), vlinedata[0]):
        s[0][0] = x
        s[1][0] = x
    # vlines.set_data(*vlinedata[:2])
    return (line, vlines)


def create_fig(
    graph: GraphObject, page: PageObject, document=DocumentObject, line_compo_id: int = -1, vline_compo_id: int = -1
):
    figure = axes = line = vlines = None
    handles = list()
    for component in graph.content:
        figure, axes, handles = plot_graphs(
            component.content,
            figure=figure,
            axes=axes,
            handles=handles,
            component_args=component.properties,
            plot_args=graph.properties,
            page_args=page.properties,
            doc_args=document.properties,
        )
        if component.id == line_compo_id:
            line = handles[-1]
        if component.id == vline_compo_id:
            vlines = handles[-1]
    return figure, axes[0], line, vlines


def run(data, xdata, ydata, vlinedata, ax, line, vlines):
    # update the data
    x, y = data
    delta = (x - xdata[-1]) if xdata else 0
    xdata.append(x)
    ydata.append(y)
    # vlinedata[0] = list(np.array(vlinedata[0]) + delta)

    xmin, xmax = ax.get_xlim()
    if x >= xmax:
        ax.set_xlim(xmin, 2 * xmax)
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)
    seg_new = [np.array([seg[:, 0] + delta, seg[:, 1]]).T for seg in vlines.get_segments()]
    vlines.set_segments(seg_new)
    return (line,)


# Only save last 100 frames, but run forever
def step_plot_animated_diss_graph(document: DocumentObject, filepath: str):
    page = document.content[0]
    document.properties[PlotItem.PAGESIZE] = PageSizes.A4
    graph = page.content[0]
    filepath = filepath[:-4] + ".mp4"
    line_compo = next(
        (compo for compo in graph.content if compo.properties.get(PlotItem.PLOTTYPE, None) == PlotType.LINE.value), None
    )
    vline_compo = next(
        (compo for compo in graph.content if compo.properties.get(PlotItem.PLOTTYPE, None) == PlotType.VLINES.value),
        None,
    )
    if not (line_compo and vline_compo):
        exit()
    line_data = [
        [x for x, i in zip(line_compo.content[0], itertools.count()) if not i % 20],
        [y for y, i in zip(line_compo.content[1], itertools.count()) if not i % 20],
    ]
    xdata, ydata = [], []
    vlinedata = [vline.copy() for vline in vline_compo.content]

    fig, axis, line, vlines = create_fig(graph, page, document, line_compo.id, vline_compo.id)
    run_func = partial(run, xdata=xdata, ydata=ydata, vlinedata=vlinedata, ax=axis, line=line, vlines=vlines)
    init_func = partial(init, ax=axis, line=line, vlines=vlines, xdata=xdata, ydata=ydata, vlinedata=vlinedata)
    data_gen_func = partial(data_gen, data=line_data)
    (line, vlines) = init(axis, line, vlines, xdata, ydata, vlinedata)
    ani = animation.FuncAnimation(
        fig, run_func, data_gen_func, interval=10, init_func=init_func, save_count=len(line_data[0])
    )
    metadata = dict(title=filepath[:-4], artist="Matplotlib", comment="Animated dissonance graph")
    FFwriter = animation.FFMpegWriter(fps=15, metadata=metadata)
    ani.save(filepath, writer=FFwriter)
    plt.show()


if __name__ == "__main__":
    groupname = InstrumentGroupName.GENDERWAYANG
    folder = Folder.ANALYSES_DISS
    json_filename = "dissonance_graph-DING2_aggr[InstrumentType-OmbakType].json"
    mp4_filename = "dissonance_graph-DING2_aggr[InstrumentType-OmbakType].mp4"
    document = step_read_object_from_jsonfile(
        groupname=groupname, objecttype=DocumentObject, folder=folder, filename=json_filename
    )
    step_plot_animated_diss_graph(document, get_path(groupname, folder, mp4_filename))

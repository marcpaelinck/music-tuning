import os
import pathlib
from enum import Enum, auto
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from tuning.common.classes import AggregatedPartialDict, InstrumentGroup
from tuning.common.constants import Folder, InstrumentGroupName, PageSizes
from tuning.common.utils import get_logger, get_path

logger = get_logger(__name__)


class PlotType(Enum):
    SOUNDDATAPLOT = auto()
    SPECTRUMPLOT = auto()
    CONSONANCEPLOT = auto()
    VLINES = auto()


def plot_graphs(
    *args: tuple[list[float]],
    ncols: int = 1,
    plottype: PlotType = PlotType.SPECTRUMPLOT,
    pagetitle: str = "",
    plottexts: list[list[dict[str, Any]]] = None,
    xmin: int = None,
    xmax: int = None,
    ymin: int = None,
    ymax: int = None,
    autox: bool = False,
    autoy: bool = False,
    show: bool = False,
    pagesize: PageSizes = PageSizes.A4,
    figure: Figure = None,
    axes: Axes = None,
    **kwargs,
):
    x_index = 0
    y_index = 1
    ymin_index = 1
    ymax_index = 2
    mins_index = 2

    def set_axes_boundaries(x_index, y_index) -> tuple[float]:
        x_min = x_max = y_min = y_max = None
        if not autox:
            for arg, axis in zip(args, axes):
                x_min = xmin if xmin != None else min(arg[x_index])
                x_max = xmax if xmax != None else max(arg[x_index])
                plt.setp(axis, xlim=(x_min, x_max))
        if not autoy:
            # min and max y should be the same for all graphs
            y_min = ymin if ymin != None else min(min(args[i][y_index]) for i in range(len(args)))
            y_max = ymax if ymax != None else max(max(args[i][y_index]) for i in range(len(args))) * 1.05
            plt.setp(axes, ylim=(y_min, y_max))
        return (x_min, x_max, y_min, y_max)

    if not figure:
        figure, axes = plt.subplots(len(args) // ncols, ncols, figsize=pagesize.value)
        if len(args) == 1:
            axes = np.array([axes])
        figure.suptitle(pagetitle)

    axis_list = axes.flatten("C")
    for axis, idx in zip(axis_list, range(len(axis_list))):
        match plottype:
            case PlotType.SOUNDDATAPLOT | PlotType.SPECTRUMPLOT:
                if idx == 0:
                    set_axes_boundaries(x_index, y_index)
                # vis = [axis.plot for axis in axis_list]
                axis.plot(*args[idx])
            case PlotType.CONSONANCEPLOT:
                if idx == 0:
                    set_axes_boundaries(x_index, y_index)
                handles = []
                # vis = [axis.plot for axis in axis_list]
                for line in args[idx]:
                    (handle,) = axis.plot(*line[:2], color=line[3], label=line[4])
                    handles.append(handle)
                axis.legend(handles=handles, loc="upper right")
                # plotvalues = [(args[i][x_index], args[i][y_index]) for i in range(len(args))]
            case PlotType.VLINES:
                if idx == 0:
                    _, _, y_min, _ = set_axes_boundaries(x_index, ymax_index)
                # vis = [axis.vlines for axis in axis_list]
                # for arg in args:
                args[idx][ymin_index] = (ymin,) * len(args[idx][ymin_index])
                axis.vlines(*args[idx])
            case _:
                ...
        # for idx in range(len(axis_list)):
        # vis[idx](*plotvalues[idx])
        xmin, xmax = axis_list[idx].get_xlim()
        ymin, ymax = axis_list[idx].get_ylim()
        if plottexts:
            for text in plottexts[idx]:
                axis_list[idx].text(**text)
    if show:
        plt.show()
    return figure, axes


def create_pdf(
    group: InstrumentGroup | AggregatedPartialDict,
    groupname: InstrumentGroupName,
    plotter: callable,
    folder: Folder,
    filename: str,
    iterlist: list = None,  ## list of objects to iterate through
    **kwargs,
) -> bool:

    filepath = get_path(groupname, folder, filename)
    if (
        isinstance(group, AggregatedPartialDict)
        and ("ref_note" in kwargs and "ref_octave" in kwargs)
        or ("analysis_results" in kwargs)
    ):
        filepath = filepath.format(
            ref_note=kwargs["ref_note"],
            ref_octave=kwargs["ref_octave"],
            aggregation="_" + "_".join(group.aggregation or ("all",)),
        )
    # Check if the file is closed. If not, exit.
    if os.path.exists(filepath):
        success = False
        warned = False
        while not success:
            try:
                with open(filepath, "r+") as file:
                    success = True
            except:
                if not warned:
                    logger.error(f"File {filepath} is in use. Please close the file and try again.")
                warned = True
    if iterlist is None:
        iterlist = group.instruments

    with PdfPages(filepath) as pdf:
        plotname = pathlib.Path(filepath).stem.replace("_", " ")
        logger.info(f"Generating PDF plots of {plotname}.")
        for item in iterlist:
            hasplots = plotter(
                group=group,
                groupname=groupname,
                item=item,
                **kwargs,
            )
            if hasplots:
                pdf.savefig()
                plt.close()
    return group


if __name__ == "__main__":
    ...

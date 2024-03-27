import math
import os
import pathlib
from enum import Enum, auto
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from tuning.common.classes import InstrumentGroup
from tuning.common.constants import PageSizes
from tuning.common.utils import get_logger

logger = get_logger(__name__)


class PlotType(Enum):
    SPECTRUMPLOT = auto()
    CONSONANCEPLOT = auto()
    VLINES = auto()


def plot_graphs(
    *args: tuple[list[float]],
    ncols: int = 1,
    plottype: PlotType = PlotType.SPECTRUMPLOT,
    pagetitle: str = "",
    plottitles: list[str] = None,
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
            x_min = xmin or min(min(args[i][x_index]) for i in range(len(args)))
            x_max = xmax or max(max(args[i][x_index]) for i in range(len(args)))
            plt.setp(axes, xlim=(x_min, x_max))
        if not autoy:
            y_min = ymin or min(min(args[i][y_index]) for i in range(len(args)))
            y_max = ymax or max(max(args[i][y_index]) for i in range(len(args))) * 1.05
            plt.setp(axes, ylim=(y_min, y_max))
        return (x_min, x_max, y_min, y_max)

    if not figure:
        figure, axes = plt.subplots(len(args) // ncols, ncols, figsize=pagesize.value)
        if len(args) == 1:
            axes = np.array([axes])
        figure.suptitle(pagetitle)

    axis_list = axes.flatten("C")
    match plottype:
        case PlotType.SPECTRUMPLOT:
            set_axes_boundaries(x_index, y_index)
            vis = [axis.plot for axis in axis_list]
            plotvalues = args
        case PlotType.CONSONANCEPLOT:
            set_axes_boundaries(x_index, y_index)
            vis = [axis.plot for axis in axis_list]
            for axis, arg in zip(axis_list, args):
                for x, y in arg[mins_index]:
                    axis.text(x, y, f"{abs(x):1.3f}", fontsize="x-small", ha="center", va="top")
            plotvalues = [(args[i][x_index], args[i][y_index]) for i in range(len(args))]
        case PlotType.VLINES:
            _, _, y_min, _ = set_axes_boundaries(x_index, ymax_index)
            vis = [axis.vlines for axis in axis_list]
            for axis, arg in zip(axis_list, args):
                for x, y in zip(arg[x_index], arg[ymax_index]):
                    axis.text(x, y, f"{x:1.3f}", fontsize="x-small", ha="left")
            for arg in args:
                arg[ymin_index] = (ymin,) * len(arg[ymin_index])
            plotvalues = args
        case _:
            ...

    for idx in range(len(axis_list)):
        vis[idx](*plotvalues[idx])
        xmin, xmax = axis_list[idx].get_xlim()
        ymin, ymax = axis_list[idx].get_ylim()
        if plottitles:
            axis_list[idx].text(
                x=xmin + 0.01 * (xmax - xmin),
                y=ymax - 0.1 * (ymax - ymin),
                s=plottitles[idx],
                fontsize="small",
                color="b",
            )
    if show:
        plt.show()
    return figure, axes


def create_pdf(
    group: InstrumentGroup,
    iterlist: list,  ## list of objects to iterate through
    plotter: callable,
    filepath: str,
    **kwargs,
) -> bool:

    # Check if the file is closed. If not, exit.
    if os.path.exists(filepath):
        try:
            with open(filepath, "r+") as file:
                ...
        except:
            logger.error(f"File {filepath} is in use. Please close the file and try again.")
            exit()

    with PdfPages(filepath) as pdf:
        logger.info(
            f"Generating PDF plots of {pathlib.Path(filepath).stem.replace("_", " ")}."
        )
        for object in iterlist:
            hasplots = plotter(
                group=group,
                object=object,
                **kwargs,
            )
            if hasplots:
                pdf.savefig()
                plt.close()
    return True


if __name__ == "__main__":
    ...

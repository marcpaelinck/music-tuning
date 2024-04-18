import os
import pathlib
from enum import Enum, auto
from typing import Any, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from tuning.common.classes import AggregatedPartialDict, InstrumentGroup
from tuning.common.constants import Folder, InstrumentGroupName, PageSizes
from tuning.common.utils import get_logger, get_path, open_create_file

logger = get_logger(__name__)


class PlotType(Enum):
    SOUNDDATAPLOT = auto()
    SPECTRUMPLOT = auto()
    CONSONANCEPLOT = auto()
    VLINES = auto()


class PlotItem(Enum):
    CONTENT = "content"
    PAGETITLE = "pagetitle"
    PLOTTYPE = "plottype"
    PLOTTEXTS = "plottexts"
    SHOW = "show"
    FIGURE = "figure"
    AXES = "axes"
    XMIN = "xmin"
    XMAX = "xmax"
    YMIN = "ymin"
    YMAX = "ymax"
    AUTOX = "autox"

    @classmethod
    def kwargs(cls, d: dict["PlotItem", Any]) -> dict["PlotItem", Any]:
        return {key.value: d[key] for key in PlotItem if key not in {cls.CONTENT} and key in d}


PlotComponent: TypeAlias = list[dict[PlotItem, Any]]
PageContent: TypeAlias = list[dict[PlotItem, list[PlotComponent]]]
DocContent: TypeAlias = list[PageContent]


def plot_graphs(
    content: tuple[list[float]],
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
            for arg, axis in zip(content, axes):
                x_min = xmin if xmin != None else min(arg[x_index])
                x_max = xmax if xmax != None else max(arg[x_index])
                plt.setp(axis, xlim=(x_min, x_max))
        if not autoy:
            # min and max y should be the same for all graphs
            y_min = ymin if ymin != None else min(min(content[i][y_index]) for i in range(len(content)))
            y_max = ymax if ymax != None else max(max(content[i][y_index]) for i in range(len(content))) * 1.05
            plt.setp(axes, ylim=(y_min, y_max))
        return (x_min, x_max, y_min, y_max)

    if not figure:
        figure, axes = plt.subplots(len(content) // ncols, ncols, figsize=pagesize.value)
        if len(content) == 1:
            axes = np.array([axes])
        figure.suptitle(pagetitle)

    axis_list = axes.flatten("C")
    for axis, idx in zip(axis_list, range(len(axis_list))):
        match plottype:
            case PlotType.SOUNDDATAPLOT | PlotType.SPECTRUMPLOT:
                if idx == 0:
                    set_axes_boundaries(x_index, y_index)
                axis.plot(*content[idx])
            case PlotType.CONSONANCEPLOT:
                if idx == 0:
                    set_axes_boundaries(x_index, y_index)
                handles = []
                for line in content[idx]:
                    (handle,) = axis.plot(*line[:2], color=line[3], label=line[4])
                    handles.append(handle)
                axis.legend(handles=handles, loc="upper right")
            case PlotType.VLINES:
                if idx == 0:
                    _, _, y_min, _ = set_axes_boundaries(x_index, ymax_index)
                content[idx][ymin_index] = (ymin,) * len(content[idx][ymin_index])
                axis.vlines(*content[idx])
            case _:
                ...
        xmin, xmax = axis_list[idx].get_xlim()
        ymin, ymax = axis_list[idx].get_ylim()
        if plottexts:
            for text in plottexts[idx]:
                axis_list[idx].text(**text)
    if show:
        plt.show()
    return figure, axes


def create_pdf_with_plots(
    document: dict[str, Any],
    *,
    filepath: str,
) -> bool:
    """
    Creates a PDF document and plots the graphs with the given data.

    Args:
        document (dict[PlotItem, Any]): plot data with format:
                The document hierarchy is Document -> Page -> PlotComponent where:
                Document = dict[PlotItem, Any]
                Page = dict[PlotItem, Any]
                PlotComponent = list[tuple[list[dict[str, Any]], PlotType]
                and:
                document[PlotItem.CONTENT]: list[Page]
                page[PlotItem.CONTENT]: list[PlotComponent]
                Except for PlotItem.CONTENT, all other key-value combinations
                of Document, Page and PlotComponent contain arguments that will
                be passed as **kwargs to the plot_graphs function.
        filepath (str): _description_

    Returns:
        bool: _description_
    """

    # Check if the file is closed. If not, give a warning and wait
    # until the user closes it. If folder does not exist, create it.
    with open_create_file(filepath=filepath, mode="a+", warn_if_open=True):
        pass

    with PdfPages(filepath) as pdf:
        plotname = pathlib.Path(filepath).stem.replace("_", " ")
        logger.info(f"Generating PDF plots of {plotname}.")
        for page in document[PlotItem.CONTENT]:
            figure = None
            axes = None
            for plotcontent in page[PlotItem.CONTENT]:
                figure, axes = plot_graphs(
                    plotcontent[PlotItem.CONTENT],
                    figure=figure,
                    axes=axes,
                    **PlotItem.kwargs(plotcontent),
                    **PlotItem.kwargs(page),
                    **PlotItem.kwargs(document),
                )
            pdf.savefig()
            plt.close()


if __name__ == "__main__":
    ...

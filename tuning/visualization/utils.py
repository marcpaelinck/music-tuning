import pathlib
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count
from typing import Any, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from tuning.common.classes import AggregatedPartialDict, InstrumentGroup
from tuning.common.constants import Folder, InstrumentGroupName, PageSizes
from tuning.common.utils import get_logger, get_path, open_create_file

logger = get_logger(__name__)


class PlotType(Enum):
    SOUNDDATAPLOT = auto()
    SPECTRUMPLOT = auto()
    LINE = auto()
    VLINES = auto()
    TEXT = auto()


class PlotItem(Enum):
    # CONTENT = "content"
    ID = "id"
    TITLE = "pagetitle"
    PAGESIZE = "pagesize"
    PLOTTYPE = "plottype"
    PLOTTEXTS = "plottexts"
    NBR_COLUMNS = "nbr_columns"
    NBR_ROWS = "nbr_rows"
    DATA = "data"
    COLOR = "color"
    LABEL = "label"
    LEGEND = "legend"
    SHOW = "show"
    FIGURE = "figure"
    AXES = "axes"
    XMIN = "xmin"
    XMAX = "xmax"
    YMIN = "ymin"
    YMAX = "ymax"
    AUTOX = "autox"
    SKIP_GROUPS = "skip"

    # def __str__(self):
    #     return self.value

    # def __repr__(self):
    #     return self.value

    @classmethod
    def kwargs(cls, d: dict["PlotItem", Any]) -> dict["PlotItem", Any]:
        return {key.value: d[key] for key in PlotItem if key not in {cls.CONTENT} and key in d}


@dataclass
class PlotObject:
    id: int = None
    content: list["PlotObject"] = field(default_factory=list)
    properties: dict = field(default_factory=dict)

    def __setid__(self, id: int):
        self.id = id
        self.properties[PlotItem.ID] = self.id
        if not isinstance(self, ComponentObject):
            print(f"{self.__class__.__name__} {self.id=}")


class HierachicalPlotObject(PlotObject):
    id: int  # = field(default_factory=itertools.count().__next__)
    content: list["PlotObject"] = field(default_factory=list)
    properties: dict = field(default_factory=dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.__count__ = count()

    def add_content(self, val: object):
        self.content.append(val)
        if isinstance(val, PlotObject):
            val.__setid__(next(self.__count__))

    def lastcontent(self) -> "PlotObject":
        return self.content[-1] if self.content else None


class DocumentObject(HierachicalPlotObject): ...


class PageObject(HierachicalPlotObject): ...


class GraphObject(HierachicalPlotObject): ...


class GroupName(Enum):
    SOUND_PLOT = auto()
    SOUND_DELIMITERS = auto()
    DISSONANCE_PLOT = auto()
    DISSONANCE_TEXT_MINIMA = auto()
    DISSONANCE_GRAPH_TITLE = auto()
    DISSONANCE_NOTELINES = auto()
    SPECTRA_PLOT = auto()
    SPECTRA_PARTIALS = auto()
    SPECTRA_ANALYSIS_PARTIALS = auto()
    SPECTRA_AGGR_PARTIALS = auto()
    LEGEND = auto()
    GRAPH_TITLE = auto()


@dataclass
class ComponentObject(PlotObject):
    group: GroupName = None


def plot_graphs(
    content: tuple[list[float]],
    component_args: dict[PlotItem, Any] = {},
    plot_args: dict[PlotItem, Any] = {},
    page_args: dict[PlotItem, Any] = {},
    doc_args: dict[PlotItem, Any] = {},
    figure: Figure = None,
    axes: Axes = None,
    handles: list[Line2D] = None,
    **kwargs,
):
    def get_args(items: list[PlotItem], strict: bool = True) -> tuple[Any]:
        values = [
            doc_args.get(item, page_args.get(item, plot_args.get(item, component_args.get(item, None))))
            for item in items
        ]
        if strict:
            if missing := [item.value for args, item in zip(values, items) if not args]:
                raise Exception(f"Missing arguments {', '.join(missing)}")
        return values

    if not figure:
        nbr_columns, nbr_rows, pagesize = get_args([PlotItem.NBR_COLUMNS, PlotItem.NBR_ROWS, PlotItem.PAGESIZE])
        figure, axes = plt.subplots(nbr_rows, nbr_columns, figsize=pagesize.value)
        if pagetitle := page_args.get(PlotItem.TITLE, None):
            figure.suptitle(pagetitle)
    if not handles:
        handles = list()

    axis_list = list(axes.flatten()) if isinstance(axes, np.ndarray) else [axes]
    idx = plot_args.get(PlotItem.ID)
    plottype = component_args.get(PlotItem.PLOTTYPE)
    axis = axis_list[idx]

    xmin, xmax, ymin, ymax = get_args([PlotItem.XMIN, PlotItem.XMAX, PlotItem.YMIN, PlotItem.YMAX], strict=False)
    if xmin and xmax:
        plt.setp(axis, xlim=(xmin, xmax))
    if ymin and ymax:
        plt.setp(axis, ylim=(ymin, ymax))

    if color := component_args.get(PlotItem.COLOR, None):
        kwargs["color"] = color
    if label := component_args.get(PlotItem.LABEL, None):
        kwargs["label"] = label

    match plottype:
        case PlotType.LINE:
            kwargs = {}
            if component_args.get(PlotItem.LABEL, None):
                (handle,) = axis.plot(*content, **kwargs)
                handles.append(handle)
                axis.legend(handles=handles, loc="upper right")
            else:
                axis.plot(*content, **kwargs)
        case PlotType.VLINES:
            if component_args.get(PlotItem.LABEL, None):
                handle = axis.vlines(*content, **kwargs)
                handles.append(handle)
                axis.legend(handles=handles, loc="upper right")
            else:
                axis.vlines(*content)
        case PlotType.TEXT:
            for item in content:
                axis.text(**item)

        case _:
            ...
    return figure, axes, handles


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
        for page in document.content:
            logger.info(f"Plotting page {page.properties[PlotItem.ID]+1} of {len(document.content)}")
            figure = None
            axes = None
            for plot in page.content:
                handles = None
                for component in plot.content:
                    if component.group not in document.properties[PlotItem.SKIP_GROUPS]:
                        figure, axes, handles = plot_graphs(
                            component.content,
                            figure=figure,
                            axes=axes,
                            handles=handles,
                            component_args=component.properties,
                            plot_args=plot.properties,
                            page_args=page.properties,
                            doc_args=document.properties,
                        )
            pdf.savefig()
            plt.close()


if __name__ == "__main__":
    ...

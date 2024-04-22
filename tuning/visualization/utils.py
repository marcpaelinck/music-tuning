import pathlib
from enum import Enum, auto
from itertools import count
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pydantic import BaseModel, Field

from tuning.common.utils import get_logger, open_create_file

logger = get_logger(__name__)


class PlotType(Enum):
    LINE = "LINE"
    VLINES = "VLINES"
    TEXT = "TEXT"


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

    @classmethod
    def kwargs(cls, d: dict["PlotItem", Any]) -> dict["PlotItem", Any]:
        return {key.value: d[key] for key in PlotItem if key not in {cls.CONTENT} and key in d}


class GroupName(Enum):
    SOUND_PLOT = "SOUND_PLOT"
    SOUND_DELIMITERS = "SOUND_DELIMITERS"
    DISSONANCE_PLOT = "DISSONANCE_PLOT"
    DISSONANCE_PLOT_PP = "DISSONANCE_PLOT_PP"
    DISSONANCE_PLOT_PH = "DISSONANCE_PLOT_PH"
    DISSONANCE_PLOT_HP = "DISSONANCE_PLOT_HP"
    DISSONANCE_TEXT_MINIMA = "DISSONANCE_TEXT_MINIMA"
    DISSONANCE_GRAPH_TITLE = "DISSONANCE_GRAPH_TITLE"
    DISSONANCE_NOTELINES = "DISSONANCE_NOTELINES"
    SPECTRA_PLOT = "SPECTRA_PLOT"
    SPECTRA_PARTIALS = "SPECTRA_PARTIALS"
    SPECTRA_ANALYSIS_PARTIALS = "SPECTRA_ANALYSIS_PARTIALS"
    SPECTRA_AGGR_PARTIALS = "SPECTRA_AGGR_PARTIALS"
    LEGEND = "LEGEND"
    GRAPH_TITLE = "GRAPH_TITLE"


class PlotObject(BaseModel):
    id: Optional[int] = None
    kind: str = Field(default="PLOT", init=False)
    properties: Optional[dict[PlotItem, Any]] = Field(default_factory=dict)
    content: Optional[Any] = Field(default_factory=list)
    count__: Callable = Field(default_factory=count, init=False, exclude=True)

    def __setid__(self, id: int):
        self.id = id
        self.properties[PlotItem.ID] = self.id

    class Config:
        arbitrary_types_allowed = True


class HierachicalPlotObject(PlotObject):
    kind: str = Field(default="GENERIC_HIERARCHCAL_PLOT", init=False)
    # def __post_model_init__(self, __context):
    #     self.__count__ = count()

    def add_content(self, val: object):
        self.content.append(val)
        if isinstance(val, PlotObject):
            val.__setid__(next(self.count__))

    def lastcontent(self) -> "PlotObject":
        return self.content[-1] if self.content else None


class ComponentObject(PlotObject):
    kind: str = Field(default="COMPONENT", init=False)
    groups: list[GroupName] = Field(default_factory=list)
    content: Optional[list[Any]] = Field(default_factory=list)


class GraphObject(HierachicalPlotObject):
    kind: str = Field(default="GRAPH", init=False)
    content: Optional[list[ComponentObject]] = Field(default_factory=list)


class PageObject(HierachicalPlotObject):
    kind: str = Field(default="PAGE", init=False)
    content: Optional[list[GraphObject]] = Field(default_factory=list)


class DocumentObject(HierachicalPlotObject):
    kind: str = Field(default="DOCUMENT", init=False)
    content: Optional[list[PageObject]] = Field(default_factory=list)


def plot_graphs(
    content: tuple[list[float]],
    component_args: dict[PlotItem, Any] = dict(),
    plot_args: dict[PlotItem, Any] = dict(),
    page_args: dict[PlotItem, Any] = dict(),
    doc_args: dict[PlotItem, Any] = dict(),
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
        # If it is a plottype string value, cast to PlotType
        # TODO This is a workaround becausa properties are of type Any. Need to find a better solution.
        values = [PlotType(val) if val in PlotType else val for val in values]
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
    [plottype] = get_args([PlotItem.PLOTTYPE])
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
            if component_args.get(PlotItem.LABEL, None):
                (handle,) = axis.plot(*content, **kwargs)
                handles.append(handle)
                axis.legend(handles=handles, loc="upper right")
            else:
                axis.plot(*content, **kwargs)
        case PlotType.VLINES:
            # if component_args.get(PlotItem.LABEL, None):
            handle = axis.vlines(*content, **kwargs)
            handles.append(handle)
            axis.legend(handles=handles, loc="upper right")
            # else:
            #     axis.vlines(*content)
        case PlotType.TEXT:
            for item in content:
                axis.text(**item)

        case _:
            ...
    return figure, axes, handles


def create_pdf_with_plots(
    document: DocumentObject,
    *,
    filepath: str,
) -> bool:
    """
    Creates a PDF document and plots the graphs with the given data.

    Args:
        document (DocumentObject): plot data.
                The hierarchy is DocumentObject -> PageObject -> GraphObject -> PlotObject
                Each of these classes has an attribute `content` which is a list containing the
                children objects:
                document.content: list[PageObject]
                page.content: list[GraphObject]
                graph.content: list[PlotObject]
                The PlotObject content attribute contains input data for a specific plot.
                All objects also have a `properties` attribute. These attributes are passed
                as **kwargs to the plot_graphs function.
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
                    if not any(group in document.properties[PlotItem.SKIP_GROUPS] for group in component.groups):
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


def create_animation(
    document: DocumentObject,
    *,
    filepath: str,
): ...


if __name__ == "__main__":
    ...

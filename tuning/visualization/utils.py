import pathlib
from enum import Enum, auto, nonmember
from hashlib import file_digest
from itertools import count
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pydantic import BaseModel, Field, model_validator

from tuning.common.constants import PartialCombiType
from tuning.common.utils import get_logger, open_create_file

logger = get_logger(__name__)


class ComponentType(Enum):
    LINE = "LINE"
    VLINES = "VLINES"
    TEXT = "TEXT"


class PltProperty(Enum):

    # Object types, used to determine
    # if the property should be used as an argo or kwarg
    FIGURE = nonmember("FIGURE")
    AXIS = nonmember("AXIS")
    COMPONENT = nonmember("COMPONENT")
    ARG = nonmember("ARG")
    KWARG = nonmember("KWARG")

    # Actual properties
    # arguments are: value, level indicator if value is kwarg
    ID = "id", None, None
    SEQ = "sequence", None, None
    TITLE = "pagetitle", None, None
    PAGESIZE = "figsize", KWARG, FIGURE
    COMPONENTTYPE = "componenttype", None, None
    PLOTTEXTS = "plottexts", None, None
    NBR_COLUMNS = "nrows", ARG, FIGURE
    NBR_ROWS = "ncols", ARG, FIGURE
    CONSTRAINED_LAYOUT = "constrained_layout", KWARG, FIGURE
    HEIGHT_RATIOS = "height_ratios", KWARG, FIGURE
    AXIS_NOFRAME = "axis no frame", None, None
    AXIS_NOTICKS_X = "axis no ticks x", None, None
    AXIS_NOTICKS_Y = "axis no ticks y", None, None
    COLOR = "color", KWARG, COMPONENT
    LABEL = "label", KWARG, COMPONENT
    XRANGE = "xlim", KWARG, AXIS
    YRANGE = "ylim", KWARG, AXIS
    SKIP = "skip", None, None
    DURATION = "duration", None, None
    INTRO = "intro", None, None

    def __init__(self, label, argtype, artist):
        self.label = label
        self.argtype = argtype
        self.artist = artist


class ContentType(Enum):
    PARTIALS = "partials"
    HARMONICS = "harmonics"

    @classmethod
    def kwargs(cls, d: dict["PltProperty", Any]) -> dict["PltProperty", Any]:
        return {key.value: d[key] for key in PltProperty if key in d}


class GroupName(Enum):
    SOUND_GRAPH = "SOUND_GRAPH"
    DISSONANCE_GRAPH = "DISSONANCE_GRAPH"
    DISSONANCE_ANIMATION_GRAPH = "DISSONANCE_ANIMATION_GRAPH"
    SOUND_DELIMITERS = "SOUND_DELIMITERS"
    DISSONANCE_LINE = "DISSONANCE_LINE"
    DISSONANCE_LINE_PP = "DISSONANCE_LINE_PP"
    DISSONANCE_LINE_PH = "DISSONANCE_LINE_PH"
    DISSONANCE_LINE_HP = "DISSONANCE_LINE_HP"
    DISSONANCE_LINE_NOTES = "DISSONANCE_LINE_NOTES"
    DISSONANCE_TEXT = "DISSONANCE_TEXT"
    DISSONANCE_TEXT_MINIMA = "DISSONANCE_TEXT_MINIMA"
    DISSONANCE_GRAPH_TITLE = "DISSONANCE_GRAPH_TITLE"
    DISSONANCE_NOTELINES = "DISSONANCE_NOTELINES"
    PARTIALS = "PARTIALS"
    HARMONICS = "HARMONICS"
    SPECTRA_PLOT = "SPECTRA_PLOT"
    SPECTRA_PARTIALS = "SPECTRA_PARTIALS"
    SPECTRA_ANALYSIS_PARTIALS = "SPECTRA_ANALYSIS_PARTIALS"
    SPECTRA_AGGR_PARTIALS = "SPECTRA_AGGR_PARTIALS"
    LEGEND = "LEGEND"
    GRAPH_TITLE = "GRAPH_TITLE"

    @classmethod
    def from_enum(cls, value: Enum):
        if isinstance(value, PartialCombiType):
            return {
                PartialCombiType.PARTIALS_HARMONICS: GroupName.DISSONANCE_LINE_PH,
                PartialCombiType.HARMONICS_PARTIALS: GroupName.DISSONANCE_LINE_HP,
                PartialCombiType.PARTIALS_PARTIALS: GroupName.DISSONANCE_LINE_PP,
            }[value]
        else:
            raise Exception(f"Enum type {type(value).__name__} not implemented")


plotobject_counter = count(1)


class PlotObject(BaseModel):
    id: Optional[int] = None
    sequence: Optional[int] = None
    kind: str = Field(default="PLOT", init=False)
    properties: Optional[dict[PltProperty, Any]] = Field(default_factory=dict)
    pdfproperties: Optional[dict[PltProperty, Any]] = Field(default_factory=dict)
    animationproperties: Optional[dict[PltProperty, Any]] = Field(default_factory=dict)
    groups: set[GroupName] = Field(default_factory=set)
    include_only_list: list[GroupName] = list()
    skip_list: list[GroupName] = list()
    skip: bool = False
    content: Optional[Any] = Field(default_factory=list)
    counter__: Callable = Field(default_factory=count, init=False, exclude=True)

    @model_validator(mode="after")
    def __set_id__(self):
        if not self.id:
            self.id = next(plotobject_counter)
            self.properties[PltProperty.ID] = self.id
        return self

    def property_enum_matches(self, proptype: PltProperty, prop: Enum | str, enum_type: type):
        if not isinstance(proptype, PltProperty):
            raise Exception(f"Expected PltProperty but got {type(proptype).__name__}")
        if not (isinstance(prop, str) or isinstance(prop, Enum)):
            raise Exception(f"Expected Enum or string but got {type(prop).__name__}")
        myprop = self.properties.get(proptype, None)
        if not myprop:
            return
        myprop = myprop.value if isinstance(myprop, Enum) else myprop
        prop = prop.value if isinstance(prop, Enum) else prop
        return myprop == prop

    def get_property_value(self, proptype: PltProperty):
        # TODO Flaw in properties attribute: values of this dict are of type Any.
        # As a consequence, when a JSON file is read, str values representing Enum variables
        # will not be cast into the corresponding Enum but will remain of type str.
        myprop = self.properties.get(proptype, None)
        return myprop

    def has_property_value(self, proptype: PltProperty, prop: Any):
        prop = prop.value if isinstance(prop, Enum) else prop
        myprop = myprop.value if isinstance(myprop, Enum) else myprop
        return self.get_property_value(proptype) == prop

    class Config:
        arbitrary_types_allowed = True


class HierachicalPlotObject(PlotObject):
    kind: str = Field(default="GENERIC_HIERARCHCAL_PLOT", init=False)
    children: list["HierachicalPlotObject"]

    def add_child(self, child: PlotObject):
        self.children.append(child)
        child.sequence = next(self.counter__)
        child.properties[PltProperty.SEQ] = child.sequence
        child.include_only_list = self.include_only_list
        child.skip_list = self.skip_list
        # Pass all graph groups to the child, unless the graph group of the child is explicitly set
        child_graph_groups = set([g for g in child.groups if g.value.endswith("GRAPH")])
        if not child_graph_groups:
            parent_graph_groups = set([g for g in self.groups if g.value.endswith("GRAPH")])
            child.groups = child.groups | parent_graph_groups
        if self.skip:
            child.skip = True
        if not child.groups:
            child.skip = False
        else:
            child_in_include_only_list = any([(group in self.include_only_list) for group in child.groups])
            child_in_skip_list = any([(group in self.skip_list) for group in child.groups])
            child.skip = (self.include_only_list and not child_in_include_only_list) or child_in_skip_list

    def last_child(self) -> "PlotObject":
        return self.content[-1] if self.content else None


class ComponentObject(PlotObject):
    kind: str = Field(default="COMPONENT", init=False)


class GraphObject(HierachicalPlotObject):
    kind: str = Field(default="GRAPH", init=False)
    children: Optional[list[ComponentObject]] = Field(default_factory=list)
    sequence: int = None


class PageObject(HierachicalPlotObject):
    kind: str = Field(default="PAGE", init=False)
    children: Optional[list[GraphObject]] = Field(default_factory=list)


class DocumentObject(HierachicalPlotObject):
    kind: str = Field(default="DOCUMENT", init=False)
    children: Optional[list[PageObject]] = Field(default_factory=list)


def plot_component(
    content: tuple[list[float]],
    component_args: dict[PltProperty, Any] = dict(),
    graph_args: dict[PltProperty, Any] = dict(),
    page_args: dict[PltProperty, Any] = dict(),
    doc_args: dict[PltProperty, Any] = dict(),
    figure: Figure = None,
    axes: Axes = None,
    handles: list[Line2D] = None,
    **kwargs,
):
    def get_arg(property: PltProperty, strict: bool = True) -> tuple[Any]:
        """Searches for a property in the document and retrieves it if found.
        If the property occurs in multiple objects in the hierarchy, the property
        of the lowest object in the hierarchy will be returned.

        Args:
            property (Property): The property to search for.
            strict (bool, optional): if true, will raise an exception if the property is not found. Defaults to True.

        Raises:
            Exception: Property not found.

        Returns:
            Any: The property value or None if none was found.
        """
        value = doc_args.get(
            property, component_args.get(property, graph_args.get(property, page_args.get(property, None)))
        )

        # If it is a componenttype string value, cast to ComponentType
        # TODO This is a workaround because properties are of type Any. Need to find a better solution.
        value = ComponentType(value) if str(value) in ComponentType else value
        if strict and value is None:
            raise Exception(f"Missing property {property}")
        return value

    def get_args_kwargs(artist: str) -> dict[str, Any]:
        """Returns all properties that are kwargs for the given artist type

        Args:
            level (PltProperty): the artist ("FIGURE", "AXIS" or "COMPONENT")

        Returns:
            dict[str, Any]: a dict of
        """
        all_args = page_args
        all_args.update(graph_args)
        all_args.update(component_args)
        return (
            [value for arg, value in all_args.items() if arg.argtype == PltProperty.ARG and arg.artist == artist],
            {
                arg.label: value
                for arg, value in all_args.items()
                if arg.argtype == PltProperty.KWARG and arg.artist == artist
            },
        )

    if not figure:
        fig_args, fig_kwargs = get_args_kwargs(PltProperty.FIGURE)
        figure, axes = plt.subplots(*fig_args, **fig_kwargs, dpi=300)
        if pagetitle := page_args.get(PltProperty.TITLE, None):
            figure.suptitle(pagetitle)
    if not handles:
        handles = list()

    axis_list = list(axes.flatten()) if isinstance(axes, np.ndarray) else [axes]
    idx = graph_args.get(PltProperty.SEQ)
    axis: Axis = axis_list[idx]

    # Apply specific axis properties
    if lines := graph_args.get(PltProperty.AXIS_NOFRAME, []):
        for line in lines:
            axis.spines[line].set_visible(False)
    # Remove tick marks
    if graph_args.get(PltProperty.AXIS_NOTICKS_X, False):
        axis.tick_params(bottom=False, top=False)
        axis.set_xticks([])
    if graph_args.get(PltProperty.AXIS_NOTICKS_Y, False):
        axis.tick_params(left=False, right=False)
        axis.set_yticks([])

    # Retrieve and apply settings for the axis
    _, axis_kwargs = get_args_kwargs(PltProperty.AXIS)
    if axis_kwargs:
        plt.setp(axis, **axis_kwargs)

    # Draw the plot
    componenttype = get_arg(PltProperty.COMPONENTTYPE)
    _, compo_kwargs = get_args_kwargs(PltProperty.COMPONENT)
    match componenttype:
        case ComponentType.LINE:
            (handle,) = axis.plot(*content, **compo_kwargs)
            handles.append(handle)
            if component_args.get(PltProperty.LABEL, None):
                axis.legend(handles=handles, loc="upper right")
        case ComponentType.VLINES:
            # if component_args.get(PlotItem.LABEL, None):
            handle = axis.vlines(*content, **kwargs)
            handles.append(handle)
            if component_args.get(PltProperty.LABEL, None):
                axis.legend(handles=handles, loc="upper right")
            # else:
            #     axis.vlines(*content)
        case ComponentType.TEXT:
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
        for page, counter in zip(document.children, count(1)):
            if not page.skip:
                logger.info(f"Plotting page {counter} of {len(document.children)}")
                figure = None
                axes = None
                for graph in [g for g in page.children if not g.skip]:
                    handles = None
                    for component in [c for c in graph.children if not c.skip]:
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
                # plt.show()
                pdf.savefig()
                plt.close()
    logger.info(f"Plot(s) saved to {filepath}")


def create_animation(
    document: DocumentObject,
    *,
    filepath: str,
): ...


if __name__ == "__main__":
    ...

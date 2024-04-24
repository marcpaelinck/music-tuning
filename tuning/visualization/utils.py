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
from pydantic import BaseModel, Field, model_validator

from tuning.common.constants import PartialCombiType
from tuning.common.utils import get_logger, open_create_file

logger = get_logger(__name__)


class ComponentType(Enum):
    LINE = "LINE"
    VLINES = "VLINES"
    TEXT = "TEXT"


class PltProperty(Enum):
    # CONTENT = "content"
    ID = "id"
    SEQ = "sequence"
    TITLE = "pagetitle"
    PAGESIZE = "pagesize"
    COMPONENTTYPE = "componenttype"
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
    SKIP = "skip"
    INCLUDE_ONLY = "include"


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

    def has_property_value(self, proptype: PltProperty, prop: Any):
        # TODO Flaw in properties attribute: values of this dict are of type Any.
        # As a consequence, when a JSON file is read, str values representing Enum variables
        # will not be cast into the corresponding Enum but will remain of type str.
        myprop = self.properties.get(proptype, None)
        if not myprop:
            return False
        myprop = myprop.value if isinstance(myprop, Enum) else myprop
        prop = prop.value if isinstance(prop, Enum) else prop
        return myprop == prop

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
    plot_args: dict[PltProperty, Any] = dict(),
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
            property, component_args.get(property, plot_args.get(property, page_args.get(property, None)))
        )

        # If it is a componenttype string value, cast to ComponentType
        # TODO This is a workaround becausa properties are of type Any. Need to find a better solution.
        value = ComponentType(value) if value in ComponentType else value
        if strict and value is None:
            raise Exception(f"Missing property {property}")
        return value

    if not figure:
        nbr_columns = get_arg(PltProperty.NBR_COLUMNS)
        nbr_rows = get_arg(PltProperty.NBR_ROWS)
        pagesize = get_arg(PltProperty.PAGESIZE, strict=False)
        fig_kwargs = {"figsize": pagesize.value} if pagesize else dict()
        figure, axes = plt.subplots(nbr_rows, nbr_columns, **fig_kwargs)
        if pagetitle := page_args.get(PltProperty.TITLE, None):
            figure.suptitle(pagetitle)
    if not handles:
        handles = list()

    axis_list = list(axes.flatten()) if isinstance(axes, np.ndarray) else [axes]
    idx = plot_args.get(PltProperty.SEQ)
    axis = axis_list[idx]

    # Retrieve and apply settings
    if (xmin := get_arg(PltProperty.XMIN, strict=False)) and (xmax := get_arg(PltProperty.XMAX, strict=False)):
        plt.setp(axis, xlim=(xmin, xmax))
    if (ymin := get_arg(PltProperty.YMIN, strict=False)) and (ymax := get_arg(PltProperty.YMAX, strict=False)):
        plt.setp(axis, ylim=(ymin, ymax))
    if color := component_args.get(PltProperty.COLOR, None):
        kwargs["color"] = color
    if label := component_args.get(PltProperty.LABEL, None):
        kwargs["label"] = label

    # Draw the plot
    componenttype = get_arg(PltProperty.COMPONENTTYPE)
    match componenttype:
        case ComponentType.LINE:
            if component_args.get(PltProperty.LABEL, None):
                (handle,) = axis.plot(*content, **kwargs)
                handles.append(handle)
                axis.legend(handles=handles, loc="upper right")
            else:
                axis.plot(*content, **kwargs)
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
            logger.info(f"Plotting page {counter+1} of {len(document.content)}")
            figure = None
            axes = None
            for graph in page.children:
                handles = None
                for component in graph.children:
                    if not component.skip:
                        # all_groups = page.groups | graph.groups | component.groups
                        # if any(group in document.properties[PltProperty.INCLUDE_ONLY] for group in all_groups) and not any(
                        #     group in document.properties[PltProperty.SKIP] for group in all_groups
                        # ):
                        figure, axes, handles = plot_component(
                            component.content,
                            figure=figure,
                            axes=axes,
                            handles=handles,
                            component_args=component.properties,
                            plot_args=graph.properties,
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

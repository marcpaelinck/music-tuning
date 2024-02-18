import math
from enum import Enum, auto

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from tuning.common.classes import InstrumentGroup
from tuning.common.constants import PageSizes
from tuning.common.utils import get_logger

logger = get_logger(__name__)


class PlotType(Enum):
    PLOT = auto()
    VLINES = auto()


def plot_graphs(
    *args: tuple[list[float]],
    plottype: PlotType = PlotType.PLOT,
    pagetitle: str = "",
    plottitles: list[str] = None,
    xmin: int = None,
    xmax: int = None,
    ymin: int = None,
    ymax: int = None,
    show: bool = False,
    pagesize: PageSizes = PageSizes.A4,
    figure: Figure = None,
    axes: Axes = None,
):
    def set_axes_boundaries(x_index, y_index) -> tuple[float]:
        x_min = xmin or min(min(args[i][x_index]) for i in range(len(args)))
        x_max = xmax or max(max(args[i][x_index]) for i in range(len(args)))
        y_min = ymin or min(min(args[i][y_index]) for i in range(len(args)))
        y_max = ymax or max(max(args[i][y_index]) for i in range(len(args))) * 1.05
        plt.setp(axes, xlim=(x_min, x_max), ylim=(y_min, y_max))
        return x_min, x_max, y_min, y_max

    if not figure:
        figure, axes = plt.subplots(len(args), figsize=pagesize.value)
        if len(args) == 1:
            axes = [axes]
        figure.suptitle(pagetitle)
    x_index = 0
    y_index = 1
    match plottype:
        case PlotType.PLOT:
            xmin, xmax, ymin, ymax = set_axes_boundaries(x_index, y_index)
            vis = [axes[i].plot for i in range(len(args))]
        case PlotType.VLINES:
            vis = [axes[i].vlines for i in range(len(args))]
            [
                axes[i].text(x, y, f"{x:1.3f}", fontsize=6, ha="left")
                for i in range(len(args))
                for (x, y) in zip(args[i][0], args[i][2])
            ]

            y_index = 2
            xmin, xmax, ymin, ymax = set_axes_boundaries(x_index, y_index)
            for arg in args:
                arg[1] = (ymin,) * len(arg[1])
        case _:
            ...

    for idx in range(len(args)):
        if plottitles:
            axes[idx].text(
                x=xmin + 0.01 * (xmax - xmin),
                y=ymax - 0.1 * (ymax - ymin),
                s=plottitles[idx],
                fontsize="small",
                color="b",
            )
        vis[idx](*args[idx])
    if show:
        plt.show()
    return figure, axes


def create_pdf(
    group: InstrumentGroup,
    plotter: callable,
    filepath: str,
    instrumentcodes=None,
    ratio: bool = False,
    **kwargs,
) -> bool:
    with PdfPages(filepath) as pdf:
        logger.info("Generating graphs")
        # Set the maximum x-axis value per instrument type, according
        # to the maximum partial frequency for that type.
        instrument_types = {instr.instrumenttype for instr in group.instruments}
        xmax_dict = {
            instrumenttype: (
                max(
                    math.ceil(p.ratio) if ratio else p.tone.frequency + 100
                    for instr in group.instruments
                    if instr.instrumenttype is instrumenttype
                    for note in instr.notes
                    for p in note.partials
                )
            )
            for instrumenttype in instrument_types
        }
        for instrument in group.instruments:
            if not instrumentcodes or instrument.code in instrumentcodes:
                logger.info(f"--- {group.grouptype.value} {instrument.code}")
                for key in {"xmin", "xmax", "ymin", "ymax"}.intersection(kwargs.keys()):
                    del kwargs[key]
                plotter(
                    group=group,
                    instrument=instrument,
                    ratio=ratio,
                    xmin=0,
                    xmax=xmax_dict[instrument.instrumenttype],
                    ymin=-100,
                    ymax=0,
                    **kwargs,
                )
                pdf.savefig()
                plt.close()
    return True


if __name__ == "__main__":
    ...

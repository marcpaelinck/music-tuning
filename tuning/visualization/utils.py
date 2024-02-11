import logging
from enum import Enum, auto
from itertools import chain

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from tuning.common.classes import InstrumentGroup
from tuning.common.constants import FileType, PageSizes
from tuning.common.utils import get_path

logging.basicConfig(
    format="%(asctime)s - %(name)-12s %(levelname)-7s: %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PlotType(Enum):
    PLOT = auto()
    VLINES = auto()


def plot_graphs(
    *args: tuple[list[float]],
    plottype: PlotType = PlotType.PLOT,
    pagetitle: str = "",
    plottitles: list[str],
    xmin: int = None,
    xmax: int = None,
    ymin: int = None,
    ymax: int = None,
    show: bool = False,
    pagesize: PageSizes = PageSizes.A4,
):
    def set_axes_boundaries(x_index, y_index) -> tuple[float]:
        x_min = xmin or min(min(args[i][x_index]) for i in range(len(args)))
        x_max = xmax or max(max(args[i][x_index]) for i in range(len(args)))
        y_min = ymin or min(min(args[i][y_index]) for i in range(len(args)))
        y_max = ymax or max(max(args[i][y_index]) for i in range(len(args))) * 1.05
        plt.setp(axes, xlim=(x_min, x_max), ylim=(y_min, y_max))
        return x_min, x_max, y_min, y_max

    fig, axes = plt.subplots(len(args), figsize=pagesize.value)
    fig.suptitle(pagetitle)
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


def create_pdf(
    group: InstrumentGroup,
    plotter: callable,
    filepath: str,
    instrumentcodes=None,
    **kwargs,
) -> None:
    with PdfPages(filepath) as pdf:
        logger.info("Generating graphs")
        for instrument in group.instruments:
            if not instrumentcodes or instrument.code in instrumentcodes:
                logger.info(f"--- {group.grouptype.value} {instrument.code}")
                plotter(group=group, instrument=instrument, **kwargs)
                pdf.savefig()
                plt.close()


if __name__ == "__main__":
    ...

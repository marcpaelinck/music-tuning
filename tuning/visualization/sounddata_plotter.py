import math

import numpy as np
from scipy.io import wavfile

from tuning.common.classes import InstrumentGroup
from tuning.common.constants import Folder, PageSizes
from tuning.common.utils import get_logger, get_path
from tuning.visualization.utils import (
    ComponentObject,
    DocumentObject,
    GraphObject,
    PageObject,
    PlotItem,
    PlotType,
    create_pdf_with_plots,
)

logger = get_logger(__name__)


def plot_sounddata(
    group: InstrumentGroup,
    filename: str,
    original_file: bool = True,
) -> bool:
    """
    Creates plots of the sound data with an indication of the detected clippings with note names.

    Args:
        group (InstrumentGroup): The group of instruments to plot.
        filename (str): Name of the PDF file (without path)
        original_file (bool, optional): If True, the original sound files will be used, otherwise
                                        the enhanced files. Defaults to True.
    """

    document = DocumentObject(id=1)

    for instrument in group.instruments:
        document.add_content(page := PageObject())
        page.properties[PlotItem.PAGESIZE] = PageSizes.A4
        page.properties[PlotItem.TITLE] = (
            f"Sound graph for {group.grouptype.value} {instrument.instrumenttype.value} {instrument.code}"
        )

        sample_rate, data = wavfile.read(
            get_path(
                group.grouptype,
                Folder.SOUND,
                instrument.original_soundfilename if original_file else instrument.soundfilename,
            )
        )
        nr_of_channels = data.shape[-1]

        page.properties[PlotItem.XMIN] = 0
        page.properties[PlotItem.XMAX] = math.ceil(len(data) / sample_rate)
        page.properties[PlotItem.YMIN] = ymin = math.floor(np.min(data))
        page.properties[PlotItem.YMAX] = ymax = math.ceil(np.max(data))
        page.properties[PlotItem.NBR_ROWS] = nr_of_channels
        page.properties[PlotItem.NBR_COLUMNS] = 1

        # Stereo or mono
        for channel in range(nr_of_channels):
            page.add_content(graph := GraphObject())

            sounddata = [[i / sample_rate for i in range(len(data[:, channel]))], data[:, channel]]
            graph.add_content(component := ComponentObject())
            component.properties[PlotItem.PLOTTYPE] = PlotType.LINE
            component.content.extend(sounddata)

            cliprange_borders = list(
                zip(
                    *[
                        (border / sample_rate, ymin, ymax, "r")
                        for note in instrument.notes
                        for sample in note.samples
                        for border in (sample.cliprange.start, sample.cliprange.end)
                    ]
                )
            )
            graph.add_content(component := ComponentObject())
            component.properties[PlotItem.PLOTTYPE] = PlotType.VLINES
            component.content.extend(cliprange_borders)

            plottexts = [
                {
                    "x": (sample.cliprange.start + sample.cliprange.end) / (2 * sample_rate),
                    "y": ymax - 0.1 * (ymax - ymin),
                    "s": note.name.short,
                    "horizontalalignment": "center",
                    "fontsize": "small",
                    "color": "b",
                }
                for note in instrument.notes
                for sample in note.samples
            ]
            graph.add_content(component := ComponentObject())
            component.properties[PlotItem.PLOTTYPE] = PlotType.TEXT
            component.content.extend(plottexts)

    filepath = get_path(groupname=group.grouptype, folder=Folder.ANALYSES, filename=filename)
    create_pdf_with_plots(document=document, filepath=filepath)


if __name__ == "__main__":
    ...

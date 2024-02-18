import os
from functools import partial

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tuning.analysis.dissonance import dissonance_j, dissonance_s
from tuning.common.classes import Instrument, InstrumentGroup, Tone
from tuning.common.constants import FileType, InstrumentGroupName
from tuning.common.utils import get_logger, get_path, read_group_from_jsonfile
from tuning.visualization.utils import PlotType, create_pdf, plot_graphs

logger = get_logger(__name__)


def plot_dissonance_functions():
    frequencies = [freq for freq in range(400, 850, 1)]
    g_list = [Tone(frequency=freq, amplitude=1) for freq in frequencies]
    f = Tone(frequency=400, amplitude=1)
    diss_j = [(dissonance_j(f, g)) for g in g_list]
    diss_s = [(dissonance_s(f, g)) for g in g_list]
    plot_graphs((frequencies, diss_j), (frequencies, diss_s), plottype=PlotType.PLOT, show=True)


def plot_notes(
    group: InstrumentGroup,
    instrument: Instrument,
    *,
    what: list[str],
    max_partials: int = 100,
    ratio: bool = False,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    show: bool = False,
):
    plot_content = []
    if "partials" in what:
        partials = [
            list(
                zip(
                    *[
                        (
                            p.ratio if ratio else p.tone.frequency,
                            0,
                            p.tone.amplitude,
                            "r" if p.isfundamental else "y",
                        )
                        for p in note.partials[:max_partials]
                    ]
                )
            )
            for note in instrument.notes
        ]
        plot_content.append((partials, PlotType.VLINES))

    if "spectra" in what:
        spectra = [
            (note.spectrum.frequencies(), note.spectrum.amplitudes()) for note in instrument.notes
        ]
        if ratio:
            fund_freqs = [
                note.partials[note.partial_index].tone.frequency for note in instrument.notes
            ]
            spectra = [
                ([freq / fund_freq for freq in spectrum[0]], spectrum[1])
                for spectrum, fund_freq in zip(spectra, fund_freqs)
            ]
        plot_content.append((spectra, PlotType.PLOT))

    title = f"{group.grouptype.value} - {instrument.instrumenttype} {instrument.code}"

    figure = None
    axes = None
    for content in plot_content:
        is_last = content is plot_content[-1]
        figure, axes = plot_graphs(
            *content[0],
            plottype=content[1],
            pagetitle=title,
            plottitles=[note.name.value for note in instrument.notes] if is_last else None,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            show=show if is_last else False,
            figure=figure,
            axes=axes,
        )


if __name__ == "__main__":

    plot_dissonance_functions()

    # groupname = InstrumentGroupName.SEMAR_PAGULINGAN
    # pdf_filename = "spectra_with_partials_ratio.pdf"

    # filepath = get_path(
    #     groupname=groupname,
    #     filetype=FileType.ANALYSES,
    #     filename=pdf_filename,
    # )
    # # Check if the file is closed. If not, exit.
    # if os.path.exists(filepath):
    #     logger.info(f"Checking if the output file is in use by another process.")
    #     try:
    #         with open(filepath, "r+") as file:
    #             logger.info(f"OK, file not in use.")
    #     except:
    #         logger.error(f"File {filepath} is in use. Please close the file and try again.")
    #         exit()

    # orchestra = read_group_from_jsonfile(groupname, read_sounddata=False, read_spectrumdata=True)

    # create_pdf(
    #     group=orchestra,
    #     plotter=plot_notes,
    #     filepath=filepath,
    #     instrumentcodes=None,
    #     what=["spectra", "partials"],
    #     ratio=True,
    #     xmax=6,
    # )

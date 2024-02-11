from tuning.analysis.dissonance import dissonance_j, dissonance_s
from tuning.common.classes import Instrument, InstrumentGroup
from tuning.common.constants import FileType, InstrumentGroupName
from tuning.common.utils import get_path, read_group_from_jsonfile
from tuning.visualization.utils import PlotType, create_pdf, plot_graphs


def plot_dissonance_functions():
    g_list = list(range(400, 850, 1))
    f = (400, 1)
    diss_j = [(dissonance_j(f, (g, 1))) for g in g_list]
    diss_s = [(dissonance_s(f, (g, 1))) for g in g_list]
    plot_graphs((g_list, diss_j), (g_list, diss_s), plottype=PlotType.PLOT, show=True)


def plot_note_spectra(
    group: InstrumentGroup,
    instrument: Instrument,
    *,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
):
    spectra = [
        (note.spectrum.frequencies(), note.spectrum.amplitudes()) for note in instrument.notes
    ]
    title = f"{group.grouptype.value} - {instrument.code}"
    plot_graphs(
        *spectra,
        plottype=PlotType.PLOT,
        pagetitle=title,
        plottitle=[note.name.value for note in instrument.notes],
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )


def plot_note_partials(
    group: InstrumentGroup,
    instrument: Instrument,
    *,
    max_partials: int = 100,
    ratio: bool = False,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
):
    # define vlines for each note by creating tuples (freq, 0, ampl, color)
    spectra = [
        list(
            zip(
                *[
                    (
                        p.ratio if ratio else p.tone.frequency,
                        0,
                        p.tone.amplitude,
                        "r" if p.isfundamental else "b",
                    )
                    for p in note.partials[:max_partials]
                ]
            )
        )
        for note in instrument.notes
    ]

    title = f"{group.grouptype.value} - {instrument.instrumenttype} {instrument.code}"
    plot_graphs(
        *spectra,
        plottype=PlotType.VLINES,
        pagetitle=title,
        plottitles=[note.name.value for note in instrument.notes],
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )


if __name__ == "__main__":
    groupname = InstrumentGroupName.SEMAR_PAGULINGAN
    orchestra = read_group_from_jsonfile(groupname, read_sounddata=False, read_spectrumdata=False)
    filepath = get_path(
        group=orchestra.grouptype, filetype=FileType.ANALYSES, filename="partial_plots.pdf"
    )
    create_pdf(
        group=orchestra,
        plotter=plot_note_partials,
        filepath=filepath,
        instrumentcodes=None,
        ratio=True,
        xmax=2,
    )

from functools import partial

from tuning.analysis.aggregation import summarize_partials
from tuning.analysis.dissonance import plot_dissonance_graphs
from tuning.analysis.partials import create_partials
from tuning.analysis.spectrum import create_spectra
from tuning.common.classes import AggregatedPartialDict, InstrumentType
from tuning.common.constants import (
    AGGREGATED_PARTIALS_FILE,
    Folder,
    InstrumentGroupName,
)
from tuning.common.utils import (
    get_path,
    read_group_from_jsonfile,
    read_object_from_jsonfile,
    save_group_to_jsonfile,
    save_object_to_jsonfile,
)
from tuning.soundfiles.instrumentinfo import create_group_from_info_file
from tuning.soundfiles.process_soundfiles import process_sound_files
from tuning.visualization.plotter import plot_note_spectra
from tuning.visualization.utils import create_pdf


def pipeline(groupname: InstrumentGroupName):
    orchestra = create_group_from_info_file(groupname)
    # save_group_to_jsonfile(group=orchestra)
    # ---------------------------------------------------------------------------
    # orchestra = read_group_from_jsonfile(groupname)
    process_sound_files(orchestra)
    # save_group_to_jsonfile(group=orchestra)
    # ---------------------------------------------------------------------------
    # orchestra = read_group_from_jsonfile(groupname, read_sounddata=True)
    create_spectra(orchestra)
    # save_group_to_jsonfile(orchestra, save_spectrumdata=True)
    # ---------------------------------------------------------------------------
    # orchestra = read_group_from_jsonfile(groupname=orchestra, read_spectrumdata=True)
    create_partials(orchestra)
    # save_group_to_jsonfile(orchestra)
    # ---------------------------------------------------------------------------
    # orchestra = read_group_from_jsonfile(groupname)
    aggregated_partials = summarize_partials(orchestra)
    save_object_to_jsonfile(
        aggregated_partials,
        get_path(orchestra.grouptype, Folder.ANALYSES, AGGREGATED_PARTIALS_FILE),
    )
    # ---------------------------------------------------------------------------
    # orchestra = read_group_from_jsonfile(groupname)
    pdf_filename = "spectra_with_partials_ratio.pdf"
    create_pdf(
        group=orchestra,
        iterlist=orchestra.instruments,
        plotter=partial(plot_note_spectra, max_partials=7),
        filepath=get_path(groupname=groupname, filetype=Folder.ANALYSES, filename=pdf_filename),
    )
    # ---------------------------------------------------------------------------
    aggregated_partials = read_object_from_jsonfile(
        AggregatedPartialDict, groupname, Folder.ANALYSES, AGGREGATED_PARTIALS_FILE
    )
    pdf_filename = "dissonance_graph.pdf"
    create_pdf(
        group=aggregated_partials,
        iterlist=InstrumentType,
        plotter=plot_dissonance_graphs,
        filepath=get_path(groupname=groupname, filetype=Folder.ANALYSES, filename=pdf_filename),
        groupname=InstrumentGroupName.SEMAR_PAGULINGAN,
    )
    # ---------------------------------------------------------------------------
    save_group_to_jsonfile(orchestra, save_spectrumdata=True)
    # ---------------------------------------------------------------------------


if __name__ == "__main__":
    pipeline(InstrumentGroupName.TEST)

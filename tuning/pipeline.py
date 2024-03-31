from enum import Enum, auto
from functools import partial
from itertools import pairwise

from tuning.analysis.aggregation import create_aggregated_partials
from tuning.analysis.dissonance import plot_dissonance_graphs
from tuning.analysis.partials import create_partials
from tuning.analysis.spectrum import (
    create_spectra,
    create_spectrum,
    create_spectrum_audacity,
)
from tuning.common.classes import AggregatedPartialDict, InstrumentGroup, InstrumentType
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
from tuning.visualization.plotter import plot_note_spectra, plot_sounddata
from tuning.visualization.utils import create_pdf


class Step(Enum):
    READ_GROUP = 1
    READ_GROUP_WITH_SOUNDDATA = auto()
    READ_GROUP_WITH_SPECTRUMDATA = auto()
    READ_AGGREGATED_PARTIALS = auto()
    CREATE_GROUP = auto()
    PROCESS_SOUNDFILES = 100
    CREATE_SPECTRA = auto()
    CREATE_SPECTRA_AUDACITY = auto()
    CREATE_PARTIALS = auto()
    AGGREGATE_PARTIALS = auto()
    SAVE_JSON_ONLY = 200
    SAVE_JSON_AND_SPECTRA = auto()
    SAVE_AGGREGATED_PARTIALS = auto()
    PLOT_SOUNDDATA = 300
    PLOT_NOTE_SPECTRA = auto()
    PLOT_DISSONANCE_GRAPHS = auto()


Pipe = list[Step | list[Step]]


class PipeLine:
    groupname: InstrumentGroupName = None
    pipe: Pipe = None

    # All functions in a pipeline need te have exactly one input argument so that the output of
    # one function can be used as input for the next one. Functions that have been modified in order
    # to comply with this requirement are listed here and are defined in the __init__ step.
    step_read_group_with_sounddata = None
    step_read_group_with_spectrumdata = None
    step_create_spectra = None
    step_create_spectra_audacity = None
    step_save_json_and_spectra = None
    step_save_aggregated_partials = None
    step_read_aggregated_partials = None
    step_plot_sounddata = None
    step_plot_spectra = None
    step_plot_dissonance_graphs = None
    actions = None

    def __init_actions__(self):
        """
        Define the actions of pipeline steps here. The notation of an action is:
            (<function>, <input type>, <output type>)
        Each function should have exactly one input parameter. The <input type>, <output type> elements will
        be used to determine the validity of a sequence (pipe) when a PipeLine object is created. The steps
        in a pipeline should be ordered in such a way that the output of each function can be used as input
        by the next function.
        """
        self.actions = {
            Step.CREATE_GROUP: (create_group_from_info_file, InstrumentGroupName, InstrumentGroup),
            Step.READ_GROUP: (read_group_from_jsonfile, InstrumentGroupName, InstrumentGroup),
            Step.READ_GROUP_WITH_SOUNDDATA: (self.step_read_group_with_sounddata, InstrumentGroup, InstrumentGroup),
            Step.READ_GROUP_WITH_SPECTRUMDATA: (
                self.step_read_group_with_spectrumdata,
                InstrumentGroup,
                InstrumentGroup,
            ),
            Step.READ_AGGREGATED_PARTIALS: (self.step_read_aggregated_partials, InstrumentGroup, AggregatedPartialDict),
            Step.PROCESS_SOUNDFILES: (process_sound_files, InstrumentGroup, InstrumentGroup),
            Step.CREATE_SPECTRA: (self.step_create_spectra, InstrumentGroup, InstrumentGroup),
            Step.CREATE_SPECTRA_AUDACITY: (self.step_create_spectra_audacity, InstrumentGroup, InstrumentGroup),
            Step.CREATE_PARTIALS: (create_partials, InstrumentGroup, InstrumentGroup),
            Step.AGGREGATE_PARTIALS: (create_aggregated_partials, InstrumentGroup, AggregatedPartialDict),
            Step.SAVE_JSON_ONLY: (save_group_to_jsonfile, InstrumentGroup, InstrumentGroup),
            Step.SAVE_JSON_AND_SPECTRA: (self.step_save_json_and_spectra, InstrumentGroup, InstrumentGroup),
            Step.SAVE_AGGREGATED_PARTIALS: (
                self.step_save_aggregated_partials,
                AggregatedPartialDict,
                AggregatedPartialDict,
            ),
            Step.PLOT_SOUNDDATA: (self.step_plot_sounddata, InstrumentGroup, InstrumentGroup),
            Step.PLOT_NOTE_SPECTRA: (self.step_plot_spectra, InstrumentGroup, InstrumentGroup),
            Step.PLOT_DISSONANCE_GRAPHS: (
                self.step_plot_dissonance_graphs,
                AggregatedPartialDict,
                AggregatedPartialDict,
            ),
        }

    def __init__(self, groupname: InstrumentGroupName, pipe: Pipe):
        """
        This method ensures that all functions that are used in a pipeline expect exactly one input parameter.
        Functions that expect more than one argument should be modified by pre-assigning all but one argument.
        If you want to use the same function with different sets of values for several arguments, you will
        need to define separate actions for each combination.
        Args:
            groupname (InstrumentGroupName): This will be the input for the first step of the pipeline.
            pipe (Pipe): A list of Step objects which defines the sequence in which actions should be performed.

        Raises:
            Exception: Illegal sequence if the output and input types of subsequent steps do not match.
        """
        self.groupname = groupname
        self.pipe = pipe

        # Each step function should expect exactly one argument. Pre-assign all other arguments here.
        self.step_read_group_with_sounddata = partial(read_group_from_jsonfile, read_sounddata=True)
        self.step_read_group_with_spectrumdata = partial(read_group_from_jsonfile, read_spectrumdata=True)
        self.step_create_spectra = partial(create_spectra, method=create_spectrum)
        self.step_create_spectra_audacity = partial(create_spectra, spectrum_method=create_spectrum_audacity)
        self.step_save_json_and_spectra = partial(save_group_to_jsonfile, save_spectrumdata=True)
        self.step_save_aggregated_partials = partial(
            save_object_to_jsonfile,
            filepath=get_path(groupname, Folder.ANALYSES, AGGREGATED_PARTIALS_FILE),
        )
        self.step_read_aggregated_partials = partial(
            read_object_from_jsonfile,
            objecttype=AggregatedPartialDict,
            filetype=Folder.ANALYSES,
            filename=AGGREGATED_PARTIALS_FILE,
        )
        self.step_plot_sounddata = partial(
            create_pdf,
            plotter=plot_sounddata,
            filepath=get_path(
                groupname=groupname,
                filetype=Folder.ANALYSES,
                filename="sounddata.pdf",
            ),
        )
        self.step_plot_spectra = partial(
            create_pdf,
            plotter=partial(
                plot_note_spectra,
                max_partials=7,
                ratio=(is_ratio := True),
            ),
            filepath=get_path(
                groupname=groupname,
                filetype=Folder.ANALYSES,
                filename="spectra_with_partials_{type}.pdf".format(type="ratio" if is_ratio else "frequency"),
            ),
        )
        self.step_plot_dissonance_graphs = partial(
            create_pdf,
            iterlist=InstrumentType,
            plotter=plot_dissonance_graphs,
            filepath=get_path(groupname=groupname, filetype=Folder.ANALYSES, filename="dissonance_graph.pdf"),
            groupname=groupname,
        )
        self.__init_actions__()
        if not self.is_valid_pipe(pipe):
            raise Exception("Illegal pipeline sequence")

    def is_valid_pipe(self, pipe: Pipe) -> bool:
        sequence = [self.actions[step] for step in pipe if isinstance(step, Step)]
        return sequence[0][1] is InstrumentGroupName and all(
            [outputtype is inputtype for (_, _, outputtype), (_, inputtype, _) in pairwise(sequence)]
        )

    def execute(self):
        result = self.groupname
        for step in pipe:
            function = self.actions[step][0]
            result = function(result)
        return result


if __name__ == "__main__":
    pipe = [
        Step.CREATE_GROUP,
        Step.PROCESS_SOUNDFILES,
        # Step.SAVE_JSON_ONLY,
        # Step.READ_GROUP_WITH_SOUNDDATA,
        Step.CREATE_SPECTRA_AUDACITY,
        # Step.CREATE_SPECTRA,
        Step.CREATE_PARTIALS,
        Step.SAVE_JSON_AND_SPECTRA,
        Step.PLOT_SOUNDDATA,
        Step.PLOT_NOTE_SPECTRA,
        Step.AGGREGATE_PARTIALS,
        Step.SAVE_AGGREGATED_PARTIALS,
        # Step.READ_AGGREGATED_PARTIALS,
        Step.PLOT_DISSONANCE_GRAPHS,
    ]
    pipeline = PipeLine(InstrumentGroupName.SEMAR_PAGULINGAN, pipe)
    pipeline.execute()

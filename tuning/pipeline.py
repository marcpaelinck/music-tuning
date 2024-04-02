from enum import Enum, auto
from functools import partial
from itertools import pairwise

from pydantic import BaseModel

from tuning.analysis.aggregation import step_create_aggregated_partials_dict
from tuning.analysis.dissonance import plot_dissonance_graphs
from tuning.analysis.partials import step_create_partials
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
    read_object_from_jsonfile,
    save_object_to_jsonfile,
    step_read_group_from_jsonfile,
    step_save_group_to_jsonfile,
)
from tuning.soundfiles.instrumentinfo import step_create_group_from_info_file
from tuning.soundfiles.process_soundfiles import step_process_sound_files
from tuning.visualization.plotter import plot_note_spectra, plot_sounddata
from tuning.visualization.utils import create_pdf


class Step(Enum):
    CREATE_GROUP = 1
    READ_GROUP = auto()
    READ_GROUP_WITH_SOUNDDATA = auto()
    READ_GROUP_WITH_SPECTRUMDATA = auto()
    READ_AGGREGATED_PARTIALS = auto()
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
    PLOT_NOTE_SPECTRA_WITH_AGGREGATED_PARTIALS = auto()
    PLOT_DISSONANCE_GRAPHS = auto()


Pipe = list[Step | list[Step]]


class PipeLine:
    groupname: InstrumentGroupName = None
    data = {}

    pipe: Pipe = None

    # The pipeline enables to execute a sequence of functions that generate output which is used
    # by subsequent functions. The sequence is determined in self.actions.
    # Currently, only data of types InstrumentGroupName, InstrumentGroup or AggregatedPartialDict
    # Can be used as input or output by the functions.
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
            tuple[<function>, dict[<keyword>, <input type>], <output type>]
        The first element is the function that should be called.
        The second element is a dict containing the keyword-type combinations for the function parameters.
        The third element is the output type of the function.
        The <input type>, <output type> elements will be used to select the correct input values for each
        step and to perform a validity check before running the pipeline.
        """
        self.actions = {
            Step.CREATE_GROUP: (
                step_create_group_from_info_file,
                {"groupname": InstrumentGroupName},
                InstrumentGroup,
            ),
            Step.READ_GROUP: (step_read_group_from_jsonfile, {"groupname": InstrumentGroupName}, InstrumentGroup),
            Step.READ_GROUP_WITH_SOUNDDATA: (
                self.step_read_group_with_sounddata,
                {"groupname": InstrumentGroupName},
                InstrumentGroup,
            ),
            Step.READ_GROUP_WITH_SPECTRUMDATA: (
                self.step_read_group_with_spectrumdata,
                {"groupname": InstrumentGroupName},
                InstrumentGroup,
            ),
            Step.READ_AGGREGATED_PARTIALS: (
                self.step_read_aggregated_partials,
                {"groupname": InstrumentGroupName},
                AggregatedPartialDict,
            ),
            Step.PROCESS_SOUNDFILES: (step_process_sound_files, {"group": InstrumentGroup}, InstrumentGroup),
            Step.CREATE_SPECTRA: (self.step_create_spectra, {"group": InstrumentGroup}, InstrumentGroup),
            Step.CREATE_SPECTRA_AUDACITY: (
                self.step_create_spectra_audacity,
                {"group": InstrumentGroup},
                InstrumentGroup,
            ),
            Step.CREATE_PARTIALS: (step_create_partials, {"group": InstrumentGroup}, InstrumentGroup),
            Step.AGGREGATE_PARTIALS: (
                step_create_aggregated_partials_dict,
                {"group": InstrumentGroup},
                AggregatedPartialDict,
            ),
            Step.SAVE_JSON_ONLY: (step_save_group_to_jsonfile, {"group": InstrumentGroup}, InstrumentGroup),
            Step.SAVE_JSON_AND_SPECTRA: (
                self.step_save_json_and_spectra,
                {"group": InstrumentGroup},
                InstrumentGroup,
            ),
            Step.SAVE_AGGREGATED_PARTIALS: (
                self.step_save_aggregated_partials,
                {"object": AggregatedPartialDict},
                AggregatedPartialDict,
            ),
            Step.PLOT_SOUNDDATA: (self.step_plot_sounddata, {"group": InstrumentGroup}, InstrumentGroup),
            Step.PLOT_NOTE_SPECTRA: (
                self.step_plot_spectra,
                {"group": InstrumentGroup, "aggregatedpartials": AggregatedPartialDict},
                InstrumentGroup,
            ),
            Step.PLOT_DISSONANCE_GRAPHS: (
                self.step_plot_dissonance_graphs,
                AggregatedPartialDict,
                AggregatedPartialDict,
            ),
        }

    def __init__(self, groupname: InstrumentGroupName, pipe: Pipe):
        """
        the init method modifies functions so that they only require input of types InstrumentGroupName,
        InstrumentGroup or AggregatedPartialDict. Functions that expect other input types are modified
        by pre-assigning the arguments for these additional data types.
        Args:
            groupname (InstrumentGroupName): This will be the input for the first step of the pipeline.
            pipe (Pipe): A list of Step objects which defines the sequence in which actions should be performed.

        Raises:
            Exception: Illegal sequence if the sequence of the Pipe object is infeasible.
        """
        self.data[InstrumentGroupName] = groupname
        self.pipe = pipe

        # Each step function should expect exactly one argument. Pre-assign all other arguments here.
        self.step_read_group_with_sounddata = partial(step_read_group_from_jsonfile, read_sounddata=True)
        self.step_read_group_with_spectrumdata = partial(step_read_group_from_jsonfile, read_spectrumdata=True)
        self.step_create_spectra = partial(create_spectra, spectrum_method=create_spectrum)
        self.step_create_spectra_audacity = partial(create_spectra, spectrum_method=create_spectrum_audacity)
        self.step_save_json_and_spectra = partial(step_save_group_to_jsonfile, save_spectrumdata=True)
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
        return True
        # sequence = [self.actions[step] for step in pipe if isinstance(step, Step)]
        # return sequence[0][1] is InstrumentGroupName and all(
        #     [outputtype is inputtype for (_, _, outputtype), (_, inputtype, _) in pairwise(sequence)]
        # )

    def __get_data_items__(self, datatypes: dict[str, type]):
        return {keyword: self.data[datatype] for keyword, datatype in datatypes.items()}

    def execute(self):
        result = self.groupname
        for step in pipe:
            function = self.actions[step][0]
            inputdict = self.__get_data_items__(self.actions[step][1])
            result = function(**inputdict)
            self.data[type(result)] = result


if __name__ == "__main__":
    pipe = [
        # Step.CREATE_GROUP,
        # Step.PROCESS_SOUNDFILES,
        # Step.SAVE_JSON_ONLY,
        # Step.READ_GROUP_WITH_SPECTRUMDATA,
        # Step.READ_GROUP_WITH_SOUNDDATA,
        # Step.READ_GROUP,
        # Step.CREATE_SPECTRA_AUDACITY,
        # Step.CREATE_SPECTRA,
        # Step.CREATE_PARTIALS,
        # Step.SAVE_JSON_ONLY,
        # Step.SAVE_JSON_AND_SPECTRA,
        # Step.PLOT_SOUNDDATA,
        Step.READ_AGGREGATED_PARTIALS,
        Step.READ_GROUP_WITH_SPECTRUMDATA,
        Step.PLOT_NOTE_SPECTRA,
        # Step.AGGREGATE_PARTIALS,
        # Step.SAVE_AGGREGATED_PARTIALS,
        # Step.PLOT_DISSONANCE_GRAPHS,
    ]
    pipeline = PipeLine(InstrumentGroupName.SEMAR_PAGULINGAN, pipe)
    pipeline.execute()

from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import Any

from pydantic import BaseModel

from tuning.analysis.aggregation import step_create_aggregated_partials_dict
from tuning.analysis.dissonance_analysis import (
    step_determine_best_matching_dissonance_curves,
    step_perform_consonance_analysis,
)
from tuning.analysis.partials import step_create_partials
from tuning.analysis.spectrum import (
    create_spectra,
    create_spectrum,
    create_spectrum_audacity,
)
from tuning.common.classes import (
    AggregatedPartialDict,
    ConsonanceAnalysis,
    InstrumentGroup,
    InstrumentType,
    NoteName,
)
from tuning.common.constants import (
    AGGREGATED_PARTIALS_FILE,
    CONSONANCE_ANALYSIS_FILE,
    Folder,
    InstrumentGroupName,
)
from tuning.common.utils import (
    get_logger,
    get_path,
    read_object_from_jsonfile,
    save_object_to_jsonfile,
    step_read_group_from_jsonfile,
    step_save_group_to_jsonfile,
)
from tuning.soundfiles.instrumentinfo import step_create_group_from_info_file
from tuning.soundfiles.process_soundfiles import step_process_sound_files
from tuning.visualization.plotter import (
    plot_dissonance_graphs,
    plot_note_spectra,
    plot_sounddata,
)
from tuning.visualization.utils import create_pdf

logger = get_logger(__name__)


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
    AGGREGATE_PARTIALS_ENTIREGROUP = auto()
    SAVE_JSON_ONLY = 200
    SAVE_JSON_AND_SPECTRA = auto()
    SAVE_AGGREGATED_PARTIALS = auto()
    PLOT_SOUNDDATA = 300
    PLOT_NOTE_SPECTRA = auto()
    PLOT_NOTE_SPECTRA_WITH_AGGREGATED_PARTIALS = auto()
    PLOT_DISSONANCE_GRAPHS = auto()
    PLOT_DISSONANCE_RESULTS_OF_ANALYSIS = auto()
    PERFORM_CONSONANCE_ANALYSIS = auto()
    STEP_DETERMINE_BEST_MATCHING_DISSONANCE_CURVES = auto()
    SAVE_CONSONANCE_ANALYSIS = auto()
    READ_CONSONANCE_ANALYSIS = auto()
    PLOT_SPECTRA_OF_ANALYSIS = auto()


@dataclass
class Action:
    method: callable
    expected_input: dict[Any]
    output: BaseModel


Pipe = list[Step]


class PipeLine:
    """
    The pipeline enables to execute a sequence of functions that generate output which is used
    by subsequent functions. The sequence is determined in self.actions.
    Currently, only data of types InstrumentGroupName, InstrumentGroup or AggregatedPartialDict
    Can be used as input or output by the functions.

    Returns:
        PipeLine: A PipeLine object.
    """

    groupname: InstrumentGroupName = None
    data = {}

    pipe: Pipe = None
    actions: dict[Step, Action]

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
            Step.CREATE_GROUP: Action(
                method=step_create_group_from_info_file,
                expected_input={"groupname": InstrumentGroupName},
                output=InstrumentGroup,
            ),
            Step.READ_GROUP: Action(
                method=step_read_group_from_jsonfile,
                expected_input={"groupname": InstrumentGroupName},
                output=InstrumentGroup,
            ),
            Step.READ_GROUP_WITH_SOUNDDATA: Action(
                method=partial(step_read_group_from_jsonfile, read_sounddata=True),
                expected_input={"groupname": InstrumentGroupName},
                output=InstrumentGroup,
            ),
            Step.READ_GROUP_WITH_SPECTRUMDATA: Action(
                method=partial(step_read_group_from_jsonfile, read_spectrumdata=True),
                expected_input={"groupname": InstrumentGroupName},
                output=InstrumentGroup,
            ),
            Step.READ_AGGREGATED_PARTIALS: Action(
                method=partial(
                    read_object_from_jsonfile,
                    objecttype=AggregatedPartialDict,
                    folder=Folder.ANALYSES,
                    filename=AGGREGATED_PARTIALS_FILE,
                ),
                expected_input={"groupname": InstrumentGroupName},
                output=AggregatedPartialDict,
            ),
            Step.PROCESS_SOUNDFILES: Action(
                method=step_process_sound_files,
                expected_input={"group": InstrumentGroup},
                output=InstrumentGroup,
            ),
            Step.CREATE_SPECTRA: Action(
                method=partial(create_spectra, spectrum_method=create_spectrum),
                expected_input={"group": InstrumentGroup},
                output=InstrumentGroup,
            ),
            Step.CREATE_SPECTRA_AUDACITY: Action(
                method=partial(create_spectra, spectrum_method=create_spectrum_audacity),
                expected_input={"group": InstrumentGroup},
                output=InstrumentGroup,
            ),
            Step.CREATE_PARTIALS: Action(
                method=step_create_partials,
                expected_input={"group": InstrumentGroup},
                output=InstrumentGroup,
            ),
            Step.AGGREGATE_PARTIALS: Action(
                method=step_create_aggregated_partials_dict,
                expected_input={"group": InstrumentGroup},
                output=AggregatedPartialDict,
            ),
            Step.AGGREGATE_PARTIALS_ENTIREGROUP: Action(
                method=partial(
                    step_create_aggregated_partials_dict,
                    aggregation=(),
                ),
                expected_input={"group": InstrumentGroup},
                output=AggregatedPartialDict,
            ),
            Step.SAVE_JSON_ONLY: Action(
                method=step_save_group_to_jsonfile,
                expected_input={"group": InstrumentGroup},
                output=InstrumentGroup,
            ),
            Step.SAVE_JSON_AND_SPECTRA: Action(
                method=partial(step_save_group_to_jsonfile, save_spectrumdata=True),
                expected_input={"group": InstrumentGroup},
                output=InstrumentGroup,
            ),
            Step.SAVE_AGGREGATED_PARTIALS: Action(
                method=partial(
                    save_object_to_jsonfile,
                    folder=Folder.ANALYSES,
                    filename=AGGREGATED_PARTIALS_FILE,
                ),
                expected_input={"object": AggregatedPartialDict, "groupname": InstrumentGroupName},
                output=AggregatedPartialDict,
            ),
            Step.PLOT_SOUNDDATA: Action(
                method=partial(
                    create_pdf,
                    plotter=plot_sounddata,
                    folder=Folder.ANALYSES,
                    filename="sounddata.pdf",
                ),
                expected_input={"group": InstrumentGroup, "groupname": InstrumentGroupName},
                output=InstrumentGroup,
            ),
            Step.PLOT_NOTE_SPECTRA: Action(
                method=partial(
                    create_pdf,
                    plotter=partial(
                        plot_note_spectra,
                        max_partials=7,
                        ratio=(is_ratio := True),
                        what=["spectra", "aggregatedpartials"],
                    ),
                    folder=Folder.ANALYSES,
                    filename="spectra_with_partials_{type}.pdf".format(type="ratio" if is_ratio else "frequency"),
                ),
                expected_input={
                    "group": InstrumentGroup,
                    "groupname": InstrumentGroupName,
                    "aggregatedpartials": AggregatedPartialDict,
                },
                output=InstrumentGroup,
            ),
            Step.PLOT_DISSONANCE_GRAPHS: Action(
                method=partial(
                    create_pdf,
                    plotter=plot_dissonance_graphs,
                    iterlist=InstrumentType,
                    folder=Folder.ANALYSES,
                    filename="dissonance_graph-{ref_note}{ref_octave}{aggregation}.pdf",
                    ref_octave=2,
                    ref_note=NoteName.DING,
                ),
                expected_input={
                    "group": AggregatedPartialDict,
                    "groupname": InstrumentGroupName,
                },
                output=AggregatedPartialDict,
            ),
            Step.PLOT_DISSONANCE_RESULTS_OF_ANALYSIS: Action(
                method=partial(
                    create_pdf,
                    plotter=plot_dissonance_graphs,
                    iterlist=InstrumentType,
                    folder=Folder.ANALYSES,
                    filename="analysis_graph-{ref_note}{ref_octave}{aggregation}.pdf",
                ),
                expected_input={
                    "group": AggregatedPartialDict,
                    "groupname": InstrumentGroupName,
                    "analysis_results": ConsonanceAnalysis,
                },
                output=AggregatedPartialDict,
            ),
            Step.PLOT_SPECTRA_OF_ANALYSIS: Action(
                method=partial(
                    create_pdf,
                    plotter=partial(
                        plot_note_spectra,
                        ratio=(is_ratio := True),
                        what=["spectra", "analysis-partials"],
                    ),
                    folder=Folder.ANALYSES,
                    filename="analysis_spectra.pdf",
                ),
                expected_input={
                    "group": InstrumentGroup,
                    "groupname": InstrumentGroupName,
                    "analysis": ConsonanceAnalysis,
                },
                output=InstrumentGroup,
            ),
            Step.PERFORM_CONSONANCE_ANALYSIS: Action(
                method=step_perform_consonance_analysis,
                expected_input={"aggregatedpartials": AggregatedPartialDict},
                output=ConsonanceAnalysis,
            ),
            Step.STEP_DETERMINE_BEST_MATCHING_DISSONANCE_CURVES: Action(
                method=step_determine_best_matching_dissonance_curves,
                expected_input={"analysis": ConsonanceAnalysis},
                output=ConsonanceAnalysis,
            ),
            Step.SAVE_CONSONANCE_ANALYSIS: Action(
                method=partial(
                    save_object_to_jsonfile,
                    folder=Folder.ANALYSES,
                    filename=CONSONANCE_ANALYSIS_FILE,
                ),
                expected_input={"object": ConsonanceAnalysis, "groupname": InstrumentGroupName},
                output=ConsonanceAnalysis,
            ),
            Step.READ_CONSONANCE_ANALYSIS: Action(
                method=partial(
                    read_object_from_jsonfile,
                    objecttype=ConsonanceAnalysis,
                    folder=Folder.ANALYSES,
                    filename=CONSONANCE_ANALYSIS_FILE,
                ),
                expected_input={"groupname": InstrumentGroupName},
                output=ConsonanceAnalysis,
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

        self.__init_actions__()
        if not self.is_valid_pipe(pipe):
            logger.error(f"Aborting script.")
            exit()

    def is_valid_pipe(self, pipe: Pipe) -> bool:
        available_data = {item.__class__ for item in self.data.values()}
        for step in pipe:
            action = self.actions[step]
            if not all([item in available_data for item in action.expected_input.values()]):
                logger.error(f"Invalid pipeline sequence: not all input is available for step {step}.")
                return False
            if action.output:
                available_data.add(action.output)
        return True

    def __get_data_items__(self, datatypes: dict[str, type]):
        return {keyword: self.data[datatype] for keyword, datatype in datatypes.items()}

    def execute(self):
        logger.info(f"Starting pipeline for {self.data[InstrumentGroupName].value}")
        for step in pipe:
            function = self.actions[step].method
            inputdict = self.__get_data_items__(self.actions[step].expected_input)
            result = function(**inputdict)
            self.data[type(result)] = result


if __name__ == "__main__":
    pipe = [
        # Step.CREATE_GROUP,
        # Step.PROCESS_SOUNDFILES,
        # Step.SAVE_JSON_ONLY,
        # Step.PLOT_SOUNDDATA,
        Step.READ_GROUP_WITH_SPECTRUMDATA,
        # Step.READ_GROUP_WITH_SOUNDDATA,
        # Step.READ_GROUP,
        # Step.CREATE_SPECTRA_AUDACITY,
        # Step.CREATE_SPECTRA,
        # Step.CREATE_PARTIALS,
        # Step.SAVE_JSON_ONLY,
        # Step.SAVE_JSON_AND_SPECTRA,
        # Step.READ_AGGREGATED_PARTIALS,
        # Step.READ_GROUP_WITH_SPECTRUMDATA,
        # Step.AGGREGATE_PARTIALS,
        # Step.AGGREGATE_PARTIALS_ENTIREGROUP,
        # Step.OPTIMIZE_DISSONANCE_GRAPH,
        # Step.SAVE_AGGREGATED_PARTIALS,
        # Step.PLOT_NOTE_SPECTRA,
        Step.READ_CONSONANCE_ANALYSIS,
        Step.PLOT_SPECTRA_OF_ANALYSIS,
        # Step.PLOT_DISSONANCE_GRAPHS,
    ]
    # pipe = [
    #     Step.READ_AGGREGATED_PARTIALS,
    #     Step.PLOT_DISSONANCE_GRAPHS,
    # ]
    # pipe = [
    #     Step.READ_GROUP,
    #     Step.AGGREGATE_PARTIALS_ENTIREGROUP,
    #     Step.PLOT_DISSONANCE_GRAPHS,
    # ]
    # pipe = [
    #     Step.READ_GROUP,
    #     Step.AGGREGATE_PARTIALS_ENTIREGROUP,
    #     # Step.PERFORM_CONSONANCE_ANALYSIS,
    #     Step.READ_CONSONANCE_ANALYSIS,
    #     Step.STEP_DETERMINE_BEST_MATCHING_DISSONANCE_CURVES,
    #     Step.SAVE_CONSONANCE_ANALYSIS,
    #     Step.PLOT_DISSONANCE_RESULTS_OF_ANALYSIS,
    # ]
    pipeline = PipeLine(InstrumentGroupName.ANGKLUNG, pipe)
    pipeline.execute()

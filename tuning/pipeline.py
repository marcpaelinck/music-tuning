from dataclasses import dataclass, field
from enum import Enum
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
    Instrument,
    InstrumentGroup,
    InstrumentType,
    Note,
    NoteName,
)
from tuning.common.constants import (
    AGGREGATED_PARTIALS_FILE,
    CONSONANCE_ANALYSIS_FILE,
    INSTRUMENTTYPE,
    OMBAKTYPE,
    Folder,
    InstrumentGroupName,
)
from tuning.common.utils import (
    get_logger,
    read_object_from_jsonfile,
    save_object_to_jsonfile,
    step_read_group_from_jsonfile,
    step_save_group_to_jsonfile,
)
from tuning.soundfiles.instrumentinfo import step_create_group_from_info_file
from tuning.soundfiles.process_soundfiles import step_process_sound_files
from tuning.visualization.dissonance_plotter import plot_dissonance_graphs
from tuning.visualization.sounddata_plotter import plot_sounddata
from tuning.visualization.spectra_plotter import plot_note_spectra

logger = get_logger("Pipeline")


@dataclass
class Action:
    """
    This class defines actions that correspond with each step of a pipeline (see class Step below).
    method: function that should be called.
    expected_input: dict containing {keyword: type} combinations representing arguments for the function.
                    This parameter should only be used for output from a previous step in the pipeline that
                    should be passed to the function. The value of `expected_input` will be passed as **kwargs
                    during the execution of the pipeline, whereby the type indicator will be replaced
                    with the actual output of a previous step.
                    Other arguments of the function should be pre-defined using the Python `partial` function.
    output: the output type of the function.
    paramters: additional parameters for the method that can be defined before executing the pipeline.
    The values `expected_input` and `output` will be used to perform a validity check before running the pipeline.
    Currently, these values are restricted to Enum and Pydantic BaseModel types, but this is not strictly necessary.
    """

    method: callable
    expected_input: dict[str, Enum | BaseModel]
    output: BaseModel
    parameters: dict[str, Any] = field(default_factory=dict)


class Step(Enum):
    """
    All possible steps that can be combined in a pipeline, with associated action.
    """

    CREATE_INSTRUMENT_GROUP = Action(
        method=step_create_group_from_info_file,
        expected_input={"groupname": InstrumentGroupName},
        output=InstrumentGroup,
    )
    READ_INSTRUMENT_GROUP = Action(
        method=step_read_group_from_jsonfile,
        expected_input={"groupname": InstrumentGroupName},
        output=InstrumentGroup,
    )
    READ_GROUP_WITH_SOUNDDATA = Action(
        method=step_read_group_from_jsonfile,
        parameters={"read_sounddata": True},
        expected_input={"groupname": InstrumentGroupName},
        output=InstrumentGroup,
    )
    READ_GROUP_WITH_SPECTRUMDATA = Action(
        method=step_read_group_from_jsonfile,
        parameters={"read_spectrumdata": True},
        expected_input={"groupname": InstrumentGroupName},
        output=InstrumentGroup,
    )
    READ_AGGREGATED_PARTIALS = Action(
        method=read_object_from_jsonfile,
        parameters={
            "objecttype": AggregatedPartialDict,
            "folder": Folder.ANALYSES_PARTIALS,
            "filename": AGGREGATED_PARTIALS_FILE,
        },
        expected_input={"groupname": InstrumentGroupName},
        output=AggregatedPartialDict,
    )
    PROCESS_SOUNDFILES = Action(
        method=step_process_sound_files,
        expected_input={"group": InstrumentGroup},
        output=InstrumentGroup,
    )
    CREATE_SPECTRA = Action(
        method=create_spectra,
        parameters={"spectrum_method": create_spectrum},
        expected_input={"group": InstrumentGroup},
        output=InstrumentGroup,
    )
    CREATE_SPECTRA_AUDACITY = Action(
        method=create_spectra,
        parameters={"spectrum_method": create_spectrum_audacity},
        expected_input={"group": InstrumentGroup},
        output=InstrumentGroup,
    )
    CREATE_PARTIALS = Action(
        method=step_create_partials,
        expected_input={"group": InstrumentGroup},
        output=InstrumentGroup,
    )
    AGGREGATE_PARTIALS = Action(
        method=step_create_aggregated_partials_dict,
        parameters={"aggregation": (INSTRUMENTTYPE, OMBAKTYPE)},
        expected_input={"group": InstrumentGroup},
        output=AggregatedPartialDict,
    )
    AGGREGATE_PARTIALS_ENTIREGROUP = Action(
        method=step_create_aggregated_partials_dict,
        parameters={"aggregation": ()},
        expected_input={"group": InstrumentGroup},
        output=AggregatedPartialDict,
    )
    SAVE_INSTRUMENT_GROUP_WITHOUT_SPECTRA = Action(
        method=step_save_group_to_jsonfile,
        expected_input={"group": InstrumentGroup},
        output=InstrumentGroup,
    )
    SAVE_INSTRUMENT_GROUP_JSON_AND_SPECTRA = Action(
        method=step_save_group_to_jsonfile,
        parameters={"save_spectrumdata": True},
        expected_input={"group": InstrumentGroup},
        output=InstrumentGroup,
    )
    SAVE_AGGREGATED_PARTIALS = Action(
        method=save_object_to_jsonfile,
        parameters={"folder": Folder.ANALYSES_PARTIALS, "filename": AGGREGATED_PARTIALS_FILE},
        expected_input={"object": AggregatedPartialDict, "groupname": InstrumentGroupName},
        output=AggregatedPartialDict,
    )
    PLOT_SOUNDDATA = Action(
        method=plot_sounddata,
        parameters={"filename": "sounddata.pdf"},
        expected_input={"group": InstrumentGroup},
        output=InstrumentGroup,
    )
    PLOT_NOTE_SPECTRA = Action(
        method=plot_note_spectra,
        parameters={
            "max_partials": 10,
            "ratio": (is_ratio := True),
            "what": ["spectra", "aggregatedpartials"],
            "filename": "spectra_with_partials_{type}.pdf".format(type="ratio" if is_ratio else "frequency"),
            "pageby": Note,
        },
        expected_input={
            "group": InstrumentGroup,
            "groupname": InstrumentGroupName,
            "aggregatedpartials": AggregatedPartialDict,
        },
        output=InstrumentGroup,
    )
    PLOT_DISSONANCE_GRAPHS = Action(
        method=plot_dissonance_graphs,
        parameters={
            "filename": "dissonance_graph-{ref_note}{ref_octave}_aggr[{aggregation}].pdf",
            "ref_octave": 2,
            "ref_note": NoteName.DING,
        },
        expected_input={
            "aggr_partials": AggregatedPartialDict,
            "groupname": InstrumentGroupName,
        },
        output=AggregatedPartialDict,
    )
    PLOT_DISSONANCE_RESULTS_OF_ANALYSIS = Action(
        method=plot_dissonance_graphs,
        parameters={
            "iterlist": InstrumentType,
            "filename": "analysis_graph-{ref_note}{ref_octave}_aggr[{aggregation}].pdf",
        },
        expected_input={
            "aggr_partials": AggregatedPartialDict,
            "groupname": InstrumentGroupName,
            "analysis_results": ConsonanceAnalysis,
        },
        output=AggregatedPartialDict,
    )
    PLOT_SPECTRA_OF_ANALYSIS = Action(
        method=plot_note_spectra,
        parameters={
            "ratio": True,
            "what": ["spectra", "analysis-partials"],
            "filename": "analysis_spectra.pdf",
            "pageby": Instrument,
        },
        expected_input={
            "group": InstrumentGroup,
            "groupname": InstrumentGroupName,
            "analysis": ConsonanceAnalysis,
        },
        output=InstrumentGroup,
    )
    PERFORM_CONSONANCE_ANALYSIS = Action(
        method=step_perform_consonance_analysis,
        expected_input={"aggregatedpartials": AggregatedPartialDict},
        output=ConsonanceAnalysis,
    )
    DETERMINE_BEST_MATCHING_DISSONANCE_CURVES = Action(
        method=step_determine_best_matching_dissonance_curves,
        expected_input={"analysis": ConsonanceAnalysis},
        output=ConsonanceAnalysis,
    )
    SAVE_CONSONANCE_ANALYSIS = Action(
        method=save_object_to_jsonfile,
        parameters={"folder": Folder.ANALYSES_DISS, "filename": CONSONANCE_ANALYSIS_FILE},
        expected_input={"object": ConsonanceAnalysis, "groupname": InstrumentGroupName},
        output=ConsonanceAnalysis,
    )
    READ_CONSONANCE_ANALYSIS = Action(
        method=read_object_from_jsonfile,
        parameters={
            "objecttype": ConsonanceAnalysis,
            "folder": Folder.ANALYSES_DISS,
            "filename": CONSONANCE_ANALYSIS_FILE,
        },
        expected_input={"groupname": InstrumentGroupName},
        output=ConsonanceAnalysis,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action = args[0]  # Better name to improve code readability.

    def params(self, p):
        # Updates/overwrites the parameters of the action.
        self.action.parameters.update(p)
        return self


class PipeLine:
    """
    The pipeline enables to execute a sequence of functions that generate output for subsequent
    functions. The sequence is determined in the `pipe` parameter.
    Currently, only data of types InstrumentGroupName, InstrumentGroup or AggregatedPartialDict
    Can be passed from one function to the other.
    Additional pre-set parameters can be set through the Step objects in the Step class definition.
    These parameters can later be overuled by calling the `param` method of a Step object.

    Returns:
        PipeLine: A PipeLine object.
    """

    groupname: InstrumentGroupName = None
    pipe: list[Step] = None
    data = dict()

    def __init__(self, groupname: InstrumentGroupName, pipe: list[Step]):
        """
        Args:
            groupname (InstrumentGroupName): The instrument group. The first step action will receive this value
                                             as input and should not expect any other input.
            pipe (Pipe): A list of Step values which defines the sequence in which actions should be performed.
            Warning: Illegal sequence if the sequence of the Pipe object is infeasible. This is the case
                     if an action expects input that is not created by any preceding step.
        """
        self.data[InstrumentGroupName] = groupname
        self.pipe = pipe

        if not self.is_valid_pipe(pipe):
            logger.error(f"Aborting script.")
            exit()

    def is_valid_pipe(self, pipe: list[Step]) -> bool:
        available_data = {item.__class__ for item in self.data.values()}
        for step in pipe:
            if not all([item in available_data for item in step.action.expected_input.values()]):
                logger.error(f"Invalid pipeline sequence: not all input is available for step {step.name}.")
                return False
            if step.action.output:
                available_data.add(step.action.output)
        return True

    def __get_data_items__(self, datatypes: dict[str, type]):
        return {keyword: self.data[datatype] for keyword, datatype in datatypes.items()}

    def execute(self):
        logger.info(f"Starting pipeline for {self.data[InstrumentGroupName].value}")
        for step in self.pipe:
            function = step.action.method
            inputdict = self.__get_data_items__(step.action.expected_input)
            result = function(**inputdict, **step.action.parameters)
            self.data[type(result)] = result

    # Step.CREATE_INSTRUMENT_GROUP,
    # Step.READ_INSTRUMENT_GROUP,
    # Step.READ_GROUP_WITH_SOUNDDATA,
    # Step.READ_GROUP_WITH_SPECTRUMDATA,
    # Step.READ_AGGREGATED_PARTIALS,
    # Step.PROCESS_SOUNDFILES,
    # Step.CREATE_SPECTRA,
    # Step.CREATE_SPECTRA_AUDACITY,
    # Step.CREATE_PARTIALS,
    # Step.AGGREGATE_PARTIALS,
    # Step.AGGREGATE_PARTIALS_ENTIREGROUP,
    # Step.SAVE_INSTRUMENT_GROUP_WITHOUT_SPECTRA,
    # Step.SAVE_INSTRUMENT_GROUP_JSON_AND_SPECTRA,
    # Step.SAVE_AGGREGATED_PARTIALS,
    # Step.PLOT_SOUNDDATA,
    # Step.is_ratio,
    # Step.PLOT_NOTE_SPECTRA,
    # Step.PLOT_DISSONANCE_GRAPHS,
    # Step.PLOT_DISSONANCE_RESULTS_OF_ANALYSIS,
    # Step.PLOT_SPECTRA_OF_ANALYSIS,
    # Step.PERFORM_CONSONANCE_ANALYSIS,
    # Step.STEP_DETERMINE_BEST_MATCHING_DISSONANCE_CURVES,
    # Step.SAVE_CONSONANCE_ANALYSIS,
    # Step.READ_CONSONANCE_ANALYSIS,


PIPE_AGGREGATE_PARTIALS_AND_SAVE = [
    Step.READ_INSTRUMENT_GROUP,
    Step.AGGREGATE_PARTIALS,
    Step.SAVE_AGGREGATED_PARTIALS,
]
PIPE_PLOT_DISS_GRAPH = [
    Step.READ_AGGREGATED_PARTIALS,
    Step.PLOT_DISSONANCE_GRAPHS,
    # Step.PLOT_DISSONANCE_GRAPHS.params({"filename": "dissonance_graph-DING2_aggr[InstrumentType-OmbakType](1).pdf"}),
]
PIPE_PLOT_NOTE_SPECTRA = [
    Step.READ_GROUP_WITH_SPECTRUMDATA,
    Step.READ_AGGREGATED_PARTIALS,
    Step.PLOT_NOTE_SPECTRA,
]
PIPE_PLOT_SPECTRA_OF_ANALYSIS = [
    Step.READ_GROUP_WITH_SPECTRUMDATA,
    Step.READ_CONSONANCE_ANALYSIS,
    Step.PLOT_SPECTRA_OF_ANALYSIS,
]
PIPE_PERFORM_ENTIRE_ANALYSIS = [
    Step.READ_INSTRUMENT_GROUP,
    Step.AGGREGATE_PARTIALS_ENTIREGROUP,
    Step.PERFORM_CONSONANCE_ANALYSIS,
    Step.SAVE_CONSONANCE_ANALYSIS,
]
PIPE_OPTIMIZE_AND_PLOT = [
    Step.READ_INSTRUMENT_GROUP,
    Step.AGGREGATE_PARTIALS_ENTIREGROUP,
    Step.READ_CONSONANCE_ANALYSIS,
    Step.DETERMINE_BEST_MATCHING_DISSONANCE_CURVES,
    Step.SAVE_CONSONANCE_ANALYSIS,
    Step.PLOT_DISSONANCE_RESULTS_OF_ANALYSIS,
]
PIPE_PLOT_DISS_OF_OPTIMUM = [
    Step.READ_INSTRUMENT_GROUP,
    Step.AGGREGATE_PARTIALS_ENTIREGROUP,
    Step.READ_CONSONANCE_ANALYSIS,
    Step.PLOT_DISSONANCE_RESULTS_OF_ANALYSIS,
]
PIPE_PLOT_SOUND_GRAPH = [
    Step.READ_GROUP_WITH_SOUNDDATA,
    Step.PLOT_SOUNDDATA,
]

if __name__ == "__main__":
    pipeline = PipeLine(InstrumentGroupName.ANGKLUNG, PIPE_PLOT_SOUND_GRAPH)
    pipeline.execute()

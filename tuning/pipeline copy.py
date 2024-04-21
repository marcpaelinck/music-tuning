""" This module enables the user to create a 'pipeline', which is a list of actions
that should be performed sequentially.
e.g. :  1. Read and process soundfiles 
        2. Create frequency spectra
        3. Determine partials
        4. Create and plot a dissonance graph
Most Actions produce output that can be used by subsequent actions in the pipeline.
Before running a sequence, the PipeLine class checks its validity by determining whether
the input that is required in each step is available from previous steps. The required
input for each step can consist of multiple values or objects.
The application mostly uses Pydantic object structures to pass the data from one
step to the other. The output of intermediate steps can be saved to JSON files and retrieved
later. This enables to read and use this data at the start of a pipeline instead of
re-running the steps that were used to generate this data.
"""

import inspect
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

from tuning.analysis.aggregation import step_create_aggregated_partials_dict
from tuning.analysis.dissonance_analysis import (
    step_determine_best_matching_dissonance_curves,
    step_perform_consonance_analysis,
)
from tuning.analysis.partials import step_create_partials
from tuning.analysis.spectrum import (
    create_spectrum,
    create_spectrum_audacity,
    step_create_spectra,
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
    step_read_group_from_jsonfile,
    step_read_object_from_jsonfile,
    step_save_group_to_jsonfile,
    step_save_object_to_jsonfile,
)
from tuning.soundfiles.instrumentinfo import step_create_group_from_info_file
from tuning.soundfiles.process_soundfiles import step_process_sound_files
from tuning.visualization.dissonance_plotter import step_plot_dissonance_graphs
from tuning.visualization.sounddata_plotter import step_plot_sounddata
from tuning.visualization.spectra_plotter import step_plot_note_spectra
from tuning.visualization.utils import GroupName

logger = get_logger("Pipeline")


@dataclass
class Action:
    """
    This class defines actions that correspond with each step of a pipeline (see class Step below).
    Args:
    method (callable): The function that should be called.
    expected_input: (dict[str, Enum | BaseModel], optional) dict containing {keyword: type} combinations
            representing arguments for the function. This parameter should only be used to determine which
            output from previous steps in the pipeline should be passed to the function.
            The value of `expected_input` will be passed as **kwargs during the execution of the pipeline,
            whereby the type indicator will be replaced with the actual output of a previous step.()
            Function arguments of which the value is known in advance (e.g. a file name) should be
            defined with the `parameters` argument.
    parameters: Function parameters of which the value is known in advance. E.g. a file name or a requested
            number of
    output: the output type of the function.
    paramters: additional parameters for the method that can be defined before executing the pipeline.
    The values `expected_input` and `output` will be used to perform a validity check before running the pipeline.
    Currently, these values are restricted to Enum and Pydantic BaseModel types, but this is not strictly necessary.
    """

    name: str
    method: callable
    parameters: Optional[dict[str, Any]] = field(default_factory=dict)
    expected_input: Optional[dict[str, Enum | BaseModel]] = field(default_factory=dict)
    output: Optional[BaseModel] = field(default_factory=dict)

    def params(self, p):
        # Updates/overwrites the parameters of the action.
        self.parameters.update(p)
        return self


class Step(Enum):
    """
    All possible steps that can be combined in a pipeline, with associated action.
    """

    def CREATE_INSTRUMENT_GROUP():
        return Action(
            method=step_create_group_from_info_file,
            expected_input={"groupname": InstrumentGroupName},
            output=InstrumentGroup,
        )

    def READ_INSTRUMENT_GROUP():
        return Action(
            name=sys._getframe().f_code.co_name,
            method=step_read_group_from_jsonfile,
            expected_input={"groupname": InstrumentGroupName},
            output=InstrumentGroup,
        )

    def READ_GROUP_WITH_SOUNDDATA():
        return Action(
            method=step_read_group_from_jsonfile,
            parameters={"read_sounddata": True},
            expected_input={"groupname": InstrumentGroupName},
            output=InstrumentGroup,
        )

    def READ_GROUP_WITH_SPECTRUMDATA():
        return Action(
            method=step_read_group_from_jsonfile,
            parameters={"read_spectrumdata": True},
            expected_input={"groupname": InstrumentGroupName},
            output=InstrumentGroup,
        )

    def READ_AGGREGATED_PARTIALS():
        return Action(
            method=step_read_object_from_jsonfile,
            parameters={
                "objecttype": AggregatedPartialDict,
                "folder": Folder.ANALYSES_PARTIALS,
                "filename": AGGREGATED_PARTIALS_FILE,
            },
            expected_input={"groupname": InstrumentGroupName},
            output=AggregatedPartialDict,
        )

    def READ_CONSONANCE_ANALYSIS():
        return Action(
            method=step_read_object_from_jsonfile,
            parameters={
                "objecttype": ConsonanceAnalysis,
                "folder": Folder.ANALYSES_DISS,
                "filename": CONSONANCE_ANALYSIS_FILE,
            },
            expected_input={"groupname": InstrumentGroupName},
            output=ConsonanceAnalysis,
        )

    def PROCESS_SOUNDFILES():
        return Action(
            method=step_process_sound_files,
            expected_input={"group": InstrumentGroup},
            output=InstrumentGroup,
        )

    def CREATE_SPECTRA():
        return Action(
            method=step_create_spectra,
            parameters={"spectrum_method": create_spectrum},
            expected_input={"group": InstrumentGroup},
            output=InstrumentGroup,
        )

    def CREATE_SPECTRA_AUDACITY():
        return Action(
            method=step_create_spectra,
            parameters={"spectrum_method": create_spectrum_audacity},
            expected_input={"group": InstrumentGroup},
            output=InstrumentGroup,
        )

    def CREATE_PARTIALS():
        return Action(
            method=step_create_partials,
            expected_input={"group": InstrumentGroup},
            output=InstrumentGroup,
        )

    def AGGREGATE_PARTIALS():
        return Action(
            method=step_create_aggregated_partials_dict,
            parameters={"aggregation": (INSTRUMENTTYPE, OMBAKTYPE)},
            expected_input={"group": InstrumentGroup},
            output=AggregatedPartialDict,
        )

    def AGGREGATE_PARTIALS_ENTIREGROUP():
        return Action(
            method=step_create_aggregated_partials_dict,
            parameters={"aggregation": ()},
            expected_input={"group": InstrumentGroup},
            output=AggregatedPartialDict,
        )

    def PERFORM_CONSONANCE_ANALYSIS():
        return Action(
            method=step_perform_consonance_analysis,
            expected_input={"aggregatedpartials": AggregatedPartialDict},
            output=ConsonanceAnalysis,
        )

    def DETERMINE_BEST_MATCHING_DISSONANCE_CURVES():
        return Action(
            method=step_determine_best_matching_dissonance_curves,
            expected_input={"analysis": ConsonanceAnalysis},
            output=ConsonanceAnalysis,
        )

    def SAVE_INSTRUMENT_GROUP():
        return Action(
            method=step_save_group_to_jsonfile,
            expected_input={"group": InstrumentGroup},
            output=InstrumentGroup,
        )

    def SAVE_INSTRUMENT_GROUP_AND_SPECTRA():
        return Action(
            method=step_save_group_to_jsonfile,
            parameters={"save_spectrumdata": True},
            expected_input={"group": InstrumentGroup},
            output=InstrumentGroup,
        )

    def SAVE_AGGREGATED_PARTIALS():
        return Action(
            method=step_save_object_to_jsonfile,
            parameters={"folder": Folder.ANALYSES_PARTIALS, "filename": AGGREGATED_PARTIALS_FILE},
            expected_input={"object": AggregatedPartialDict, "groupname": InstrumentGroupName},
            output=AggregatedPartialDict,
        )

    def SAVE_CONSONANCE_ANALYSIS():
        return Action(
            method=step_save_object_to_jsonfile,
            parameters={"folder": Folder.ANALYSES_DISS, "filename": CONSONANCE_ANALYSIS_FILE},
            expected_input={"object": ConsonanceAnalysis, "groupname": InstrumentGroupName},
            output=ConsonanceAnalysis,
        )

    def PLOT_SOUNDDATA():
        return Action(
            method=step_plot_sounddata,
            parameters={"filename": "sounddata.pdf"},
            expected_input={"group": InstrumentGroup},
            output=InstrumentGroup,
        )

    def PLOT_NOTE_SPECTRA():
        return Action(
            method=step_plot_note_spectra,
            parameters={
                "max_partials": 10,
                "ratio": (is_ratio := True),
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

    def PLOT_SPECTRA_OF_ANALYSIS():
        return Action(
            method=step_plot_note_spectra,
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

    def PLOT_DISSONANCE_GRAPHS():
        return Action(
            method=step_plot_dissonance_graphs,
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

    def PLOT_DISSONANCE_RESULTS_OF_ANALYSIS():
        return Action(
            method=step_plot_dissonance_graphs,
            parameters={
                "filename": "analysis_graph-{ref_note}{ref_octave}_aggr[{aggregation}].pdf",
            },
            expected_input={
                "aggr_partials": AggregatedPartialDict,
                "groupname": InstrumentGroupName,
                "analysis_results": ConsonanceAnalysis,
            },
            output=AggregatedPartialDict,
        )


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
                     if an action expects input that is not created by any preceding step.()
        """
        self.data[InstrumentGroupName] = groupname
        self.pipe = pipe

        if not self.is_valid_pipe(pipe):
            logger.error(f"Aborting script.")
            exit()

    def is_valid_pipe(self, pipe: list[Step]) -> bool:
        available_data = {item.__class__ for item in self.data.values()}
        action: Action
        for action in pipe:
            if not all([item in available_data for item in list(action.expected_input.values())]):
                logger.error(f"Invalid pipeline sequence: not all input is available for step {action.name}.")
                return False
            if action.output:
                available_data.add(action.output)
        return True

    def __get_data_items__(self, datatypes: dict[str, type]):
        return {keyword: self.data[datatype] for keyword, datatype in datatypes.items()}

    def execute(self):
        logger.info(f"STARTING PIPELINE FOR {self.data[InstrumentGroupName].value.upper()}")
        action: Action
        for action in self.pipe:
            # if isinstance(action, tuple):
            #     action, parameters = action
            #     action.apply_params()(parameters)
            logger.info(f"STEP {action.name()}")
            function = action.method
            inputdict = self.__get_data_items__(action.expected_input)
            result = function(**inputdict, **action.parameters)
            self.data[type(result)] = result


# FULL SEQUENCE STARTS HERE
PIPE_NEW_GROUP_PROCESS_AND_PLOT_AUDIO_NO_CLIPFIX = [
    Step.CREATE_INSTRUMENT_GROUP(),
    Step.SAVE_INSTRUMENT_GROUP(),
    Step.PROCESS_SOUNDFILES().params({"clip_fix": False}),
    Step.SAVE_INSTRUMENT_GROUP(),
    Step.PLOT_SOUNDDATA(),
]
PIPE_CREATE_NOTE_SPECTRA = [
    Step.READ_INSTRUMENT_GROUP().params({"read_sounddata": True}),
    Step.CREATE_SPECTRA_AUDACITY(),
    Step.SAVE_INSTRUMENT_GROUP_AND_SPECTRA(),
]
PIPE_CREATE_PARTIALS_AND_SAVE = [
    Step.READ_GROUP_WITH_SPECTRUMDATA(),
    Step.CREATE_PARTIALS(),
    Step.SAVE_INSTRUMENT_GROUP(),
]
PIPE_AGGREGATE_PARTIALS_AND_SAVE = [
    Step.READ_INSTRUMENT_GROUP(),
    Step.AGGREGATE_PARTIALS(),
    Step.SAVE_AGGREGATED_PARTIALS(),
]
PIPE_PLOT_NOTE_SPECTRA = [
    Step.READ_GROUP_WITH_SPECTRUMDATA(),
    Step.READ_AGGREGATED_PARTIALS(),
    Step.PLOT_NOTE_SPECTRA(),
]
PIPE_PLOT_DISS_GRAPH = [
    Step.READ_AGGREGATED_PARTIALS(),
    Step.PLOT_DISSONANCE_GRAPHS().params({"ref_octave": 2, "ref_note": NoteName.DONG}),
    # Step.PLOT_DISSONANCE_GRAPHS().params({"skip": [GroupName.DISSONANCE_NOTELINES]}),
]
PIPE_ANALYZE_AND_SAVE_ALL_DISSONANCE_CURVES = [
    Step.READ_INSTRUMENT_GROUP(),
    Step.AGGREGATE_PARTIALS_ENTIREGROUP(),
    Step.PERFORM_CONSONANCE_ANALYSIS(),
    Step.SAVE_CONSONANCE_ANALYSIS(),
]
PIPE_OPTIMIZE_AND_PLOT = [
    Step.READ_INSTRUMENT_GROUP(),
    Step.AGGREGATE_PARTIALS_ENTIREGROUP(),
    Step.READ_CONSONANCE_ANALYSIS(),
    Step.DETERMINE_BEST_MATCHING_DISSONANCE_CURVES(),
    Step.SAVE_CONSONANCE_ANALYSIS(),
    Step.PLOT_DISSONANCE_RESULTS_OF_ANALYSIS().params({"nr_partials": 7, "ref_octave": 2, "ref_note": NoteName.DONG}),
]

PIPE_PLOT_SPECTRA_OF_ANALYSIS = [
    Step.READ_GROUP_WITH_SPECTRUMDATA(),
    Step.READ_CONSONANCE_ANALYSIS(),
    Step.PLOT_SPECTRA_OF_ANALYSIS(),
]
PIPE_PLOT_DISS_OF_OPTIMUM = [
    Step.READ_INSTRUMENT_GROUP(),
    Step.AGGREGATE_PARTIALS_ENTIREGROUP(),
    Step.READ_CONSONANCE_ANALYSIS(),
    Step.PLOT_DISSONANCE_RESULTS_OF_ANALYSIS().params({"ref_octave": 2, "ref_note": NoteName.DONG}),
]
# FULL SEQUENCE ENDS HERE

PIPE_PLOT_SOUND_GRAPH = [
    Step.READ_GROUP_WITH_SOUNDDATA(),
    Step.PLOT_SOUNDDATA(),
]

if __name__ == "__main__":
    pipeline = PipeLine(InstrumentGroupName.GENDERWAYANG, PIPE_PLOT_DISS_OF_OPTIMUM)
    pipeline.execute()
OT = [
    Step.READ_INSTRUMENT_GROUP(),
    Step.AGGREGATE_PARTIALS_ENTIREGROUP(),
    Step.READ_CONSONANCE_ANALYSIS(),
    Step.DETERMINE_BEST_MATCHING_DISSONANCE_CURVES(),
    Step.SAVE_CONSONANCE_ANALYSIS(),
    Step.PLOT_DISSONANCE_RESULTS_OF_ANALYSIS(),
]
PIPE_PLOT_DISS_OF_OPTIMUM = [
    Step.READ_INSTRUMENT_GROUP(),
    Step.AGGREGATE_PARTIALS_ENTIREGROUP(),
    Step.READ_CONSONANCE_ANALYSIS(),
    Step.PLOT_DISSONANCE_RESULTS_OF_ANALYSIS(),
]
PIPE_PLOT_SOUND_GRAPH = [
    Step.READ_GROUP_WITH_SOUNDDATA(),
    Step.PLOT_SOUNDDATA(),
]

if __name__ == "__main__":
    pipeline = PipeLine(InstrumentGroupName.ANGKLUNG, PIPE_OPTIMIZE_AND_PLOT)
    pipeline.execute()

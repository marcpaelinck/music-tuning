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
from tuning.visualization.dissonance_plotter import plot_dissonance_graphs
from tuning.visualization.plotter_copy import (
    plot_dissonance_graphs as plot_dissonance_graphs_old,
)
from tuning.visualization.sounddata_plotter import plot_sounddata
from tuning.visualization.spectra_plotter import plot_note_spectra
from tuning.visualization.utils import create_pdf

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
    The values `expected_input` and `output` will be used to perform a validity check before running the pipeline.
    Currently, these values are restricted to Enum and Pydantic BaseModel types, but this is not strictly necessary.
    """

    method: callable
    expected_input: dict[str, Enum | BaseModel]
    output: BaseModel


class Step(Enum):
    """
    All possible steps in a pipeline with associated action.
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
        method=partial(step_read_group_from_jsonfile, read_sounddata=True),
        expected_input={"groupname": InstrumentGroupName},
        output=InstrumentGroup,
    )
    READ_GROUP_WITH_SPECTRUMDATA = Action(
        method=partial(step_read_group_from_jsonfile, read_spectrumdata=True),
        expected_input={"groupname": InstrumentGroupName},
        output=InstrumentGroup,
    )
    READ_AGGREGATED_PARTIALS = Action(
        method=partial(
            read_object_from_jsonfile,
            objecttype=AggregatedPartialDict,
            folder=Folder.ANALYSES,
            filename=AGGREGATED_PARTIALS_FILE,
        ),
        expected_input={"groupname": InstrumentGroupName},
        output=AggregatedPartialDict,
    )
    PROCESS_SOUNDFILES = Action(
        method=step_process_sound_files,
        expected_input={"group": InstrumentGroup},
        output=InstrumentGroup,
    )
    CREATE_SPECTRA = Action(
        method=partial(create_spectra, spectrum_method=create_spectrum),
        expected_input={"group": InstrumentGroup},
        output=InstrumentGroup,
    )
    CREATE_SPECTRA_AUDACITY = Action(
        method=partial(create_spectra, spectrum_method=create_spectrum_audacity),
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
        expected_input={"group": InstrumentGroup},
        output=AggregatedPartialDict,
    )
    AGGREGATE_PARTIALS_ENTIREGROUP = Action(
        method=partial(
            step_create_aggregated_partials_dict,
            aggregation=(),
        ),
        expected_input={"group": InstrumentGroup},
        output=AggregatedPartialDict,
    )
    SAVE_INSTRUMENT_GROUP_WITHOUT_SPECTRA = Action(
        method=step_save_group_to_jsonfile,
        expected_input={"group": InstrumentGroup},
        output=InstrumentGroup,
    )
    SAVE_INSTRUMENT_GROUP_JSON_AND_SPECTRA = Action(
        method=partial(step_save_group_to_jsonfile, save_spectrumdata=True),
        expected_input={"group": InstrumentGroup},
        output=InstrumentGroup,
    )
    SAVE_AGGREGATED_PARTIALS = Action(
        method=partial(
            save_object_to_jsonfile,
            folder=Folder.ANALYSES,
            filename=AGGREGATED_PARTIALS_FILE,
        ),
        expected_input={"object": AggregatedPartialDict, "groupname": InstrumentGroupName},
        output=AggregatedPartialDict,
    )
    PLOT_SOUNDDATA = Action(
        method=partial(
            plot_sounddata,
            filename="sounddata.pdf",
        ),
        expected_input={"group": InstrumentGroup},
        output=InstrumentGroup,
    )
    PLOT_NOTE_SPECTRA = Action(
        method=partial(
            plot_note_spectra,
            max_partials=7,
            ratio=(is_ratio := True),
            what=["spectra", "aggregatedpartials"],
            filename="spectra_with_partials_{type}.pdf".format(type="ratio" if is_ratio else "frequency"),
        ),
        expected_input={
            "group": InstrumentGroup,
            "groupname": InstrumentGroupName,
            "aggregatedpartials": AggregatedPartialDict,
        },
        output=InstrumentGroup,
    )
    PLOT_DISSONANCE_GRAPHS = Action(
        method=partial(
            plot_dissonance_graphs,
            filename="dissonance_graph-{ref_note}{ref_octave}_aggr[{aggregation}].pdf",
            ref_octave=2,
            ref_note=NoteName.DING,
        ),
        expected_input={
            "group": AggregatedPartialDict,
            "groupname": InstrumentGroupName,
        },
        output=AggregatedPartialDict,
    )
    # PLOT_DISSONANCE_GRAPHS_OLD = Action(
    #     method=partial(
    #         create_pdf,
    #         plotter=partial(
    #             plot_dissonance_graphs_old,
    #             ref_octave=2,
    #             ref_note=NoteName.DING,
    #         ),
    #         iterlist=InstrumentType,
    #         folder=Folder.ANALYSES,
    #         filename="dissonance_graph-{ref_note}{ref_octave}{aggregation}.pdf",
    #     ),
    #     expected_input={
    #         "group": AggregatedPartialDict,
    #         "groupname": InstrumentGroupName,
    #     },
    #     output=AggregatedPartialDict,
    # )
    PLOT_DISSONANCE_RESULTS_OF_ANALYSIS = Action(
        method=partial(
            plot_dissonance_graphs,
            iterlist=InstrumentType,
            filename="analysis_graph-{ref_note}{ref_octave}_aggr[{aggregation}].pdf",
        ),
        expected_input={
            "group": AggregatedPartialDict,
            "groupname": InstrumentGroupName,
            "analysis_results": ConsonanceAnalysis,
        },
        output=AggregatedPartialDict,
    )
    PLOT_SPECTRA_OF_ANALYSIS = Action(
        method=partial(
            plot_note_spectra,
            ratio=True,
            what=["spectra", "analysis-partials"],
            filename="analysis_spectra.pdf",
        ),
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
    STEP_DETERMINE_BEST_MATCHING_DISSONANCE_CURVES = Action(
        method=step_determine_best_matching_dissonance_curves,
        expected_input={"analysis": ConsonanceAnalysis},
        output=ConsonanceAnalysis,
    )
    SAVE_CONSONANCE_ANALYSIS = Action(
        method=partial(
            save_object_to_jsonfile,
            folder=Folder.ANALYSES,
            filename=CONSONANCE_ANALYSIS_FILE,
        ),
        expected_input={"object": ConsonanceAnalysis, "groupname": InstrumentGroupName},
        output=ConsonanceAnalysis,
    )
    READ_CONSONANCE_ANALYSIS = Action(
        method=partial(
            read_object_from_jsonfile,
            objecttype=ConsonanceAnalysis,
            folder=Folder.ANALYSES,
            filename=CONSONANCE_ANALYSIS_FILE,
        ),
        expected_input={"groupname": InstrumentGroupName},
        output=ConsonanceAnalysis,
    )

    def __init__(self, value):
        self.action = value  # Better name to improve code readability.


class PipeLine:
    """
    The pipeline enables to execute a sequence of functions that generate output for subsequent
    functions. The sequence is determined in the `pipe` parameter.
    Currently, only data of types InstrumentGroupName, InstrumentGroup or AggregatedPartialDict
    Can be used as input or output by the functions.

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

        Raises:
            Exception: Illegal sequence if the sequence of the Pipe object is infeasible. This is the case
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
        for step in pipe:
            function = step.action.method
            inputdict = self.__get_data_items__(step.action.expected_input)
            result = function(**inputdict)
            self.data[type(result)] = result


if __name__ == "__main__":
    pipe = [
        # Step.CREATE_INSTRUMENT_GROUP,
        # Step.PROCESS_SOUNDFILES,
        # Step.SAVE_JSON_ONLY,
        # Step.READ_GROUP_WITH_SPECTRUMDATA,
        # Step.READ_GROUP_WITH_SOUNDDATA,
        # Step.PLOT_SOUNDDATA,
        # Step.READ_INSTRUMENT_GROUP,
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
        # Step.READ_CONSONANCE_ANALYSIS,
        # Step.PLOT_SPECTRA_OF_ANALYSIS,
        # Step.PLOT_DISSONANCE_GRAPHS,
    ]
    # pipe = [
    #     Step.READ_AGGREGATED_PARTIALS,
    #     Step.PLOT_DISSONANCE_GRAPHS,
    # ]
    # pipe = [
    #     Step.READ_GROUP_WITH_SPECTRUMDATA,
    #     Step.READ_AGGREGATED_PARTIALS,
    #     Step.PLOT_NOTE_SPECTRA,
    # ]
    pipe = [
        Step.READ_INSTRUMENT_GROUP,
        Step.AGGREGATE_PARTIALS_ENTIREGROUP,
        #     # Step.PERFORM_CONSONANCE_ANALYSIS,
        Step.READ_CONSONANCE_ANALYSIS,
        #     Step.STEP_DETERMINE_BEST_MATCHING_DISSONANCE_CURVES,
        #     Step.SAVE_CONSONANCE_ANALYSIS,
        Step.PLOT_DISSONANCE_RESULTS_OF_ANALYSIS,
    ]
    pipeline = PipeLine(InstrumentGroupName.ANGKLUNG, pipe)
    pipeline.execute()

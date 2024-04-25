from typing import Optional

import numpy as np

from tuning.analysis.utils import create_consonance_analysis
from tuning.common.classes import (
    ConsonanceAnalysis,
    ConsonanceValues,
    Frequency,
    InstrumentGroup,
    Partial,
    Ratio,
    Tone,
)
from tuning.common.constants import (
    Folder,
    InstrumentGroupName,
    InstrumentType,
    NoteName,
    PartialCombiType,
)
from tuning.common.utils import step_save_object_to_jsonfile
from tuning.visualization.dissonance_plotter import step_plot_dissonance_graphs
from tuning.visualization.utils import GroupName

# def create_consonance_analysis(
#     part_freq: list[Frequency],
#     harmonics_freq: list[Frequency],
#     noteratio_dict: Optional[dict[NoteName, Ratio]] = {NoteName.DING: 1.0},
# ) -> ConsonanceAnalysis:
#     ref_part = part_freq[0]
#     partials = [
#         Partial(tone=Tone(frequency=freq, amplitude=1), ratio=freq / ref_part, isfundamental=(freq == ref_part))
#         for freq in part_freq
#     ]
#     ref_harm = harmonics_freq[0]
#     harmonics = [
#         Partial(tone=Tone(frequency=freq, amplitude=1), ratio=freq / ref_harm, isfundamental=(freq == ref_part))
#         for freq in harmonics_freq
#     ]
#     return ConsonanceAnalysis(
#         ref_note=NoteName.DING,
#         ref_octave=0,
#         ref_frequency=part_freq[0],
#         fundamental=partials[0],
#         all_partials=partials,
#         all_harmonics=harmonics,
#         noteratio_dict=noteratio_dict,
#         optimal_consonances=(opt := ConsonanceValues(partials=partials, harmonics=harmonics, consonances=dict())),
#         results=[opt],
#     )


if __name__ == "__main__":
    part_ratios = [1.0]
    harm_ratios = [1.0]
    fundamental_freq = 400

    analysis_results = create_consonance_analysis(
        partial_ratios=part_ratios, harmonic_ratios=harm_ratios, fundamental_freq=fundamental_freq
    )
    step_save_object_to_jsonfile(
        object=analysis_results,
        groupname=InstrumentGroupName.TEST,
        folder=Folder.ANALYSES_DISS,
        filename="dissonance_analysis_PL.json",
        logginginfo="analysis Plomp-Levelt dissonance plot",
    )
    step_plot_dissonance_graphs(
        groupname=InstrumentGroupName.TEST,
        analysis_results=analysis_results,
        # filename="dissonance_graph-{ref_note}{ref_octave}_aggr[{aggregation}].pdf",
        filename="dissonance_graph_PL.pdf",
        ref_octave=2,
        ref_note=NoteName.DING,
        end_shift_ratio=1.5,
        include_only=[GroupName.DISSONANCE_GRAPH],
        skip=[GroupName.DISSONANCE_LINE_PP, GroupName.DISSONANCE_LINE_HP],
    )

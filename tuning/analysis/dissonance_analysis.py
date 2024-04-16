import itertools
import time
from collections import defaultdict
from functools import partial
from pprint import pprint

from tuning.analysis.dissonance import (
    dissonance_profile,
    generate_partial_harmonic_pair,
    reduced_freq_dict,
)
from tuning.analysis.utils import get_minima
from tuning.common.classes import (
    AggregatedPartialDict,
    ConsonanceAnalysis,
    ConsonanceValues,
    Dissonance,
    Ratio,
)
from tuning.common.constants import (
    AGGREGATED_PARTIALS_FILE,
    CONSONANCE_ANALYSIS_FILE,
    MAX_NR_HARMONICS_OPTIMIZATION,
    MAX_NR_PARTIALS_OPTIMIZATION,
    Folder,
    InstrumentGroupName,
    NoteName,
    PartialCombiType,
)
from tuning.common.utils import get_logger, read_object_from_jsonfile

logger = get_logger(__name__)


def matching_score_function(dissonance: Dissonance, target_ratio: Ratio) -> float:
    return abs(dissonance.ratio - target_ratio) + abs(dissonance.dissonance / 10e4)


def step_perform_consonance_analysis(
    aggregatedpartials: AggregatedPartialDict,
    ref_note: NoteName = NoteName.DING,
    ref_octave: int = 3,
    max_partials: int = MAX_NR_PARTIALS_OPTIMIZATION,
    max_harmonics: int = MAX_NR_HARMONICS_OPTIMIZATION,
) -> ConsonanceAnalysis:
    start_time = time.time()
    # aggregatedpartials should contain a full aggregation.
    if count := len(list(aggregatedpartials.root.values())) > 1:
        raise Exception(f"The aggregated partials dict should contain exactly one key but {count} keys were detected")

    all_partials, notes = list(aggregatedpartials.root.values())[0]
    # Determine the note frequencies for octave 0
    reduced_freqs = reduced_freq_dict(group_notes=notes, all_notes=notes, ref_note=ref_note, ref_octave=ref_octave)

    # Generate the maximum number of partials and harmonics
    total_partials, total_harmonics = generate_partial_harmonic_pair(
        partials=all_partials,
        reduced_freqs=reduced_freqs,
        nbr_partials=max_partials,
        nbr_harmonics=max_harmonics,
        ref_note=ref_note,
        ref_octave=ref_octave,
    )
    fundamental = next(
        (partial for partial in total_partials if partial.isfundamental),
        None,
    )
    total_partials.remove(fundamental)

    # Generate groups of partials and harmonics of varying lengths, and create all combinations of these groups.
    # Ensure that the fundamental is always included in a partial group.
    partial_groups = [
        (fundamental,) + group
        for length in range(1, max_partials)
        for group in itertools.combinations(total_partials, length)
    ]
    harmonics_groups = [total_harmonics[:length] for length in range(2, max_harmonics + 1)]
    partial_harmonics_combinations = list(itertools.product(partial_groups, harmonics_groups))
    nbr_combis = len(partial_harmonics_combinations)

    # Determine the ratio of each note relative to the reference note.
    noteratio_dict = {notename: freq / reduced_freqs[ref_note] for notename, freq in reduced_freqs.items()}
    analysis = ConsonanceAnalysis(
        ref_note=ref_note,
        ref_octave=ref_octave,
        fundamental=fundamental,
        all_partials=[fundamental] + total_partials,
        all_harmonics=total_harmonics,
        noteratio_dict=noteratio_dict,
        results=list(),
    )

    # Determine the consonance dips for each combination
    for (partials, harmonics), counter in zip(partial_harmonics_combinations, range(1, nbr_combis + 1)):
        if not (counter % 100):
            logger.info(f"Calculating dissonance for combination {counter} of {nbr_combis}")
        pairs = {
            PartialCombiType.PARTIALS_PARTIALS: (partials, partials),
            PartialCombiType.PARTIALS_HARMONICS: (partials, harmonics),
            PartialCombiType.HARMONICS_PARTIALS: (harmonics, partials),
        }
        diss_result = dissonance_profile(pairs, fundamental)
        consonances = defaultdict(list)
        for combitype in PartialCombiType:
            consonances[combitype] = get_minima(*diss_result[combitype])
        analysis.results.append(ConsonanceValues(partials=partials, harmonics=harmonics, consonances=consonances))
    logger.info(f"CONSONANCE ANALYSIS TOOK {time.time() - start_time} SECONDS")
    return analysis


def step_determine_best_matching_dissonance_curves(analysis: ConsonanceAnalysis) -> ConsonanceAnalysis:
    cons_dict = {
        result_id: {
            result.consonances[combi][id]: {"consonancevalues": result, "partialcombi": combi, "diss_id": id}
            for combi, diss_list in result.consonances.items()
            # if combi == PartialCombiType.HARMONICS_PARTIALS
            for id in range(len(diss_list))
        }
        for result, result_id in zip(analysis.results, range(len(analysis.results)))
    }
    best_matches_per_result = {
        result_id: {
            (note, ratio): min(diss_dict.keys(), key=partial(matching_score_function, target_ratio=ratio))
            for note, ratio in analysis.noteratio_dict.items()
        }
        for result_id, diss_dict in cons_dict.items()
    }
    sumsq = {
        result_id: sum(pow(noteratio - diss.ratio, 2) for (note, noteratio), diss in values.items())
        for result_id, values in best_matches_per_result.items()
    }
    best_result_id = min(sumsq.keys(), key=lambda x: sumsq[x])

    best_result = analysis.model_copy(deep=True)
    best_result.optimal_consonances = analysis.results[best_result_id]
    best_result.optimal_matches = {
        note: (
            analysis.noteratio_dict[note],
            cons_dict[best_result_id][dissonance]["partialcombi"].value,
            dissonance,
        )
        for (note, _), dissonance in best_matches_per_result[best_result_id].items()
    }

    return best_result


def summarize_optimal_result(analysis: ConsonanceAnalysis):
    best_result_summary = {
        "partials": [str(p) for p in analysis.optimal_consonances.partials],
        "harmonics": [str(h) for h in analysis.optimal_consonances.harmonics],
        "partial_ids": [analysis.all_partials.index(p) for p in analysis.optimal_consonances.partials],
        "harmonics_ids": [analysis.all_harmonics.index(h) for h in analysis.optimal_consonances.harmonics],
        "results": analysis.optimal_matches,
    }
    print(f"max partials: {len(analysis.all_partials)}")
    print(f"max harmonics: {len(analysis.all_harmonics)}")
    pprint(best_result_summary)


if __name__ == "__main__":
    groupname = InstrumentGroupName.ANGKLUNG

    analysis: ConsonanceAnalysis = read_object_from_jsonfile(
        groupname=groupname,
        objecttype=ConsonanceAnalysis,
        folder=Folder.ANALYSES,
        filename=CONSONANCE_ANALYSIS_FILE,
    )
    analysis = step_determine_best_matching_dissonance_curves(analysis=analysis)
    summarize_optimal_result(analysis)

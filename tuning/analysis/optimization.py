import itertools
import time
from collections import defaultdict

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
)
from tuning.common.constants import (
    MAX_NR_HARMONICS_OPTIMIZATION,
    MAX_NR_PARTIALS_OPTIMIZATION,
    NoteName,
    PartialCombiType,
)
from tuning.common.utils import get_logger

logger = get_logger(__name__)


def step_perform_consonance_analysis(
    aggregatedpartials: AggregatedPartialDict,
    ref_note: NoteName = NoteName.DING,
    ref_octave: int = 3,
    max_partials: int = MAX_NR_PARTIALS_OPTIMIZATION,
    max_harmonics: int = MAX_NR_HARMONICS_OPTIMIZATION,
):
    start_time = time.time()
    # aggregatedpartials should contain a full aggregation.
    if count := len(list(aggregatedpartials.root.values())) > 1:
        raise Exception(f"The aggregated partials dict should contain exactly one key but {count} keys were detected")

    all_partials, notes = list(aggregatedpartials.root.values())[0]
    # Determine the note frequencies for octave 0
    reduced_freqs = reduced_freq_dict(group_notes=notes, all_notes=notes, ref_note=ref_note, ref_octave=ref_octave)
    # Determine the ratio of each note relative to the reference note.
    noteratio_dict = {notename: freq / reduced_freqs[ref_note] for notename, freq in reduced_freqs.items()}

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

    # Determine the consonance dips for each combination
    analysis = ConsonanceAnalysis(ref_note=ref_note, ref_octave=ref_octave, fundamental=fundamental, results=list())
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

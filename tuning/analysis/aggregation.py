import math
import sys
from collections import defaultdict

import numpy as np
import scipy.cluster.hierarchy as hcluster

from tuning.analysis.utils import reduce_frequency
from tuning.common.classes import (
    AggregatedPartial,
    AggregatedPartialDict,
    Instrument,
    InstrumentGroup,
    Note,
    Octave,
    Partial,
    Tone,
)
from tuning.common.constants import (
    AGGREGATION,
    DISTINCTIVENESS_CENT,
    INSTRUMENTTYPE,
    KEEP_NR_PARTIALS,
    OCTAVE,
    OMBAKTYPE,
)
from tuning.common.utils import get_logger

logger = get_logger(__name__)


def dict_key_str(instrument: Instrument, note: Note, aggregation=AGGREGATION) -> str:
    """
    Creates a key for the aggregatedpartials dict.
    Only a combination of instrument type, ombak type, octave and note is allowed.
    See constant AGGREGATION.

    Args:
        instrument (Instrument): _description_
        note (Note): _description_

    Returns:
        str: _description_
    """
    keyitems = (
        []
        + ([instrument.instrumenttype.value] if INSTRUMENTTYPE in aggregation else [])
        + ([instrument.ombaktype.value] if OMBAKTYPE in aggregation else [])
        + ([f"octave {note.octave.index}"] if OCTAVE in aggregation else [])
        + ([f"octave {note.name}"] if Note in aggregation else [])
    )
    if not any(keyitems):
        return "all"
    else:
        return ", ".join(keyitems)


def get_aggr_partials(
    note: Note, aggregatedpartials: AggregatedPartialDict, instrument: Instrument
) -> list[AggregatedPartial]:
    return aggregatedpartials.root.get(dict_key_str(instrument, note, aggregatedpartials.aggregation), [[]])[0]


def dict_key_retrieve_instrgroup(key):
    return "".join(key.split(",")[1:])


def ratio_to_cents(ratio) -> int:
    return int(math.log(ratio, 2) * 1200)


def average_ratio(partials: list[Partial]) -> float:
    return np.average(p.ratio for p in partials)


def aggregated_partials_list(
    partials: list[Partial], dist_cent: int = DISTINCTIVENESS_CENT, keep_nr: int = KEEP_NR_PARTIALS
) -> list[AggregatedPartial]:
    """
    Creates clusters from a list of partials and returns a list of AggregatedPartial objects.
    The clusters are created by grouping partials with similar frequencies. The constant
    DISTINCTIVENESS_CENT determines the maximum width of each cluster.

    Args:
        partials (list[Partial]): The partials that should be clustered
        dist_cent (int): Threshold for clustering, in cents
        keep_nr (int): Number of partials to keep

    Returns:
        list[AggregatedPartial]: Partials, aggregated by cluster
    """
    cents_collection = [ratio_to_cents(partial.ratio) for partial in partials if partial.ratio > 0]
    threshold = dist_cent
    clusters = hcluster.fclusterdata(
        np.array(cents_collection).reshape(-1, 1),
        threshold,
        criterion="distance",
        method="centroid",
    )
    clustering = [[partials[i] for i in range(len(partials)) if clusters[i] == cl] for cl in set(clusters)]
    largest_clusters = sorted(clustering, key=lambda c: sys.maxsize * c[0].isfundamental + len(c), reverse=True)[
        :keep_nr
    ]
    # Note: see the explanation of tone and fundamental in the definition of the AggregatedPartial class.
    aggr_part_list = [
        AggregatedPartial(
            tone=Tone(
                frequency=(
                    tone_freq := np.average([reduce_frequency(p.tone.frequency, p.fund_octave.index) for p in cluster])
                ),
                amplitude=np.average([p.tone.amplitude for p in cluster]),
            ),
            fundamental_tone=Tone(
                frequency=(
                    fund_freq := np.average(
                        [reduce_frequency(p.fundamental.frequency, p.fund_octave.index) for p in cluster]
                    )
                ),
                amplitude=np.average([p.fundamental.amplitude for p in cluster]),
            ),
            isfundamental=cluster[0].isfundamental,
            ratio=np.average([p.ratio for p in cluster]),
            # ratio0=tone_freq / fund_freq,
            prominence=np.average([p.prominence for p in cluster]),
            partials=cluster,
            octave_idx=0,
        )
        for cluster in largest_clusters
    ]
    return sorted(aggr_part_list, key=lambda p: p.tone.amplitude, reverse=True)


def step_create_aggregated_partials_dict(
    group: InstrumentGroup, aggregation: tuple[str] = AGGREGATION
) -> AggregatedPartialDict:
    """
    Creates a dictionary with lists of aggregated partials. Aggregation is performed
    for each instrumenttype - ombaktype - octave combination.

    Args:
        group (InstrumentGroup): Group of instruments to analyze

    Raises:
        Exception: If a fundamental is missing in one of the aggregations.

    Returns:
        AggregatedPartialDict:
    """
    logger.info(
        f"Aggregating partials {'by ' + ', '.join(aggregation) + ']' if aggregation else 'over all instruments'}"
    )
    partials_dict = defaultdict(list)

    # Create a dict of partials per instrument/note combination
    note_dict = defaultdict(lambda: defaultdict(list))
    all_notes = list()
    for instrument in group.instruments:
        for note in instrument.notes:
            key = dict_key_str(instrument, note, aggregation)
            partials_dict[key].extend([partial for p_list in note.partials for partial in p_list])
            note_dict[key][note.name].append(note)
            all_notes.append(note)

    # Aggregate the note_dict by taking the average reduced frequency per note
    octave_0 = Octave(
        index=0,
        start_frequency=min([note.octave.start_frequency / pow(2, note.octave.index) for note in all_notes]),
        end_frequency=max([note.octave.end_frequency / pow(2, note.octave.index) for note in all_notes]),
    )
    notes_octave_0_dict = dict()
    for key, note_list in note_dict.items():
        notenames = sorted(list(note_list.keys()), key=lambda n: n.index())
        notes_octave_0_dict[key] = {
            notename: Note(
                name=notename,
                octave=octave_0,
                frequency=np.mean([note.frequency for note in note_list[notename]]),
            )
            for notename in notenames
        }

    # Create an aggregated 'octave 0' note list for the entire instrument group
    global_notenames = sorted(list(set([note.name for note in all_notes])), key=lambda n: n.index())
    global_notes_octave_0_dict = {
        notename: Note(
            name=notename,
            octave=octave_0,
            frequency=np.mean([note.frequency for note in all_notes if note.name == notename]),
        )
        for notename in global_notenames
    }

    partials_dict = dict(sorted(partials_dict.items()))

    aggregated_dict = defaultdict(list)
    for key, partials in partials_dict.items():
        aggregated_dict[key] = (
            aggregated_partials_list(partials),
            notes_octave_0_dict[key],
        )

    aggregated = AggregatedPartialDict(
        aggregation=aggregation, root=aggregated_dict, notes_octave_0=global_notes_octave_0_dict
    )
    # Recalculate the ratio, based on average frequencies
    for key, (agglist, _) in aggregated.root.items():
        fundamental = next((agg_partial for agg_partial in agglist if agg_partial.isfundamental), None)
        if not fundamental:
            raise Exception(f"Fundamental is missing for {key}, aggr partials are {[agg.ratio for agg in agglist]}")
        for agg_partial in agglist:
            # aggregate.ratio = aggregate.tone.frequency / fundamental.tone.frequency
            ratio_list = [p.ratio for p in agg_partial.partials]
            agg_partial.ratio = sum(ratio_list) / len(ratio_list)
            # logger.info(f"{key}: ratio {oldratio} corrected to {aggregate.ratio}")

    return aggregated


if __name__ == "__main__":
    # Set this value before running
    ...

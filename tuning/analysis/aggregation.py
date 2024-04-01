import math
import sys
from collections import defaultdict

import numpy as np
import scipy.cluster.hierarchy as hcluster

from tuning.common.classes import (
    AggregatedPartial,
    AggregatedPartialDict,
    InstrumentGroup,
    Partial,
    Tone,
)
from tuning.common.constants import (
    AGGREGATED_PARTIALS_FILE,
    DISTINCTIVENESS_CENT,
    KEEP_NR_PARTIALS,
    Folder,
    InstrumentGroupName,
)
from tuning.common.utils import (
    get_logger,
    get_path,
    save_object_to_jsonfile,
    step_read_group_from_jsonfile,
)

logger = get_logger(__name__)


def ratio_to_cents(ratio) -> int:
    return int(math.log(ratio, 2) * 1200)


def partial_avg(partials: list[Partial], dimension: str) -> float:
    return np.average(p.tone.frequency for p in partials)


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
    # Note: avg_ratio should be recalculated later, based on average fundamental freq.
    return [
        AggregatedPartial(
            tone=Tone(
                frequency=np.average([p.tone.frequency for p in cluster]),
                amplitude=np.average([p.tone.amplitude for p in cluster]),
            ),
            isfundamental=cluster[0].isfundamental,
            ratio=np.average([p.ratio for p in cluster]),
            prominence=np.average([p.prominence for p in cluster]),
            partials=cluster,
        )
        for cluster in largest_clusters
    ]


def step_create_aggregated_partials_dict(group: InstrumentGroup) -> AggregatedPartialDict:
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
    partials_dict = defaultdict(list)
    for instrument in group.instruments:
        for note in instrument.notes:
            partials_dict[(instrument.instrumenttype, instrument.ombaktype, note.octave)].extend(
                [partial for p_list in note.partials for partial in p_list]
            )

    aggregated_dict = defaultdict(list)
    for (instrumenttype, ombaktype, octave), partials in partials_dict.items():
        aggregated_dict[f"{instrumenttype.value}#{ombaktype.value}-octave {octave.index}"] = aggregated_partials_list(
            partials
        )

    aggregated = AggregatedPartialDict(root=aggregated_dict)
    # Recalculate the ratio, based on average frequencies
    for key, agglist in aggregated.root.items():
        fundamental = next((agg for agg in agglist if agg.isfundamental), None)
        if not fundamental:
            raise Exception(
                f"Fundamental is missing for {key}, aggr partials are {[(agg.tone.frequency, (agg.octave.start_frequency, agg.octave.end_frequency)) for agg in agglist]}"
            )
        for aggregate in agglist:
            oldratio = aggregate.ratio
            aggregate.ratio = aggregate.tone.frequency / fundamental.tone.frequency
            logger.info(f"{key}: ratio {oldratio} corrected to {aggregate.ratio}")

    return aggregated


if __name__ == "__main__":
    # Set this value before running
    GROUPNAME = InstrumentGroupName.SEMAR_PAGULINGAN

    orchestra = step_read_group_from_jsonfile(GROUPNAME, read_sounddata=False, read_spectrumdata=False)
    aggr_partials = step_create_aggregated_partials_dict(orchestra)
    save_object_to_jsonfile(
        aggr_partials,
        get_path(orchestra.grouptype, Folder.ANALYSES, AGGREGATED_PARTIALS_FILE),
    )

import json
import math
from pprint import pprint

import numpy as np
import scipy.cluster.hierarchy as hcluster

from tuning.common.classes import (
    AggregatedPartial,
    AggregatedPartialDict,
    InstrumentGroup,
    Partial,
    Tone,
)
from tuning.common.constants import KEEP_NR_PARTIALS, FileType, InstrumentGroupName
from tuning.common.utils import (
    get_path,
    read_group_from_jsonfile,
    save_object_to_jsonfile,
)


def ratio_to_cents(ratio) -> int:
    return int(math.log(ratio, 2) * 1200)


def partial_avg(partials: list[Partial], dimension: str) -> float:
    return np.average(p.tone.frequency for p in partials)


def average_ratio(partials: list[Partial]) -> float:
    return np.average(p.ratio for p in partials)


def aggregate(partials: list[Partial], identifier: str) -> AggregatedPartial:
    avg_freq = np.average([p.tone.frequency for p in partials])
    avg_ampl = np.average([p.tone.amplitude for p in partials])
    avg_ratio = np.average([p.ratio for p in partials])
    isfundamental = partials[0].isfundamental
    return AggregatedPartial(
        identifier=identifier,
        tone=Tone(frequency=avg_freq, amplitude=avg_ampl),
        isfundamental=isfundamental,
        partials=partials,
        ratio=avg_ratio,
    )


def summarize_partials(
    group: InstrumentGroup, grouping: str = "instrumenttype"
) -> dict[str, dict[int, list[Partial]]]:
    # collect the partials and convert the ratios to cents
    instrument_types = {instrument.instrumenttype for instrument in group.instruments}
    cents_collections = {
        itype: [
            ratio_to_cents(partial.ratio)
            for instrument in group.instruments
            if instrument.instrumenttype == itype
            for note in instrument.notes
            for partial in note.partials
            # TODO check why partial can have negative frequency (should be solved)
            if partial.ratio > 0
        ]
        for itype in instrument_types
    }
    partial_collections = {
        itype: [
            partial
            for instrument in group.instruments
            if instrument.instrumenttype == itype
            for note in instrument.notes
            for partial in note.partials
            # TODO check why partial can have negative frequency (should be solved)
        ]
        for itype in instrument_types
    }
    thresh = 50  # 3/4 of a semitone
    clusters = {
        itype: hcluster.fclusterdata(
            np.array(partials).reshape(-1, 1),
            thresh,
            criterion="distance",
            method="centroid",
        )
        for itype, partials in cents_collections.items()
    }
    clustering = {
        itype: [
            [partials[i] for i in range(len(partials)) if clusters[itype][i] == cl]
            for cl in set(clusters[itype])
        ]
        for itype, partials in partial_collections.items()
    }
    largest_clusters = {
        itype: sorted(clusters, key=lambda c: len(c), reverse=True)[:KEEP_NR_PARTIALS]
        for itype, clusters in clustering.items()
    }
    aggregated = AggregatedPartialDict(
        root={
            itype: [aggregate(cluster, itype) for cluster in clusters]
            for itype, clusters in largest_clusters.items()
        }
    )

    save_object_to_jsonfile(
        aggregated, get_path(group.grouptype, FileType.ANALYSES, "partials_per_instrumenttype.json")
    )


if __name__ == "__main__":
    orchestra = read_group_from_jsonfile(
        InstrumentGroupName.SEMAR_PAGULINGAN, read_sounddata=False, read_spectrumdata=False
    )
    summary = summarize_partials(orchestra)

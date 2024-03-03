import json
import math
from pprint import pprint

import numpy as np
import scipy.cluster.hierarchy as hcluster

from tuning.common.classes import (
    AggregatedPartial,
    AggregatedPartialDict,
    InstrumentGroup,
    InstrumentType,
    Octave,
    Partial,
    Tone,
)
from tuning.common.constants import (
    AGGREGATED_PARTIALS_FILE,
    KEEP_NR_PARTIALS,
    Folder,
    InstrumentGroupName,
)
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


def aggregated_partials(
    partials: list[Partial],
    instrumenttype: InstrumentType,
    octave: Octave,
) -> AggregatedPartial:
    avg_freq = np.average([p.tone.frequency for p in partials])
    avg_ampl = np.average([p.tone.amplitude for p in partials])
    avg_ratio = np.average([p.ratio for p in partials])
    isfundamental = partials[0].isfundamental
    return AggregatedPartial(
        instrumenttype=instrumenttype,
        octave=octave,
        tone=Tone(frequency=avg_freq, amplitude=avg_ampl),
        isfundamental=isfundamental,
        partials=partials,
        ratio=avg_ratio,
    )


def summarize_partials(
    group: InstrumentGroup, keyvalues: str = "instrumenttype"
) -> dict[str, dict[int, list[Partial]]]:
    # collect the partials and convert the ratios to cents
    keyvalues = {
        (instrument.instrumenttype, octave_idx)
        for instrument in group.instruments
        for octave_idx in {note.octave for note in instrument.notes}
    }
    cents_collections = {
        (itype, octave): [
            ratio_to_cents(partial.ratio)
            for instrument in group.instruments
            if instrument.instrumenttype == itype
            for note in instrument.notes
            if note.octave == octave
            for partial in note.partials
            # TODO check why partial can have negative frequency (should be solved)
            if partial.ratio > 0
        ]
        for (itype, octave) in keyvalues
    }
    partial_collections = {
        (itype, octave): [
            partial
            for instrument in group.instruments
            if instrument.instrumenttype == itype
            for note in instrument.notes
            if note.octave == octave
            for partial in note.partials
            # TODO check why partial can have negative frequency (should be solved)
        ]
        for (itype, octave) in keyvalues
    }
    thresh = 50  # 3/4 of a semitone
    clusters = {
        key: hcluster.fclusterdata(
            np.array(partials).reshape(-1, 1),
            thresh,
            criterion="distance",
            method="centroid",
        )
        for key, partials in cents_collections.items()
    }
    clustering = {
        key: [
            [partials[i] for i in range(len(partials)) if clusters[key][i] == cl]
            for cl in set(clusters[key])
        ]
        for key, partials in partial_collections.items()
    }
    largest_clusters = {
        group: sorted(clusters, key=lambda c: len(c), reverse=True)[:KEEP_NR_PARTIALS]
        for group, clusters in clustering.items()
    }
    aggregated = AggregatedPartialDict(
        root={
            f"{itype.value}-{octave.index}": [
                aggregated_partials(cluster, itype, octave) for cluster in clusters
            ]
            for (itype, octave), clusters in largest_clusters.items()
        }
    )

    save_object_to_jsonfile(
        aggregated, get_path(group.grouptype, Folder.ANALYSES, AGGREGATED_PARTIALS_FILE)
    )


if __name__ == "__main__":
    orchestra = read_group_from_jsonfile(
        InstrumentGroupName.SEMAR_PAGULINGAN, read_sounddata=False, read_spectrumdata=False
    )
    summary = summarize_partials(orchestra)

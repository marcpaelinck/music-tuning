import math

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
    DISTINCTIVENESS_CENT,
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
    # Note: avg_ratio will be recalculated later, based on average fundamental freq.
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


def summarize_partials(group: InstrumentGroup) -> dict[str, dict[int, list[Partial]]]:
    # Grouping by instrument type, ombaktype and octave
    keyvalues = {
        (instrument.instrumenttype, instrument.ombaktype, octave_idx)
        for instrument in group.instruments
        for octave_idx in {note.octave for note in instrument.notes}
    }
    cents_collections = {
        (itype, ombak, octave): [
            ratio_to_cents(partial.ratio)
            for instrument in group.instruments
            if instrument.instrumenttype == itype and instrument.ombaktype == ombak
            for note in instrument.notes
            if note.octave == octave
            for partial in note.partials
            if partial.ratio > 0
        ]
        for (itype, ombak, octave) in keyvalues
    }
    partial_collections = {
        (itype, ombak, octave): [
            partial
            for instrument in group.instruments
            if instrument.instrumenttype == itype and instrument.ombaktype == ombak
            for note in instrument.notes
            if note.octave == octave
            for partial in note.partials
        ]
        for (itype, ombak, octave) in keyvalues
    }
    thresh = DISTINCTIVENESS_CENT  # 1/2 of a semitone
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
            f"{itype.value}#{ombak.value}-octave {octave.index}": [
                aggregated_partials(cluster, itype, octave) for cluster in clusters
            ]
            for (itype, ombak, octave), clusters in largest_clusters.items()
        }
    )
    # Recalculate the ratio, based on average frequencies
    for key, agglist in aggregated.root.items():
        fundamental = next(agg for agg in agglist if agg.isfundamental)
        for aggregate in agglist:
            oldratio = aggregate.ratio
            aggregate.ratio = aggregate.tone.frequency / fundamental.tone.frequency
            print(f"{key}: ratio {oldratio} corrected to {aggregate.ratio}")

    save_object_to_jsonfile(
        aggregated, get_path(group.grouptype, Folder.ANALYSES, AGGREGATED_PARTIALS_FILE)
    )


if __name__ == "__main__":
    # Set this value before running
    GROUPNAME = InstrumentGroupName.SEMAR_PAGULINGAN

    orchestra = read_group_from_jsonfile(GROUPNAME, read_sounddata=False, read_spectrumdata=False)
    summary = summarize_partials(orchestra)

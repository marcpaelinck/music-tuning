import math
from pprint import pprint

import numpy as np
import scipy.cluster.hierarchy as hcluster

from tuning.common.classes import InstrumentGroup
from tuning.common.constants import InstrumentGroupName
from tuning.common.utils import read_group_from_jsonfile


def ratio_to_cents(ratio) -> int:
    return int(math.log(ratio, 2) * 1200)


def summarize_partials(group: InstrumentGroup, grouping: str = "instrumenttype"):
    # collect the partials and convert the ratios to cents
    instrument_types = {instrument.instrumenttype for instrument in group.instruments}
    cents_collections = {
        itype: [
            ratio_to_cents(partial.ratio)
            for instrument in group.instruments
            if instrument.instrumenttype == itype
            for note in instrument.notes
            for partial in note.partials
            # TODO check why partial can have negative frequency
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
            # TODO check why partial can have negative frequency
            if partial.ratio > 0
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
        itype: {
            cl: [partials[i] for i in range(len(partials)) if clusters[itype][i] == cl]
            for cl in set(clusters[itype])
        }
        for itype, partials in partial_collections.items()
    }
    return clustering


if __name__ == "__main__":
    orchestra = read_group_from_jsonfile(
        InstrumentGroupName.SEMAR_PAGULINGAN, read_sounddata=False, read_spectrumdata=False
    )
    summary = summarize_partials(orchestra)
    small = {
        itype: {cl: cluster for cl, cluster in clusters.items() if len(cluster) > 15}
        for itype, clusters in summary.items()
    }
    pprint({itype: len(clusters.keys()) for itype, clusters in small.items()})
    pprint(
        {
            itype: [
                {round(partial.ratio, 3) for partial in cluster} for cluster in clusters.values()
            ]
            for itype, clusters in small.items()
        }
    )

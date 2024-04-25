import numpy as np
import pytest

from tuning.analysis.utils import get_minima
from tuning.common.classes import (
    ConsonanceAnalysis,
    ConsonanceValues,
    Frequency,
    Partial,
    Ratio,
    Tone,
)
from tuning.common.constants import NoteName, PartialCombiType
from tuning.presentation.create_consonance_analysis import create_consonance_analysis
from tuning.visualization.dissonance_plotter import generate_dissonance_data_optimized

data = [
    # This data corresponds with the dips in a consonance graph taken from William Sethares' book Tuning, Timbre, Spectrum, Scale, fig. 10.10 page 208.
    # I use this graph to validate the consonance model (MP). The amplitude is assumed to be the same for all frequencies.
    (
        800,  # Reference frequency (freq of the fundamental)
        [1.0, 1.52, 3.46, 3.9],  # Frequencies of the partials
        [1.0, 2.0, 3.0, 4.0],  # Frequencies of the harmonics
        {  # Expected minima of the dissonance curve.
            # This list also contains the values of the shallow dips between the minima indicated in the figure of the book.
            PartialCombiType.PARTIALS_PARTIALS: [
                (1.0, 0.04),
                (1.127, 0.157),
                (1.293, 0.037),
                (1.52, 0.001),
                (1.894, 0.006),
                (2.276, 0.021),
                (2.418, 0.215),
                (2.5, 0.225),
            ],
            PartialCombiType.PARTIALS_HARMONICS: [
                (1.0, 0.201),
                (1.086, 0.269),
                (1.153, 0.073),
                (1.228, 0.235),
                (1.3, 0.048),
                (1.405, 0.181),
                (1.52, 0.023),
                (1.73, 0.058),
                (1.844, 0.223),
                (1.95, 0.022),
                (2.5, 0.0),
            ],
            PartialCombiType.HARMONICS_PARTIALS: [
                (1.0, 0.201),
                (1.026, 0.192),
                (1.156, 0.095),
                (1.25, 0.226),
                (1.316, 0.017),
                (1.599, 0.008),
                (1.974, 0.14),
                (2.0, 0.15),
                (2.328, 0.037),
            ],
        },
    ),
]


@pytest.mark.parametrize("fundamental_freq, partial_ratios, harmonic_ratios, expected", data)
def test_generate_dissonance_data(
    fundamental_freq: float,
    partial_ratios: list[Partial],
    harmonic_ratios: list[Partial],
    expected: dict[PartialCombiType, list[tuple[Ratio, float]]],
):
    """Validates the method that generates a dissonance curve. The amplitude is assumed to be 1 for all frequencies.
    Args:
        fundamental_freq (float): _description_
        partial_ratios (list[Partial]): list of partial ratios
        harmonic_ratios (list[Partial]): list of harmonic ratios
        expected (dict[PartialCombiType, list[tuple[Ratio, float]]]): (ratio, dissonance) pairs corresponding with the minima of the curve.
    """
    analysis_results = create_consonance_analysis(
        partial_ratios=partial_ratios, harmonic_ratios=harmonic_ratios, fundamental_freq=fundamental_freq
    )
    (_, _, _, _, _, diss_results) = generate_dissonance_data_optimized(analysis_results)
    diss_results = list(diss_results.values())[0]
    consonances = {
        combi: sorted([(round(c.ratio, 3), round(c.dissonance, 3)) for c in get_minima(*diss_results[combi])])
        for combi in PartialCombiType
    }

    assert consonances == expected

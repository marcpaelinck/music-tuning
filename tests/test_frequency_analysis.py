import pytest

from tuning.analysis.partials import reduce_to_octave

data = [(4, 1), (4.5, 1.125), (1.81, 1.81), (3.875, 1.9375)]


@pytest.mark.parametrize("freq, expected", data)
def test_reduce_to_octave(freq, expected):
    assert round(reduce_to_octave(freq), 5) == expected

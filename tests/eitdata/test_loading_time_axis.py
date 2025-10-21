import numpy as np
import pytest


# TODO: add other vendors
@pytest.mark.parametrize("sequence_fixture_name", ["draeger1", "draeger_porcine_2"])
def test_time_axis(sequence_fixture_name: str, request: pytest.FixtureRequest):
    sequence = request.getfixturevalue(sequence_fixture_name)
    time_diff = np.diff(sequence.time)
    assert np.allclose(time_diff, time_diff.mean())

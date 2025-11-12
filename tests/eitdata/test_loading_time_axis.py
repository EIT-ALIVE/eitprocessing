import numpy as np
import pytest


# TODO: add other vendors
@pytest.mark.parametrize(
    "sequence",
    ["draeger_20hz_healthy_volunteer_fixed_rr", "draeger_50hz_healthy_volunteer_pressure_pod"],
    indirect=True,
)
def test_time_axis(sequence: str):
    time_diff = np.diff(sequence.time)
    assert np.allclose(time_diff, time_diff.mean())

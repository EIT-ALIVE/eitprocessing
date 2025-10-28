from copy import deepcopy

import pytest  # TODO: noqa: F401 (needed for fixtures) once the pytest.skip is removed

from eitprocessing.datahandling.sequence import Sequence


# TODO: add other vendors' dataset
@pytest.mark.parametrize(
    "sequence",
    ["draeger_20hz_healthy_volunteer", "draeger_20hz_healthy_volunteer_pressure_pod"],
    indirect=True,
)
def test_copy(sequence: Sequence):
    sequence_copy = deepcopy(sequence)
    assert sequence == sequence_copy


# TODO: add tests for specific items in sequences

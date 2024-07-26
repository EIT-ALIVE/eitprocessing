from copy import deepcopy

import pytest  # TODO: noqa: F401 (needed for fixtures) once the pytest.skip is removed

from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.sequence import Sequence
from tests.conftest import draeger_file1


def test_eq():
    data = load_eit_data(draeger_file1, vendor="draeger")
    data2 = load_eit_data(draeger_file1, vendor="draeger")

    data.isequivalent(data2)


def test_copy(
    draeger1: Sequence,
    timpel1: Sequence,
):
    data: Sequence
    for data in [draeger1, timpel1]:
        data_copy = deepcopy(data)
        assert data == data_copy


def test_equals(
    draeger1: Sequence,
    timpel1: Sequence,
):
    pytest.skip("Will add tests to check that correct attributes do and don't lead to failed equality.")
    # Here we should add tests ensuring that changes that shouldn't lead to failed equality indeed don't
    # and vice versa.
    # The current tests are outdated versions of this
    data: Sequence
    for data in [draeger1, timpel1]:
        data_copy = Sequence()
        data_copy.path = deepcopy(data.path)
        data_copy.time = deepcopy(data.time)
        data_copy.nframes = deepcopy(data.nframes)
        data_copy.sample_frequency = deepcopy(data.sample_frequency)
        data_copy.framesets = deepcopy(data.framesets)
        data_copy.timing_errors = deepcopy(data.timing_errors)
        data_copy.vendor = deepcopy(data.vendor)

        assert data_copy == data

        # test wheter a difference in framesets fails equality test
        data_copy.framesets["test"] = data_copy.framesets["raw"].deepcopy()
        assert data != data_copy
        data_copy.framesets = deepcopy(data.framesets)

        data_copy.framesets["raw"].name += "_"
        assert data != data_copy
        data_copy.framesets = deepcopy(data.framesets)

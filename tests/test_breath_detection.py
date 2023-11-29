import os
import numpy as np
import pytest
from numpy.typing import NDArray
from eitprocessing.sequence import Sequence


# from eitprocessing.features.breath_detection import Breath
# from eitprocessing.features.breath_detection import BreathDetection


environment = os.environ.get(
    "EIT_PROCESSING_TEST_DATA",
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
)
data_directory = os.path.join(environment, "tests", "test_data")
draeger_file1 = os.path.join(data_directory, "Draeger_Test3.bin")
draeger_file2 = os.path.join(data_directory, "Draeger_Test.bin")
timpel_file = os.path.join(data_directory, "Timpel_Test.txt")


@pytest.fixture(scope="module")
def draeger_data1():
    return Sequence.from_path(draeger_file1, vendor="draeger")


@pytest.fixture(scope="module")
def draeger_data2():
    return Sequence.from_path(draeger_file2, vendor="draeger")


@pytest.fixture(scope="module")
def timpel_data():
    return Sequence.from_path(timpel_file, vendor="timpel")


def test_find_features():
    ...


def test_remove_edge_cases():
    ...


@pytest.mark.parametrize(
    "peak_indices,peak_values,valley_indices,valley_values,expected_peak_indices,expected_valley_indices",
    [
        (  # test removal of the first valley
            np.array([0, 3]),
            np.array([0, 0]),
            np.array([1, 2]),
            np.array([1, 2]),
            np.array([0, 3]),
            np.array([1]),
        ),
        (  # test removal of the second valley
            np.array([0, 3]),
            np.array([0, 0]),
            np.array([1, 2]),
            np.array([2, 1]),
            np.array([0, 3]),
            np.array([2]),
        ),
        (  # test removal of valleys with same value
            np.array([0, 3]),
            np.array([0, 0]),
            np.array([1, 2]),
            np.array([1, 1]),
            np.array([0, 3]),
            np.array([1]),
        ),
        (  # test removal of first peak
            np.array([1, 2]),
            np.array([1, 2]),
            np.array([0, 3]),
            np.array([0, 0]),
            np.array([2]),
            np.array([0, 3]),
        ),
        (  # test removal of second peak
            np.array([1, 2]),
            np.array([2, 1]),
            np.array([0, 3]),
            np.array([0, 0]),
            np.array([1]),
            np.array([0, 3]),
        ),
        (  # test removal of peaks with same value
            np.array([1, 2]),
            np.array([1, 1]),
            np.array([0, 3]),
            np.array([0, 0]),
            np.array([1]),
            np.array([0, 3]),
        ),
        (  # test removal of both valleys and peaks
            np.array([1, 2, 5, 6]),
            np.array([1, 1, 1, 2]),
            np.array([0, 3, 4, 7]),
            np.array([0, 0, 1, 0]),
            np.array([1, 6]),
            np.array([0, 3, 7]),
        ),
    ],
)
def test_remove_doubles(
    peak_indices: NDArray,
    peak_values: NDArray,
    valley_indices: NDArray,
    valley_values: NDArray,
    expected_peak_indices: NDArray,
    expected_valley_indices,
):
    bd = BreathDetection(sample_frequency=1)
    result_peak_indices, _, result_valley_indices, _ = bd._remove_doubles(
        peak_indices, peak_values, valley_indices, valley_values
    )

    assert np.all(result_peak_indices == expected_peak_indices)
    assert np.all(result_valley_indices == expected_valley_indices)


def test_remove_low_amplitudes():
    ...


def test_remove_breaths_around_invalid_data():
    ...


def test_with_data(draeger_data1, draeger_data2, timpel_data):
    for sequence in draeger_data1, draeger_data2, timpel_data:
        bd = BreathDetection(
            sample_frequency=sequence.framerate,
        )
        gi = sequence.framesets["raw"].global_impedance
        breaths = bd.find_breaths(gi)

        for breath in breaths:
            # Test whether the indices are in the proper order within a breath
            assert breath.start_index < breath.middle_index < breath.end_index

            # Test whether the peak values are larger than valley values
            assert gi[breath.middle_index] > gi[breath.start_index]
            assert gi[breath.middle_index] > gi[breath.end_index]

        start_indices, middle_indices, end_indices = (list(x) for x in zip(*breaths))

        # Test whether breaths are sorted properly
        assert start_indices == sorted(start_indices)
        assert middle_indices == sorted(middle_indices)
        assert end_indices == sorted(end_indices)

        # Test whether indices are unique. `set` removes non-unique values,
        # `sorted(list(...))` converts the set to a sorted list again.
        assert list(start_indices) == sorted(list(set(start_indices)))
        assert list(middle_indices) == sorted(list(set(middle_indices)))
        assert list(end_indices) == sorted(list(set(end_indices)))

        # Test whether the start of the next breath is on/after the previous breath
        assert all(
            [
                start_index >= end_index
                for start_index, end_index in zip(start_indices[1:], end_indices[:-1])
            ]
        )

import os
from pathlib import Path

import numpy as np
import pytest

from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.features.breath_detection import BreathDetection

environment = Path(
    os.environ.get(
        "EIT_PROCESSING_TEST_DATA",
        Path(__file__).parent.parent.resolve(),
    ),
)
data_directory = environment / "tests" / "test_data"
draeger_file1 = data_directory / "Draeger_Test3.bin"
draeger_file2 = data_directory / "Draeger_Test.bin"
timpel_file = data_directory / "Timpel_Test.txt"


def test_find_features(): ...


def test_remove_edge_cases(): ...


@pytest.mark.parametrize(
    (
        "peak_indices",
        "peak_values",
        "valley_indices",
        "valley_values",
        "expected_peak_indices",
        "expected_valley_indices",
    ),
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
    peak_indices: np.ndarray,
    peak_values: np.ndarray,
    valley_indices: np.ndarray,
    valley_values: np.ndarray,
    expected_peak_indices: np.ndarray,
    expected_valley_indices: np.ndarray,
):
    bd = BreathDetection(sample_frequency=1)
    result_peak_indices, _, result_valley_indices, _ = bd._remove_doubles(  # noqa: SLF001
        peak_indices,
        peak_values,
        valley_indices,
        valley_values,
    )

    assert np.all(result_peak_indices == expected_peak_indices)
    assert np.all(result_valley_indices == expected_valley_indices)


def test_remove_low_amplitudes(): ...


def test_remove_breaths_around_invalid_data(): ...


def test_with_data(draeger1: Sequence, draeger2: Sequence, timpel1: Sequence):
    for sequence in draeger1, draeger2, timpel1:
        bd = BreathDetection(
            sample_frequency=sequence.eit_data["raw"].framerate,
        )

        cd = sequence.continuous_data["global_impedance_(raw)"]
        breaths = bd.find_breaths(sequence, "global_impedance_(raw)")

        for breath in breaths.values:
            # Test whether the indices are in the proper order within a breath
            assert breath.start_time < breath.middle_time < breath.end_time

            # Test whether the peak values are larger than valley values
            assert cd.t[breath.middle_time].values[0] > cd.t[breath.start_time].values[0]
            assert cd.t[breath.middle_time].values[0] > cd.t[breath.end_time].values[0]

        start_indices, middle_indices, end_indices = (list(x) for x in zip(*breaths.values, strict=True))

        # Test whether breaths are sorted properly
        assert start_indices == sorted(start_indices)
        assert middle_indices == sorted(middle_indices)
        assert end_indices == sorted(end_indices)

        # Test whether indices are unique. `set` removes non-unique values,
        # `sorted(list(...))` converts the set to a sorted list again.
        assert list(start_indices) == sorted(set(start_indices))
        assert list(middle_indices) == sorted(set(middle_indices))
        assert list(end_indices) == sorted(set(end_indices))

        # Test whether the start of the next breath is on/after the previous breath
        assert all(
            start_index >= end_index for start_index, end_index in zip(start_indices[1:], end_indices[:-1], strict=True)
        )

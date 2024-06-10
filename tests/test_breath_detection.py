import copy
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


def make_cosine_wave(sample_frequency: float, length: int, frequency: float):
    time = np.arange(length) / sample_frequency
    return time, np.cos(time * np.pi * 2 * frequency)


def test_find_features():
    pytest.skip()


test_data_remove_edge_cases = [
    (  # test removal of edge peaks
        np.array([0, 1, 3, 5, 7, 9]),
        np.array([2, 4, 6, 8]),
        np.array([1] * 10),
        np.array([3, 5, 7]),
        np.array([2, 4, 6, 8]),
    ),
    (  # test removal of first valley
        np.array([1, 3, 5, 7, 9]),
        np.array([2, 4, 6, 8]),
        np.array([1, 1, 3, *[1] * 7]),
        np.array([5, 7]),
        np.array([4, 6, 8]),
    ),
    (  # test removal of last valley
        np.array([1, 3, 5, 7, 9]),
        np.array([2, 4, 6, 8]),
        np.array([*[1] * 8, 3, 1]),
        np.array([3, 5]),
        np.array([2, 4, 6]),
    ),
]


@pytest.mark.parametrize(
    ("peak_indices", "valley_indices", "moving_average", "expected_peak_indices", "expected_valley_indices"),
    test_data_remove_edge_cases,
)
def test_remove_edge_cases(
    peak_indices: np.ndarray,
    valley_indices: np.ndarray,
    moving_average: np.ndarray,
    expected_peak_indices: np.ndarray,
    expected_valley_indices: np.ndarray,
):
    data = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

    bd = BreathDetection(1)
    results = bd._remove_edge_cases(  # noqa: SLF001
        peak_indices,
        data[peak_indices],
        valley_indices,
        data[valley_indices],
        data,
        moving_average,
    )
    assert np.array_equal(results.peak_indices, expected_peak_indices)
    assert np.array_equal(results.valley_indices, expected_valley_indices)


test_data_remove_doubles = [
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
    (
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    ),
]


@pytest.mark.parametrize(
    (
        "peak_indices",
        "peak_values",
        "valley_indices",
        "valley_values",
        "expected_peak_indices",
        "expected_valley_indices",
    ),
    test_data_remove_doubles,
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


test_data_remove_low_amplitude = [
    (  # test remove none
        np.array([2, 4, 6, 8]),
        np.array([3, 3, 3, 3]),
        np.array([1, 3, 5, 7, 9]),
        np.array([1, 1, 1, 1, 1]),
        np.array([2, 4, 6, 8]),
        np.array([1, 3, 5, 7, 9]),
    ),
    (  # test remove first peak, keeping the first valley
        np.array([2, 4, 6, 8]),
        np.array([1.2, 3, 3, 3]),
        np.array([1, 3, 5, 7, 9]),
        np.array([1, 1.1, 1, 1, 1]),
        np.array([4, 6, 8]),
        np.array([1, 5, 7, 9]),
    ),
    (  # test remove first peak, keeping the second valley
        np.array([2, 4, 6, 8]),
        np.array([1.2, 3, 3, 3]),
        np.array([1, 3, 5, 7, 9]),
        np.array([1.1, 1, 1, 1, 1]),
        np.array([4, 6, 8]),
        np.array([3, 5, 7, 9]),
    ),
    (  # test remove second peak, keeping the second valley
        np.array([2, 4, 6, 8]),
        np.array([3, 1.2, 3, 3]),
        np.array([1, 3, 5, 7, 9]),
        np.array([1, 1, 1.1, 1, 1]),
        np.array([2, 6, 8]),
        np.array([1, 3, 7, 9]),
    ),
    (  # test remove second peak, keeping the third valley
        np.array([2, 4, 6, 8]),
        np.array([3, 1.2, 3, 3]),
        np.array([1, 3, 5, 7, 9]),
        np.array([1, 1.1, 1, 1, 1]),
        np.array([2, 6, 8]),
        np.array([1, 5, 7, 9]),
    ),
    (  # test remove last peak, keeping the second to last valley
        np.array([2, 4, 6, 8]),
        np.array([3, 3, 3, 1.2]),
        np.array([1, 3, 5, 7, 9]),
        np.array([1, 1, 1, 1, 1.1]),
        np.array([2, 4, 6]),
        np.array([1, 3, 5, 7]),
    ),
    (  # test remove last peak, keeping the last valley
        np.array([2, 4, 6, 8]),
        np.array([3, 3, 3, 1.2]),
        np.array([1, 3, 5, 7, 9]),
        np.array([1, 1, 1, 1.1, 1]),
        np.array([2, 4, 6]),
        np.array([1, 3, 5, 9]),
    ),
]


@pytest.mark.parametrize(
    (
        "peak_indices",
        "peak_values",
        "valley_indices",
        "valley_values",
        "expected_peak_indices",
        "expected_valley_indices",
    ),
    test_data_remove_low_amplitude,
)
def test_remove_low_amplitudes(
    peak_indices: np.ndarray,
    peak_values: np.ndarray,
    valley_indices: np.ndarray,
    valley_values: np.ndarray,
    expected_peak_indices: np.ndarray,
    expected_valley_indices: np.ndarray,
):
    bd = BreathDetection(1)
    results = bd._remove_low_amplitudes(peak_indices, peak_values, valley_indices, valley_values)  # noqa: SLF001
    assert np.array_equal(results.peak_indices, expected_peak_indices)
    assert np.array_equal(results.valley_indices, expected_valley_indices)


@pytest.mark.parametrize(
    (
        "peak_indices",
        "peak_values",
        "valley_indices",
        "valley_values",
        "expected_peak_indices",
        "expected_valley_indices",
    ),
    test_data_remove_low_amplitude,
)
def test_no_remove_low_amplitudes(
    peak_indices: np.ndarray,
    peak_values: np.ndarray,
    valley_indices: np.ndarray,
    valley_values: np.ndarray,
    expected_peak_indices: np.ndarray,
    expected_valley_indices: np.ndarray,
):
    """This test uses the same data as test_remove_low_amplitudes, but expects the output to be the same as the input."""
    bd = BreathDetection(1, amplitude_cutoff_fraction=None)
    results = bd._remove_low_amplitudes(peak_indices, peak_values, valley_indices, valley_values)  # noqa: SLF001
    assert np.array_equal(results.peak_indices, peak_indices)
    assert np.array_equal(results.valley_indices, valley_indices)

    bd = BreathDetection(1, amplitude_cutoff_fraction=0)
    results = bd._remove_low_amplitudes(peak_indices, peak_values, valley_indices, valley_values)  # noqa: SLF001
    assert np.array_equal(results.peak_indices, peak_indices)
    assert np.array_equal(results.valley_indices, valley_indices)


def test_remove_no_breaths_around_valid_data():
    sample_frequency = 20
    time, y = make_cosine_wave(sample_frequency, 1000, 1)
    bd = BreathDetection(sample_frequency)

    peak_valley_data = bd._detect_peaks_and_valleys(y)
    assert np.array_equal(time[peak_valley_data.peak_indices], np.arange(1, 50))
    assert np.array_equal(
        time[peak_valley_data.valley_indices],
        np.arange(0, 50) + 0.5,
    )


def test_remove_breaths_around_invalid_data():
    sample_frequency = 20
    time, y = make_cosine_wave(sample_frequency, 1000, 1)
    bd = BreathDetection(sample_frequency)

    peak_valley_data = bd._detect_peaks_and_valleys(y)

    start_invalid = np.argmax(time >= 3.25)
    end_invalid = np.argmax(time >= 3.75)
    y_with_invalid = np.copy(y)
    y_with_invalid[start_invalid:end_invalid] = -1000

    peak_valley_data2 = bd._detect_peaks_and_valleys(y_with_invalid)
    assert not np.array_equal(peak_valley_data.peak_indices, peak_valley_data2.peak_indices)
    assert not np.array_equal(peak_valley_data.valley_indices, peak_valley_data2.valley_indices)

    assert np.array_equal(time[peak_valley_data2.peak_indices], np.concatenate([[1], np.arange(5, 50)]))
    assert np.array_equal(
        time[peak_valley_data2.valley_indices],
        np.array([0.5, 1.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]),
    )


def test_with_data(draeger1: Sequence, draeger2: Sequence, timpel1: Sequence):
    # pytest.skip("Skip so formalized tests can cover 100%.")
    draeger1 = copy.deepcopy(draeger1)
    draeger2 = copy.deepcopy(draeger2)
    timpel1 = copy.deepcopy(timpel1)
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
